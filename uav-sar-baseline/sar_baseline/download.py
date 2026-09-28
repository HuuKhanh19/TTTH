"""Fetch immutable upstream assets; verify heatmaps against Git blob hashes.

Only numeric maps and the GeoJSON metadata suffix are needed for likelihood
scoring. Full OSM feature geometry is deliberately not downloaded. The saved
metadata is an extraction, not a substitute original GeoJSON dataset.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

REVISION = "0f2d344d17c0b9c18cec2176dd9b171beba727d6"
REPO = "namurproject/SAREnv"
RAW = f"https://raw.githubusercontent.com/{REPO}/{REVISION}"
MEDIA = f"https://media.githubusercontent.com/media/{REPO}/{REVISION}"

def fetch(url: str, headers=None) -> tuple[bytes, dict]:
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers=headers or {}), timeout=90) as r:
                return r.read(), dict(r.headers)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)
    raise RuntimeError("unreachable")

def sha256(data):
    return hashlib.sha256(data).hexdigest()

def git_blob_sha(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()

def prepare_one(map_id, root, tree, local_upstream=None):
    folder = root / str(map_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = f"sarenv_dataset/{map_id}/heatmap.npy"
    expected = tree[path]
    target = folder / "heatmap.npy"
    if target.exists() and git_blob_sha(target.read_bytes()) == expected["sha"]:
        content = target.read_bytes()
    else:
        cached = Path(local_upstream) / path if local_upstream else None
        if cached and cached.exists():
            content = cached.read_bytes()
        else:
            content, _ = fetch(f"{RAW}/{path}")
        if len(content) != expected["size"] or git_blob_sha(content) != expected["sha"]:
            raise ValueError(f"Invalid heatmap Git hash/size: {path}")
        tmp = target.with_suffix(".part")
        tmp.write_bytes(content)
        tmp.replace(target)
    manifest_path = folder / "provenance.json"
    if manifest_path.exists() and (folder / "metadata.json").exists():
        prior = json.loads(manifest_path.read_text())
        if (prior.get("revision") == REVISION and prior.get("heatmap_sha256") == sha256(content)
                and prior.get("metadata_sha256") == sha256((folder / "metadata.json").read_bytes())):
            return map_id
    geo_path = f"sarenv_dataset/{map_id}/features.geojson"
    pointer, _ = fetch(f"{RAW}/{geo_path}")
    if git_blob_sha(pointer) != tree[geo_path]["sha"]:
        raise ValueError("LFS pointer checksum mismatch")
    match = re.search(rb"oid sha256:(\w+)\nsize (\d+)", pointer)
    if not match:
        raise ValueError("Expected upstream Git LFS pointer")
    oid, size = match[1].decode(), int(match[2])
    start = max(0, size-4096)
    suffix, response = fetch(f"{MEDIA}/{geo_path}", {"Range": f"bytes={start}-{size-1}"})
    if len(suffix) != size-start or response.get("Content-Range", "") != f"bytes {start}-{size-1}/{size}":
        raise ValueError("Server did not supply the requested metadata byte range")
    if response.get("ETag", "").strip('"') != oid:
        raise ValueError("GeoJSON ETag does not match LFS SHA256")
    marker = b'"environment_type":'
    offset = suffix.rfind(marker)
    if offset < 0:
        raise ValueError("Metadata not found in the final 4096 bytes")
    metadata = json.loads(b"{" + suffix[offset:])
    required = {"environment_type", "climate", "center_point", "meter_per_bin", "radius_km", "bounds"}
    if not required.issubset(metadata):
        raise ValueError("Incomplete metadata")
    metadata_bytes = (json.dumps(metadata, indent=2) + "\n").encode()
    (folder / "metadata.json").write_bytes(metadata_bytes)
    (folder / "features.metadata-suffix.bin").write_bytes(suffix)
    manifest = {
        "dataset": "SAREnv", "map_id": map_id, "revision": REVISION,
        "paper_doi": "10.3390/drones9090628", "heatmap_url": f"{RAW}/{path}",
        "heatmap_git_blob_sha1": expected["sha"], "heatmap_sha256": sha256(content),
        "heatmap_bytes": len(content), "geojson_url": f"{MEDIA}/{geo_path}",
        "geojson_lfs_sha256": oid, "geojson_full_bytes": size,
        "metadata_range": [start, size-1], "metadata_suffix_sha256": sha256(suffix),
        "metadata_sha256": sha256(metadata_bytes),
        "note": "Heatmap is byte-identical upstream data. Metadata extracted from LFS suffix; full geometry not downloaded."
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return map_id

def parse_ids(value):
    ids = []
    for part in value.split(','):
        if '-' in part:
            first, last = map(int, part.split('-'))
            ids.extend(range(first, last+1))
        else:
            ids.append(int(part))
    if not ids or any(i < 1 or i > 60 for i in ids):
        raise ValueError("SAREnv IDs must be between 1 and 60")
    return sorted(set(ids))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ids', default='1-60')
    parser.add_argument('--data-dir', type=Path, default=Path('data/sarenv'))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--local-upstream')
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.data_dir / 'upstream-tree.json'
    if lock_path.exists():
        document = json.loads(lock_path.read_text())
    else:
        blob, _ = fetch(f'https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1')
        document = json.loads(blob)
        lock_path.write_bytes(blob)
    if document.get('sha') != REVISION or document.get('truncated'):
        raise ValueError('Unexpected upstream tree')
    tree = {e['path']: e for e in document['tree']}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done in pool.map(lambda i: prepare_one(i, args.data_dir, tree, args.local_upstream), parse_ids(args.ids)):
            print(f'verified SAREnv map {done}', flush=True)

if __name__ == '__main__':
    main()
