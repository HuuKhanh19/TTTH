"""Numeric equivalent of SAREnv's loader: identical crop and no renormalization."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from pyproj import Transformer
from .download import REVISION, sha256

RADII_KM = {
    ('flat', 'temperate'): (0.6, 1.8, 3.2, 9.9),
    ('flat', 'dry'): (1.3, 2.1, 6.6, 13.1),
    ('mountainous', 'temperate'): (1.1, 3.1, 5.8, 18.3),
    ('mountainous', 'dry'): (1.6, 3.2, 6.5, 19.3),
}

@dataclass
class Environment:
    map_id: int
    heatmap: np.ndarray
    bounds: tuple
    center: tuple
    radius: float
    meters_per_bin: float
    terrain: str
    climate: str
    size: str

    @property
    def mass(self):
        return float(self.heatmap.sum())

    @property
    def cell_size(self):
        x0, y0, x1, y1 = self.bounds
        h, w = self.heatmap.shape
        return (x1-x0)/w, (y1-y0)/h


def load_environment(root, map_id, size='medium', verify=True):
    folder = Path(root) / str(map_id)
    metadata_bytes = (folder / 'metadata.json').read_bytes()
    metadata = json.loads(metadata_bytes)
    if verify:
        manifest = json.loads((folder / 'provenance.json').read_text())
        if manifest['revision'] != REVISION:
            raise ValueError('Unexpected dataset revision')
        if sha256((folder / 'heatmap.npy').read_bytes()) != manifest['heatmap_sha256']:
            raise ValueError('Heatmap checksum mismatch')
        if sha256(metadata_bytes) != manifest['metadata_sha256']:
            raise ValueError('Metadata checksum mismatch')
    master = np.load(folder / 'heatmap.npy', allow_pickle=False)
    if master.ndim != 2 or not np.isfinite(master).all() or np.any(master < 0):
        raise ValueError('Invalid probability heatmap')
    radius = 1000 * RADII_KM[(metadata['environment_type'], metadata['climate'])][
        ('small','medium','large','xlarge').index(size)]
    lon, lat = metadata['center_point']
    epsg = (32600 if lat >= 0 else 32700) + int((lon+180)/6) + 1
    cx, cy = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True).transform(lon,lat)
    bounds = (cx-radius, cy-radius, cx+radius, cy+radius)
    bx, by = metadata['bounds'][:2]
    res = metadata['meter_per_bin']
    x0, x1 = np.clip(np.array([(bounds[0]-bx)/res, (bounds[2]-bx)/res]).astype(int),0,master.shape[1])
    y0, y1 = np.clip(np.array([(bounds[1]-by)/res, (bounds[3]-by)/res]).astype(int),0,master.shape[0])
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (xx-(cx-bx)/res)**2 + (yy-(cy-by)/res)**2 <= (radius/res)**2
    heatmap = np.where(mask, master[y0:y1,x0:x1], 0)
    if heatmap.size == 0 or heatmap.sum() <= 0:
        raise ValueError('Empty search region')
    return Environment(map_id,heatmap,bounds,(cx,cy),radius,res,
                       metadata['environment_type'],metadata['climate'],size)
