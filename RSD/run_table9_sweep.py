"""Run the paper's risk-parameter sweep in RSD's one-dimensional model."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


THETAS = [(0.0, "theta_0"), (0.02, "theta_0_02"),
          (0.05, "theta_0_05"), (0.1, "theta_0_1")]
METRICS = ["coverage", "mission_time", "energy", "path_length", "connectivity"]


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    output_arg = args.output or Path(f"results/paper_table9_1d_seed{args.seed}")
    output = output_arg if output_arg.is_absolute() else repo / output_arg
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    all_rows = []
    proposed_rows = []
    for theta, label in THETAS:
        run_dir = output / label
        run_dir.mkdir(exist_ok=True)
        command = [sys.executable, "-u", str(repo / "main.py"),
                   "--profile", "paper-table9-1d", "--theta", str(theta),
                   "--seed", str(args.seed)]
        completed = subprocess.run(command, cwd=run_dir, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, check=False)
        (run_dir / "run.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"theta={theta} failed; see {run_dir / 'run.log'}")

        with (run_dir / "data/tables/method_metrics.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if [row["method"] for row in rows] != ["proposed", "adam", "javed", "alawad"]:
            raise RuntimeError(f"Unexpected methods for theta={theta}")
        all_rows.extend({"theta": theta, **row} for row in rows)

        manifest = json.loads((run_dir / "data/tables/run_config.json").read_text())
        diagnostics = manifest["diagnostics"]
        proposed_rows.append({"theta": theta, **{key: rows[0][key] for key in METRICS},
                              **diagnostics})
        print(f"theta={theta}: coverage={rows[0]['coverage']}%, "
              f"admissibility={diagnostics['minimum_admissibility_eigenvalue']:.4f}")

    write_csv(output / "all_methods.csv", all_rows, ["theta", "method", *METRICS])
    write_csv(output / "proposed_theta_sweep.csv", proposed_rows,
              ["theta", *METRICS, "minimum_admissibility_eigenvalue",
               "maximum_closed_loop_spectral_radius"])
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
