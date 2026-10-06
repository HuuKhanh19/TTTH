# evaluation/summary_table.py

import pandas as pd
from pathlib import Path


class SummaryTable:
    """
    Generates Overall Statistical Performance Summary Table.
    """

    def __init__(self):
        pass

    # -----------------------------
    # Compute Improvement
    # -----------------------------
    def compute_improvement(self, metric, proposed, baseline, higher_is_better=True):
        """
        Compute % improvement.

        Parameters:
        ----------
        higher_is_better : bool
        """

        if higher_is_better:
            return (proposed - baseline) / baseline * 100
        else:
            return (baseline - proposed) / baseline * 100

    # -----------------------------
    # Build Table
    # -----------------------------
    def generate(self, results_dict):
        """
        results_dict format:
        {
            "proposed": {...},
            "adam": {...},
            "javed": {...},
            "alawad": {...}
        }
        """

        table = []

        metrics_info = {
            "coverage": True,
            "mission_time": False,
            "energy": False,
            "path_length": False,
            "connectivity": True
        }

        for metric, higher_is_better in metrics_info.items():

            proposed_value = results_dict["proposed"][metric]

            # Find best baseline
            baseline_values = {
                k: v[metric] for k, v in results_dict.items() if k != "proposed"
            }

            if higher_is_better:
                best_baseline_method = max(baseline_values, key=baseline_values.get)
            else:
                best_baseline_method = min(baseline_values, key=baseline_values.get)

            best_baseline_value = baseline_values[best_baseline_method]

            improvement = self.compute_improvement(
                metric,
                proposed_value,
                best_baseline_value,
                higher_is_better
            )

            table.append({
                "Metric": metric,
                "Proposed": proposed_value,
                "Best Baseline": f"{best_baseline_method} ({best_baseline_value:.2f})",
                "Improvement (%)": improvement
            })

        df = pd.DataFrame(table)
        return df

    # -----------------------------
    # Save Table
    # -----------------------------
    def save(self, df, path="data/tables/overall_summary.csv"):
        """
        Save table to CSV.
        """

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
