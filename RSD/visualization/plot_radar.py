# visualization/plot_radar.py

import numpy as np
import matplotlib.pyplot as plt


class PlotRadar:
    """
    Generates radar chart for multi-metric comparison.
    """

    def __init__(self):
        self.metrics = [
            "coverage",
            "mission_time",
            "energy",
            "path_length",
            "connectivity"
        ]

    # -----------------------------
    # Normalize Metrics
    # -----------------------------
    def normalize(self, results_dict):
        """
        Normalize metrics across methods.
        """

        normalized = {}

        # Collect values
        values = {metric: [] for metric in self.metrics}

        for method, res in results_dict.items():
            for m in self.metrics:
                values[m].append(res[m])

        for method, res in results_dict.items():

            normalized[method] = []

            for m in self.metrics:

                val = res[m]
                max_val = max(values[m])
                min_val = min(values[m])

                # Normalize to [0,1]
                if m in ["coverage", "connectivity"]:
                    norm = (val - min_val) / (max_val - min_val + 1e-6)
                else:
                    norm = (max_val - val) / (max_val - min_val + 1e-6)

                normalized[method].append(norm)

        return normalized

    # -----------------------------
    # Plot Radar
    # -----------------------------
    def plot(self, results_dict):
        """
        Plot radar chart.
        """

        normalized = self.normalize(results_dict)

        labels = self.metrics
        num_vars = len(labels)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))

        plt.figure(figsize=(7, 7))
        ax = plt.subplot(111, polar=True)

        for method, values in normalized.items():
            values = np.array(values)
            values = np.concatenate((values, [values[0]]))

            ax.plot(angles, values, linewidth=2, label=method.capitalize())

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)

        plt.title("Normalized Multi-Metric Performance Comparison")

        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig("radar_plot.png", dpi=600)
        plt.show()