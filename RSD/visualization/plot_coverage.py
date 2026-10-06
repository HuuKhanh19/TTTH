# visualization/plot_coverage.py

import matplotlib.pyplot as plt
import numpy as np


class PlotCoverage:
    """
    Generates coverage vs time plot.
    """

    def __init__(self, grid_size):
        self.grid_size = grid_size

    def compute_coverage_over_time(self, history_states):
        """
        Compute coverage progression over time.
        """

        visited = set()
        coverage_curve = []

        for states in history_states:

            for x in states:
                idx = int(np.clip(x[0], 0, self.grid_size - 1))
                visited.add(idx)

            coverage = len(visited) / self.grid_size * 100
            coverage_curve.append(coverage)

        return coverage_curve

    def plot(self, results_dict):
        """
        Plot coverage for all methods.

        Parameters:
        ----------
        results_dict : dict
            {method_name: history}
        """

        plt.figure(figsize=(8, 5))

        for method, history in results_dict.items():
            coverage_curve = self.compute_coverage_over_time(history["states"])
            plt.plot(coverage_curve, linewidth=2, label=method.capitalize())

        plt.xlabel("Time Steps")
        plt.ylabel("Coverage (%)")
        plt.title("Coverage vs Time")

        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("coverage_plot.png", dpi=600)
        plt.show()