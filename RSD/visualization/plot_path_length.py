# visualization/plot_path_length.py

import matplotlib.pyplot as plt
import numpy as np


class PlotPathLength:
    """
    Generates path length plot.
    """

    def __init__(self):
        pass

    def compute_path_curve(self, history_states):
        """
        Compute cumulative path length over time.
        """

        path_curve = []
        total_distance = 0.0

        for t in range(1, len(history_states)):

            prev = history_states[t - 1]
            curr = history_states[t]

            step_distance = 0.0

            for i in range(len(prev)):
                step_distance += np.linalg.norm(curr[i] - prev[i])

            total_distance += step_distance
            path_curve.append(total_distance)

        return path_curve

    def plot(self, results_dict):
        """
        Plot path length for all methods.
        """

        plt.figure(figsize=(8, 5))

        for method, history in results_dict.items():
            path_curve = self.compute_path_curve(history["states"])
            plt.plot(path_curve, linewidth=2, label=method.capitalize())

        plt.xlabel("Time Steps")
        plt.ylabel("Cumulative Path Length")
        plt.title("Path Length vs Time")

        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("path_length_plot.png", dpi=600)
        plt.show()