# visualization/plot_robustness.py

import matplotlib.pyplot as plt
import numpy as np


class PlotRobustness:
    """
    Generates robustness plot under varying noise.
    """

    def __init__(self, simulator, metrics, graph):
        """
        Parameters:
        ----------
        simulator : Simulator instance
        metrics : Metrics instance
        graph : CommunicationGraph instance
        """
        self.simulator = simulator
        self.metrics = metrics
        self.graph = graph

    def evaluate_under_noise(self, method, noise_levels):
        """
        Run simulations for different noise levels.
        """

        performance = []

        for sigma in noise_levels:

            # Update noise level
            self.simulator.uav_dynamics.Sigma = sigma * np.eye(
                self.simulator.uav_dynamics.state_dim
            )

            history = self.simulator.run(method=method)
            results = self.metrics.compute_all(history, self.graph)

            # Use coverage as performance metric
            performance.append(results["coverage"])

        return performance

    def plot(self, noise_levels):
        """
        Plot robustness curves.
        """

        plt.figure(figsize=(8, 5))

        methods = ["proposed", "adam", "javed", "alawad"]

        for method in methods:
            perf = self.evaluate_under_noise(method, noise_levels)
            plt.plot(noise_levels, perf, linewidth=2, label=method.capitalize())

        plt.xlabel("Noise Level (σ)")
        plt.ylabel("Coverage (%)")
        plt.title("Robustness under Increasing Noise")

        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("robustness_plot.png", dpi=600)
        plt.show()