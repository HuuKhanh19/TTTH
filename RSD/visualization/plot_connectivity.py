# visualization/plot_connectivity.py

import matplotlib.pyplot as plt
import numpy as np


class PlotConnectivity:
    """
    Generates connectivity plot.
    """

    def __init__(self, graph):
        """
        Parameters:
        ----------
        graph : CommunicationGraph instance
        """
        self.graph = graph

    def compute_connectivity_curve(self, history_states):
        """
        Compute connectivity ratio over time.
        """

        connectivity_curve = []

        for states in history_states:
            A = self.graph.compute_adjacency(states)

            # Connectivity measure: 1 if connected, else 0
            is_conn = 1 if self.graph.is_connected(A) else 0
            connectivity_curve.append(is_conn)

        return connectivity_curve

    def plot(self, results_dict):
        """
        Plot connectivity for all methods.
        """

        plt.figure(figsize=(8, 5))

        for method, history in results_dict.items():
            conn_curve = self.compute_connectivity_curve(history["states"])
            plt.plot(conn_curve, linewidth=2, label=method.capitalize())

        plt.xlabel("Time Steps")
        plt.ylabel("Connectivity (0/1)")
        plt.title("Network Connectivity vs Time")

        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("connectivity_plot.png", dpi=600)
        plt.show()