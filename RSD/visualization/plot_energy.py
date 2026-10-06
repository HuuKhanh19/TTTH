# visualization/plot_energy.py

import matplotlib.pyplot as plt
import numpy as np


class PlotEnergy:
    """
    Generates energy consumption plot.
    """

    def __init__(self):
        pass

    def compute_energy_curve(self, history_energy):
        """
        Compute cumulative energy consumption over time.
        """

        energy_curve = []

        initial_total = np.sum(history_energy[0])

        for energy in history_energy:
            current_total = np.sum(energy)
            consumed = initial_total - current_total
            energy_curve.append(consumed)

        return energy_curve

    def plot(self, results_dict):
        """
        Plot energy consumption for all methods.

        Parameters:
        ----------
        results_dict : dict
            {method_name: history}
        """

        plt.figure(figsize=(8, 5))

        for method, history in results_dict.items():
            energy_curve = self.compute_energy_curve(history["energy"])
            plt.plot(energy_curve, linewidth=2, label=method.capitalize())

        plt.xlabel("Time Steps")
        plt.ylabel("Energy Consumed")
        plt.title("Energy Consumption vs Time")

        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("energy_plot.png", dpi=600)
        plt.show()