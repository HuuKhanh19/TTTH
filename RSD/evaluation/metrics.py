# evaluation/metrics.py

import numpy as np


class Metrics:
    """
    Computes performance metrics for UAV swarm.
    """

    def __init__(self, grid_size):
        self.grid_size = grid_size

    # -----------------------------
    # Coverage
    # -----------------------------
    def compute_coverage(self, history_states):
        """
        Percentage of grid explored.

        Simplified:
        count visited positions.
        """

        visited = set()

        for states in history_states:
            for x in states:
                idx = int(np.clip(x[0], 0, self.grid_size - 1))
                visited.add(idx)

        coverage = len(visited) / self.grid_size * 100
        return coverage

    # -----------------------------
    # Mission Completion Time
    # -----------------------------
    def compute_mission_time(self, threshold, history_states):
        """
        Time until sufficient coverage achieved.
        """

        visited = set()

        for t, states in enumerate(history_states):
            for x in states:
                idx = int(np.clip(x[0], 0, self.grid_size - 1))
                visited.add(idx)

            if len(visited) / self.grid_size >= threshold:
                return t

        return len(history_states)

    # -----------------------------
    # Energy Consumption
    # -----------------------------
    def compute_energy(self, history_energy):
        """
        Total energy consumed.
        """

        initial_energy = np.sum(history_energy[0])
        final_energy = np.sum(history_energy[-1])

        return initial_energy - final_energy

    # -----------------------------
    # Path Length
    # -----------------------------
    def compute_path_length(self, history_states):
        """
        Total movement distance.
        """

        total_distance = 0.0

        for t in range(1, len(history_states)):
            prev = history_states[t - 1]
            curr = history_states[t]

            for i in range(len(prev)):
                total_distance += np.linalg.norm(curr[i] - prev[i])

        return total_distance

    # -----------------------------
    # Connectivity
    # -----------------------------
    def compute_connectivity(self, graph, history_states):
        """
        Average connectivity ratio.
        """

        connected_count = 0

        for states in history_states:
            A = graph.compute_adjacency(states)
            if graph.is_connected(A):
                connected_count += 1

        return connected_count / len(history_states)

    # -----------------------------
    # Robustness
    # -----------------------------
    def compute_robustness(self, baseline_metric, noisy_metric):
        """
        Degradation under noise.
        """

        degradation = abs(noisy_metric - baseline_metric) / baseline_metric * 100
        return degradation

    # -----------------------------
    # Aggregate All Metrics
    # -----------------------------
    def compute_all(self, history, graph):
        """
        Compute all metrics together.
        """

        results = {}

        results["coverage"] = self.compute_coverage(history["states"])
        results["mission_time"] = self.compute_mission_time(0.9, history["states"])
        results["energy"] = self.compute_energy(history["energy"])
        results["path_length"] = self.compute_path_length(history["states"])
        results["connectivity"] = self.compute_connectivity(graph, history["states"])

        return results