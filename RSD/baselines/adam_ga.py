# baselines/adam_ga.py

import numpy as np


class AdamGABaseline:
    """
    Adam et al. inspired heuristic control (GA-like).

    Characteristics:
    ----------------
    - Gradient-free
    - Sampling-based
    - Exploration + exploitation
    """

    def __init__(self, num_candidates=10, control_limit=1.0):
        """
        Initialize baseline.

        Parameters:
        ----------
        num_candidates : int
            Number of sampled control actions
        control_limit : float
            Maximum control magnitude
        """
        self.num_candidates = num_candidates
        self.control_limit = control_limit

    # -----------------------------
    # Generate candidate controls
    # -----------------------------
    def sample_controls(self, dim):
        """
        Sample random control vectors.
        """

        controls = []

        for _ in range(self.num_candidates):
            u = np.random.uniform(
                low=-self.control_limit,
                high=self.control_limit,
                size=dim
            )
            controls.append(u)

        return controls

    # -----------------------------
    # Fitness Function
    # -----------------------------
    def evaluate_control(self, x, u, disaster_field):
        """
        Evaluate candidate control.

        Objective:
        - move toward high-intensity regions
        - penalize large movement
        """

        # Predict next position (simplified)
        x_next = x + u

        index = int(np.clip(x_next[0], 0, len(disaster_field) - 1))

        # Reward high disaster intensity
        reward = disaster_field[index]

        # Penalize control effort
        penalty = 0.1 * np.linalg.norm(u)**2

        fitness = reward - penalty

        return fitness

    # -----------------------------
    # Select Best Control
    # -----------------------------
    def compute_control(self, x, disaster_field):
        """
        Compute control using GA-like selection.
        """

        dim = len(x)

        candidates = self.sample_controls(dim)

        best_u = None
        best_score = -np.inf

        for u in candidates:
            score = self.evaluate_control(x, u, disaster_field)

            if score > best_score:
                best_score = score
                best_u = u

        return best_u

    # -----------------------------
    # Multi-UAV Control
    # -----------------------------
    def compute_multi_uav_control(self, uav_states, disaster_field):
        """
        Apply control to all UAVs.
        """

        controls = []

        for x in uav_states:
            u = self.compute_control(x, disaster_field)
            controls.append(u)

        return controls