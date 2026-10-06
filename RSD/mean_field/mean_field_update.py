# mean_field/mean_field_update.py

import numpy as np


class MeanField:
    """
    Implements mean-field distribution and cost approximation.

    Equation References:
    --------------------
    Eq (22): Empirical distribution
    Eq (24): Distribution evolution
    Eq (25): Mean-field cost
    """

    def __init__(self, Q, R):
        """
        Initialize mean-field module.

        Parameters:
        ----------
        Q : np.ndarray
            State cost matrix
        R : np.ndarray
            Control cost matrix
        """
        self.Q = Q
        self.R = R

    # -----------------------------
    # Eq (22): Empirical Distribution
    # -----------------------------
    def compute_empirical_distribution(self, uav_states):
        """
        Approximate empirical distribution using samples.

        Returns:
        -------
        mean : np.ndarray
        covariance : np.ndarray
        """

        X = np.array(uav_states)

        mean = np.mean(X, axis=0)
        covariance = np.cov(X.T)

        return mean, covariance

    # -----------------------------
    # Eq (24): Distribution Evolution
    # -----------------------------
    def propagate_distribution(self, uav_states, A, B, K):
        """
        Propagate distribution via closed-loop dynamics.

        x_{k+1} = (A - B K) x_k + noise (handled elsewhere)
        """

        A_cl = A - B @ K

        next_states = []

        for x in uav_states:
            x_next = A_cl @ x
            next_states.append(x_next)

        return next_states

    # -----------------------------
    # Eq (25): Mean-Field Cost
    # -----------------------------
    def compute_mean_field_cost(self, uav_states, controls):
        """
        Compute mean-field stage cost.

        ℓ_k^MF = E[x^T Q x] + E[u^T R u]
        """

        state_cost = 0.0
        control_cost = 0.0

        for x in uav_states:
            state_cost += x.T @ self.Q @ x

        for u in controls:
            control_cost += u.T @ self.R @ u

        # Normalize by number of UAVs
        N = len(uav_states)

        state_cost /= N
        control_cost /= N

        return state_cost + control_cost