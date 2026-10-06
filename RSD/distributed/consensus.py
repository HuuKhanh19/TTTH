# distributed/consensus.py

import numpy as np


class Consensus:
    """
    Implements distributed consensus update.

    Equation Reference:
    -------------------
    Eq (34): Consensus update
    """

    def __init__(self, gamma):
        """
        Initialize consensus module.

        Parameters:
        ----------
        gamma : float
            Consensus step size
        """
        self.gamma = gamma

    # -----------------------------
    # Consensus Update
    # -----------------------------
    def update(self, Sigma_list, adjacency, innovation=None):
        """
        Perform one consensus step.

        Parameters:
        ----------
        Sigma_list : list of np.ndarray
            Each UAV's covariance estimate
        adjacency : np.ndarray
            Adjacency matrix
        innovation : list (optional)
            Local innovation terms Φ_i

        Returns:
        -------
        updated_Sigma_list : list
        """

        N = len(Sigma_list)
        updated = []

        for i in range(N):

            Sigma_i = Sigma_list[i]
            consensus_term = np.zeros_like(Sigma_i)

            # Sum over neighbors
            for j in range(N):
                if adjacency[i, j] == 1:
                    Sigma_j = Sigma_list[j]
                    consensus_term += (Sigma_i - Sigma_j)

            # Eq (34)
            Sigma_next = Sigma_i - self.gamma * consensus_term

            # Add innovation if provided
            if innovation is not None:
                Sigma_next += innovation[i]

            updated.append(Sigma_next)

        return updated

    # -----------------------------
    # Convergence Metric
    # -----------------------------
    def compute_consensus_error(self, Sigma_list):
        """
        Measure how far UAVs are from consensus.

        Returns:
        -------
        error : float
        """

        N = len(Sigma_list)
        mean_sigma = sum(Sigma_list) / N

        error = 0.0

        for Sigma in Sigma_list:
            error += np.linalg.norm(Sigma - mean_sigma, ord='fro')

        return error / N