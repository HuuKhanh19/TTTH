# mean_field/covariance_update.py

import numpy as np


class MeanFieldCovariance:
    """
    Implements mean-field covariance evolution.

    Equation Reference:
    -------------------
    Eq (29): Σ_{k+1} = (A - B K) Σ_k (A - B K)^T + W
    """

    def __init__(self, A, B, W):
        """
        Initialize covariance update.

        Parameters:
        ----------
        A : np.ndarray
            System matrix
        B : np.ndarray
            Control matrix
        W : np.ndarray
            Process noise covariance
        """
        self.A = A
        self.B = B
        self.W = W

    def update(self, Sigma_k, K_k):
        """
        Perform one covariance update step.

        Parameters:
        ----------
        Sigma_k : np.ndarray
            Current covariance matrix
        K_k : np.ndarray
            Control gain matrix

        Returns:
        -------
        Sigma_next : np.ndarray
            Updated covariance matrix
        """

        # Closed-loop matrix
        A_cl = self.A - self.B @ K_k

        # Eq (29)
        Sigma_next = A_cl @ Sigma_k @ A_cl.T + self.W

        return Sigma_next

    def initialize(self, dim, scale=1.0):
        """
        Initialize covariance matrix.

        Parameters:
        ----------
        dim : int
            State dimension
        scale : float
            Initial variance scale

        Returns:
        -------
        Sigma_0 : np.ndarray
        """

        return scale * np.eye(dim)

    def compute_error_norm(self, Sigma_k, Sigma_inf):
        """
        Compute convergence metric.

        Used for:
        Fig 5 (Mean-field convergence)

        ||Σ_k - Σ_inf||_F
        """

        return np.linalg.norm(Sigma_k - Sigma_inf, ord='fro')