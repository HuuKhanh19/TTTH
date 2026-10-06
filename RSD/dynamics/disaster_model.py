# dynamics/disaster_model.py

import numpy as np


class DisasterModel:
    """
    Implements the stochastic disaster propagation model.

    Equation Reference:
    -------------------
    Eq (1): D_{k+1} = A D_k + w_k

    where:
        D_k : disaster intensity vector (M-dimensional)
        A   : spatial propagation matrix
        w_k : Gaussian noise ~ N(0, Q)
    """

    def __init__(self, A, Q):
        """
        Initialize disaster model.

        Parameters:
        ----------
        A : np.ndarray
            State transition matrix (M x M)
        Q : np.ndarray
            Noise covariance matrix (M x M)
        """
        self.A = A
        self.Q = Q
        self.dim = A.shape[0]

    def sample_noise(self):
        """
        Sample Gaussian disturbance.

        Returns:
        -------
        w : np.ndarray
            Noise vector sampled from N(0, Q)
        """
        return np.random.multivariate_normal(
            mean=np.zeros(self.dim),
            cov=self.Q
        )

    def step(self, D_k):
        """
        Propagate disaster state forward by one time step.

        Parameters:
        ----------
        D_k : np.ndarray
            Current disaster intensity (M,)

        Returns:
        -------
        D_next : np.ndarray
            Next disaster state (M,)
        """

        # Eq (1): D_{k+1} = A D_k + w_k
        w_k = self.sample_noise()
        D_next = self.A @ D_k + w_k

        return D_next

    def initialize_state(self, M, mode="random"):
        """
        Initialize disaster field.

        Parameters:
        ----------
        M : int
            Number of spatial cells
        mode : str
            'random' or 'localized'

        Returns:
        -------
        D0 : np.ndarray
            Initial disaster state
        """

        if mode == "random":
            return np.random.rand(M)

        elif mode == "localized":
            D0 = np.zeros(M)
            center = np.random.randint(0, M)
            D0[center] = 1.0
            return D0

        else:
            raise ValueError("Invalid initialization mode")