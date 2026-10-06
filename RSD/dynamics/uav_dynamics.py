# dynamics/uav_dynamics.py

import numpy as np


class UAVDynamics:
    """
    Implements stochastic UAV dynamics and energy model.

    Equation References:
    --------------------
    Eq (2): x_{i,k+1} = f(x_{i,k}, u_{i,k}) + ξ_{i,k}
    Eq (3): E_{i,k+1} = E_{i,k} - α||u||^2 - β
    """

    def __init__(self, A, B, Sigma, alpha, beta, E_min):
        """
        Initialize UAV dynamics.

        Parameters:
        ----------
        A : np.ndarray
            State transition matrix
        B : np.ndarray
            Control input matrix
        Sigma : np.ndarray
            Process noise covariance
        alpha : float
            Control energy coefficient
        beta : float
            Baseline energy consumption
        E_min : float
            Minimum energy threshold
        """
        self.A = A
        self.B = B
        self.Sigma = Sigma
        self.alpha = alpha
        self.beta = beta
        self.E_min = E_min

        self.state_dim = A.shape[0]

    def sample_noise(self):
        """
        Sample process noise ξ ~ N(0, Σ)
        """
        return np.random.multivariate_normal(
            mean=np.zeros(self.state_dim),
            cov=self.Sigma
        )

    def step_state(self, x_k, u_k):
        """
        Update UAV state.

        Parameters:
        ----------
        x_k : np.ndarray
            Current state
        u_k : np.ndarray
            Control input

        Returns:
        -------
        x_next : np.ndarray
            Next state
        """

        # Eq (2): x_{k+1} = A x_k + B u_k + ξ_k
        noise = self.sample_noise()
        x_next = self.A @ x_k + self.B @ u_k + noise

        return x_next

    def step_energy(self, E_k, u_k):
        """
        Update UAV energy.

        Parameters:
        ----------
        E_k : float
            Current energy
        u_k : np.ndarray
            Control input

        Returns:
        -------
        E_next : float
            Updated energy
        """

        # Eq (3): E_{k+1} = E_k - α||u||^2 - β
        energy_consumption = self.alpha * np.linalg.norm(u_k)**2 + self.beta
        E_next = E_k - energy_consumption

        # Enforce constraint: E >= Emin
        E_next = max(E_next, self.E_min)

        return E_next

    def initialize_state(self, dim, mode="random"):
        """
        Initialize UAV state.

        Parameters:
        ----------
        dim : int
            State dimension
        mode : str
            'random' or 'zero'

        Returns:
        -------
        x0 : np.ndarray
        """

        if mode == "random":
            return np.random.randn(dim)

        elif mode == "zero":
            return np.zeros(dim)

        else:
            raise ValueError("Invalid initialization mode")

    def initialize_energy(self, E0):
        """
        Initialize UAV energy.

        Parameters:
        ----------
        E0 : float
            Initial energy

        Returns:
        -------
        float
        """
        return E0