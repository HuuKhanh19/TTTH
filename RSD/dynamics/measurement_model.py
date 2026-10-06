# dynamics/measurement_model.py

import numpy as np


class MeasurementModel:
    """
    Implements UAV sensing of the disaster field.

    Equation Reference:
    -------------------
    Eq (5): z_{i,k} = H_i(D_k, x_{i,k}) + v_{i,k}

    where:
        z_{i,k} : measurement
        H_i(.)  : observation function
        v_{i,k} : Gaussian noise ~ N(0, R)
    """

    def __init__(self, R, sensing_radius=1.0):
        """
        Initialize measurement model.

        Parameters:
        ----------
        R : np.ndarray
            Measurement noise covariance
        sensing_radius : float
            Radius within which UAV senses disaster
        """
        self.R = R
        self.sensing_radius = sensing_radius

    def sample_noise(self, dim):
        """
        Sample measurement noise.

        Returns:
        -------
        noise : np.ndarray
        """
        return np.random.multivariate_normal(
            mean=np.zeros(dim),
            cov=self.R
        )

    def H(self, D_k, x_k):
        """
        Observation function.

        Simplified model:
        UAV senses average disaster intensity in its vicinity.

        Parameters:
        ----------
        D_k : np.ndarray
            Disaster field (M,)
        x_k : np.ndarray
            UAV state (assume position in first 2 dims)

        Returns:
        -------
        z : np.ndarray
            Measurement vector
        """

        # Assume 1D grid or flattened field
        position_index = int(np.clip(x_k[0], 0, len(D_k) - 1))

        # Local sensing (can be extended to radius-based neighborhood)
        z = np.array([D_k[position_index]])

        return z

    def measure(self, D_k, x_k):
        """
        Generate measurement.

        Parameters:
        ----------
        D_k : np.ndarray
        x_k : np.ndarray

        Returns:
        -------
        z_k : np.ndarray
        """

        z = self.H(D_k, x_k)

        # Add measurement noise
        noise = self.sample_noise(len(z))
        z_k = z + noise

        return z_k