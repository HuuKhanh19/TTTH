# control/controller.py

import numpy as np


class Controller:
    """
    Implements optimal feedback control law.

    Equation Reference:
    -------------------
    Eq (19): u_k = -K_k s_k
    """

    def __init__(self):
        pass

    # -----------------------------
    # Single UAV Control
    # -----------------------------
    def compute_control(self, K_k, x_k):
        """
        Compute control input for a single UAV.

        Parameters:
        ----------
        K_k : np.ndarray
            Gain matrix at time k
        x_k : np.ndarray
            UAV state

        Returns:
        -------
        u_k : np.ndarray
            Control input
        """

        # Eq (19): u_k = -K_k x_k
        u_k = -K_k @ x_k

        return u_k

    # -----------------------------
    # Multi-UAV Control
    # -----------------------------
    def compute_multi_uav_control(self, K_k, uav_states):
        """
        Compute control inputs for all UAVs.

        Parameters:
        ----------
        K_k : np.ndarray
            Gain matrix
        uav_states : list of np.ndarray

        Returns:
        -------
        controls : list of np.ndarray
        """

        controls = []

        for x in uav_states:
            u = self.compute_control(K_k, x)
            controls.append(u)

        return controls

    # -----------------------------
    # Mean-Field Control (Optional)
    # -----------------------------
    def compute_mean_field_control(self, K_k, uav_states, mean_state=None):
        """
        Mean-field control (optional extension).

        Can incorporate mean-field state if needed.
        """

        controls = []

        for x in uav_states:
            u = -K_k @ x

            # Optional mean-field correction
            if mean_state is not None:
                u -= 0.1 * (x - mean_state)

            controls.append(u)

        return controls