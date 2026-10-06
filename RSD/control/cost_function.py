# control/cost_function.py

import numpy as np


class CostFunction:
    """
    Implements risk-sensitive cooperative cost.

    Equation References:
    --------------------
    Eq (6): Stage cost ℓ_k
    Eq (7): Risk-sensitive cost J
    """

    def __init__(self, lambda_weights, theta):
        """
        Initialize cost function.

        Parameters:
        ----------
        lambda_weights : dict
            {'lambda1', 'lambda2', 'lambda3', 'lambda4'}
        theta : float
            Risk sensitivity parameter
        """
        self.lambda1 = lambda_weights["lambda1"]
        self.lambda2 = lambda_weights["lambda2"]
        self.lambda3 = lambda_weights["lambda3"]
        self.lambda4 = lambda_weights["lambda4"]
        self.theta = theta

        self.cost_history = []

    # -----------------------------
    # Stage Cost (Eq 6)
    # -----------------------------
    def compute_stage_cost(
        self,
        D_k,
        D_hat_k,
        controls,
        uav_states,
        latency,
        risk_function
    ):
        """
        Compute stage cost ℓ_k.

        Parameters:
        ----------
        D_k : np.ndarray
            True disaster state
        D_hat_k : np.ndarray
            Estimated disaster state
        controls : list of np.ndarray
            Control inputs for all UAVs
        uav_states : list of np.ndarray
            UAV states
        latency : float
            Response delay term
        risk_function : callable
            Function R(x, D)

        Returns:
        -------
        ℓ_k : float
        """

        # --- Tracking error ---
        tracking_error = np.linalg.norm(D_k - D_hat_k) ** 2

        # --- Control effort ---
        control_effort = sum(np.linalg.norm(u)**2 for u in controls)

        # --- Risk exposure ---
        risk_exposure = sum(risk_function(x, D_k) for x in uav_states)

        # --- Total stage cost ---
        l_k = (
            self.lambda1 * tracking_error +
            self.lambda2 * latency +
            self.lambda3 * control_effort +
            self.lambda4 * risk_exposure
        )

        self.cost_history.append(l_k)

        return l_k

    # -----------------------------
    # Risk Function R(x, D)
    # -----------------------------
    def default_risk_function(self, x, D_k):
        """
        Penalize UAV being in high-intensity disaster region.

        Simple implementation:
        risk = disaster intensity at UAV position
        """

        index = int(np.clip(x[0], 0, len(D_k) - 1))
        return D_k[index]

    # -----------------------------
    # Risk-Sensitive Cost (Eq 7)
    # -----------------------------
    def compute_total_cost(self):
        """
        Compute total risk-sensitive cost.

        J = (1/θ) log E[exp(θ * sum ℓ_k)]

        Returns:
        -------
        J : float
        """

        cumulative_cost = sum(self.cost_history)

        # Exponential transformation
        J = (1 / self.theta) * np.log(np.exp(self.theta * cumulative_cost))

        return J

    def reset(self):
        """
        Reset cost history for new simulation run.
        """
        self.cost_history = []