# control/riccati_solver.py

import numpy as np


class RiccatiSolver:
    """
    Implements risk-sensitive Riccati recursion.

    Equation References:
    --------------------
    Eq (18): Risk-adjusted matrix
    Eq (21): Riccati recursion
    """

    def __init__(self, A, B, Q, R, W, theta, T, risk_adjustment="legacy"):
        """
        Initialize solver.

        Parameters:
        ----------
        A, B : system matrices
        Q, R : cost matrices
        W : noise covariance
        theta : risk sensitivity
        T : time horizon
        risk_adjustment : str
            "legacy" preserves the repository's approximation;
            "paper_eq21" uses P(I - 2 theta W P)^-1.
        """

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.W = W
        self.theta = theta
        self.T = T
        self.risk_adjustment = risk_adjustment

        self.n = A.shape[0]

        # Storage
        self.P = [None] * (T + 1)
        self.K = [None] * T

    # -----------------------------
    # Eq (18): Risk Adjustment
    # -----------------------------
    def compute_risk_adjusted_P(self, P_next):
        """
        Compute P̄_{k+1}

        The legacy mode uses P + θ P W P. The paper mode uses
        P(I - 2θ W P)^-1 and checks risk admissibility.
        """

        if self.risk_adjustment == "legacy":
            return P_next + self.theta * P_next @ self.W @ P_next

        if self.risk_adjustment == "paper_eq21":
            denominator = np.eye(self.n) - 2 * self.theta * self.W @ P_next
            if np.min(np.real(np.linalg.eigvals(denominator))) <= 0:
                raise ValueError("Risk parameter violates the paper's admissibility condition")
            return np.linalg.solve(denominator.T, P_next.T).T

        raise ValueError(f"Unknown risk adjustment: {self.risk_adjustment}")

    # -----------------------------
    # Backward Riccati Recursion
    # -----------------------------
    def solve(self):
        """
        Solve Riccati recursion backward in time.

        Returns:
        -------
        P : list of matrices
        K : list of gain matrices
        """

        # Terminal condition: P_T = Q
        self.P[self.T] = self.Q.copy()

        for k in reversed(range(self.T)):

            P_next = self.P[k + 1]

            # Eq (18)
            P_bar = self.compute_risk_adjusted_P(P_next)

            # Precompute terms
            BT_Pbar = self.B.T @ P_bar
            R_eff = self.R + BT_Pbar @ self.B

            # Inverse (numerically stable)
            R_inv = np.linalg.inv(R_eff)

            # Eq (21)
            term1 = self.Q
            term2 = self.A.T @ P_bar @ self.A
            term3 = self.A.T @ P_bar @ self.B @ R_inv @ BT_Pbar @ self.A

            P_k = term1 + term2 - term3

            self.P[k] = P_k

            # -----------------------------
            # Compute Gain (Eq 20)
            # -----------------------------
            K_k = R_inv @ BT_Pbar @ self.A

            self.K[k] = K_k

        return self.P, self.K

    def get_gain(self, k):
        """
        Retrieve gain matrix K_k
        """
        return self.K[k]
