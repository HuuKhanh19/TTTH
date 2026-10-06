# distributed/communication_graph.py

import numpy as np


class CommunicationGraph:
    """
    Implements UAV communication topology.

    Graph: G_k = (V, E_k)
    """

    def __init__(self, communication_radius):
        """
        Initialize graph.

        Parameters:
        ----------
        communication_radius : float
            Max distance for communication
        """
        self.radius = communication_radius

    # -----------------------------
    # Build Adjacency Matrix
    # -----------------------------
    def compute_adjacency(self, uav_states):
        """
        Compute adjacency matrix based on distance.

        Parameters:
        ----------
        uav_states : list of np.ndarray

        Returns:
        -------
        A : np.ndarray
            Adjacency matrix (N x N)
        """

        N = len(uav_states)
        A = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                if i != j:
                    dist = np.linalg.norm(uav_states[i] - uav_states[j])

                    if dist <= self.radius:
                        A[i, j] = 1

        return A

    # -----------------------------
    # Degree Matrix
    # -----------------------------
    def compute_degree(self, A):
        """
        Compute degree matrix.

        D[i,i] = sum of row i
        """

        D = np.diag(np.sum(A, axis=1))
        return D

    # -----------------------------
    # Laplacian Matrix
    # -----------------------------
    def compute_laplacian(self, A):
        """
        Compute graph Laplacian.

        L = D - A
        """

        D = self.compute_degree(A)
        L = D - A

        return L

    # -----------------------------
    # Check Connectivity (Optional)
    # -----------------------------
    def is_connected(self, A):
        """
        Check if graph is connected.

        Uses eigenvalue test:
        Graph is connected if second smallest eigenvalue > 0
        """

        L = self.compute_laplacian(A)
        eigenvalues = np.linalg.eigvals(L)

        eigenvalues = np.sort(np.real(eigenvalues))

        # Algebraic connectivity
        return eigenvalues[1] > 1e-6