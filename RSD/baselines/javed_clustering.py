# baselines/javed_clustering.py

import numpy as np


class JavedClusteringBaseline:
    """
    Clustering-based UAV coordination.

    Characteristics:
    ----------------
    - Group-based movement
    - Local coordination
    - No global optimality
    """

    def __init__(self, num_clusters=3, step_size=0.5):
        """
        Initialize baseline.

        Parameters:
        ----------
        num_clusters : int
            Number of UAV clusters
        step_size : float
            Movement scaling factor
        """
        self.num_clusters = num_clusters
        self.step_size = step_size

    # -----------------------------
    # Assign UAVs to clusters
    # -----------------------------
    def assign_clusters(self, uav_states):
        """
        Simple clustering based on index (can be extended to K-means).
        """

        clusters = [[] for _ in range(self.num_clusters)]

        for i, x in enumerate(uav_states):
            cluster_id = i % self.num_clusters
            clusters[cluster_id].append((i, x))

        return clusters

    # -----------------------------
    # Compute cluster centroid
    # -----------------------------
    def compute_centroid(self, cluster):
        """
        Compute average position of cluster.
        """

        positions = [x for _, x in cluster]
        return np.mean(positions, axis=0)

    # -----------------------------
    # Find local target
    # -----------------------------
    def find_target(self, centroid, disaster_field):
        """
        Move toward high-intensity region near centroid.
        """

        index = int(np.clip(centroid[0], 0, len(disaster_field) - 1))

        # Look around neighborhood
        window = 3
        start = max(0, index - window)
        end = min(len(disaster_field), index + window)

        local_region = disaster_field[start:end]

        best_idx = np.argmax(local_region)
        target_index = start + best_idx

        return np.array([target_index])

    # -----------------------------
    # Compute control
    # -----------------------------
    def compute_multi_uav_control(self, uav_states, disaster_field):
        """
        Compute controls for all UAVs.
        """

        controls = [None] * len(uav_states)

        clusters = self.assign_clusters(uav_states)

        for cluster in clusters:

            centroid = self.compute_centroid(cluster)
            target = self.find_target(centroid, disaster_field)

            for idx, x in cluster:

                # Move toward target
                direction = target - x

                # Normalize
                if np.linalg.norm(direction) > 1e-6:
                    direction = direction / np.linalg.norm(direction)

                u = self.step_size * direction

                controls[idx] = u

        return controls