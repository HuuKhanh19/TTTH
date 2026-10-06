# baselines/alawad_swarm.py

import numpy as np


class AlawadSwarmBaseline:
    """
    Swarm-based UAV coordination using local rules.

    Characteristics:
    ----------------
    - Local interaction only
    - No optimization
    - Simple emergent behavior
    """

    def __init__(self, attraction_weight=0.5, alignment_weight=0.3, separation_weight=0.2, radius=2.0):
        """
        Initialize swarm baseline.

        Parameters:
        ----------
        attraction_weight : float
        alignment_weight : float
        separation_weight : float
        radius : float
            Neighborhood radius
        """
        self.w_attr = attraction_weight
        self.w_align = alignment_weight
        self.w_sep = separation_weight
        self.radius = radius

    # -----------------------------
    # Find neighbors
    # -----------------------------
    def get_neighbors(self, i, uav_states):
        """
        Find neighbors within radius.
        """

        neighbors = []

        for j, xj in enumerate(uav_states):
            if i != j:
                dist = np.linalg.norm(uav_states[i] - xj)
                if dist <= self.radius:
                    neighbors.append(xj)

        return neighbors

    # -----------------------------
    # Attraction toward disaster
    # -----------------------------
    def attraction(self, x, disaster_field):
        """
        Move toward higher intensity region.
        """

        index = int(np.clip(x[0], 0, len(disaster_field) - 1))

        # Look nearby
        window = 2
        start = max(0, index - window)
        end = min(len(disaster_field), index + window)

        local_region = disaster_field[start:end]

        best_idx = np.argmax(local_region)
        target_index = start + best_idx

        return np.array([target_index]) - x

    # -----------------------------
    # Alignment (match neighbors)
    # -----------------------------
    def alignment(self, neighbors):
        """
        Align with neighbors.
        """

        if len(neighbors) == 0:
            return np.zeros(1)

        return np.mean(neighbors, axis=0)

    # -----------------------------
    # Separation (avoid crowding)
    # -----------------------------
    def separation(self, x, neighbors):
        """
        Avoid getting too close.
        """

        force = np.zeros_like(x)

        for n in neighbors:
            diff = x - n
            dist = np.linalg.norm(diff)

            if dist > 1e-6:
                force += diff / dist

        return force

    # -----------------------------
    # Compute control
    # -----------------------------
    def compute_multi_uav_control(self, uav_states, disaster_field):
        """
        Compute controls using swarm rules.
        """

        controls = []

        for i, x in enumerate(uav_states):

            neighbors = self.get_neighbors(i, uav_states)

            # Components
            u_attr = self.attraction(x, disaster_field)
            u_align = self.alignment(neighbors)
            u_sep = self.separation(x, neighbors)

            # Combine
            u = (
                self.w_attr * u_attr +
                self.w_align * u_align +
                self.w_sep * u_sep
            )

            # Normalize
            if np.linalg.norm(u) > 1e-6:
                u = u / np.linalg.norm(u)

            controls.append(u)

        return controls