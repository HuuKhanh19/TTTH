

import numpy as np


class Parameters:
    """
    Central configuration file for all simulation parameters.

    All values here correspond to those used in the manuscript.
    Modify here to reproduce or extend experiments.
    """

    def __init__(self):

        # =============================
        # 1. SIMULATION SETTINGS
        # =============================
        self.T = 50                  # Time horizon
        self.num_uavs = 10           # Number of UAVs
        self.state_dim = 1           # State dimension (1D grid)
        self.grid_size = 50          # Disaster grid size
        self.E0 = 100.0              # Initial energy per UAV

        # =============================
        # 2. SYSTEM MATRICES
        # =============================
        self.A = np.eye(self.state_dim)
        self.B = np.eye(self.state_dim)

        # Cost matrices
        self.Q = np.eye(self.state_dim)
        self.R = np.eye(self.state_dim)

        # Noise covariance
        self.W = 0.1 * np.eye(self.state_dim)
        self.measurement_noise = 0.05 * np.eye(self.state_dim)

        # =============================
        # 3. RISK-SENSITIVE CONTROL
        # =============================
        self.theta = 0.5   # Risk sensitivity parameter

        # =============================
        # 4. ENERGY MODEL
        # =============================
        self.alpha = 0.1   # Control energy coefficient
        self.beta = 0.01   # Baseline consumption
        self.E_min = 0.0   # Minimum energy

        # =============================
        # 5. MEAN-FIELD SETTINGS
        # =============================
        self.initial_covariance_scale = 1.0

        # =============================
        # 6. DISTRIBUTED SYSTEM
        # =============================
        self.communication_radius = 2.0
        self.consensus_gamma = 0.1

        # =============================
        # 7. BASELINE PARAMETERS
        # =============================

        # Adam (GA-like)
        self.adam_num_candidates = 10
        self.adam_control_limit = 1.0

        # Javed (clustering)
        self.javed_num_clusters = 3
        self.javed_step_size = 0.5

        # Alawad (swarm)
        self.swarm_attraction = 0.5
        self.swarm_alignment = 0.3
        self.swarm_separation = 0.2
        self.swarm_radius = 2.0

        # =============================
        # 8. METRIC SETTINGS
        # =============================
        self.coverage_threshold = 0.9

        # =============================
        # 9. ROBUSTNESS EXPERIMENT
        # =============================
        self.noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3]

        # =============================
        # 10. RANDOM SEED
        # =============================
        self.random_seed = 42
        np.random.seed(self.random_seed)

    # ---------------------------------
    # Convert to dictionary (for simulator)
    # ---------------------------------
    def to_dict(self):
        return {
            "T": self.T,
            "num_uavs": self.num_uavs,
            "state_dim": self.state_dim,
            "grid_size": self.grid_size,
            "E0": self.E0,
        }