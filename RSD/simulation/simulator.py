"""Simulation loop for disaster tracking and UAV control."""

# simulation/simulator.py

import numpy as np


class Simulator:
    """
    Runs full UAV swarm simulation.

    Implements Algorithm 1 from the paper.
    """

    def __init__(
        self,
        disaster_model,
        uav_dynamics,
        measurement_model,
        controller,
        riccati_solver,
        mean_field_cov,
        consensus,
        graph,
        cost_function,
        config
    ):
        self.disaster_model = disaster_model
        self.uav_dynamics = uav_dynamics
        self.measurement_model = measurement_model
        self.controller = controller
        self.riccati_solver = riccati_solver
        self.mean_field_cov = mean_field_cov
        self.consensus = consensus
        self.graph = graph
        self.cost_function = cost_function
        self.config = config

    # -----------------------------
    # Run Simulation
    # -----------------------------
    def run(self, method="proposed"):
        """
        Run simulation for a given method.

        Parameters:
        ----------
        method : str
            'proposed', 'adam', 'javed', 'alawad'

        Returns:
        -------
        results : dict
        """

        T = self.config["T"]
        N = self.config["num_uavs"]
        state_dim = self.config["state_dim"]

        # Initialize
        D = self.disaster_model.initialize_state(self.config["grid_size"])
        uav_states = [
            self.uav_dynamics.initialize_state(state_dim)
            for _ in range(N)
        ]
        energies = [self.config["E0"]] * N

        Sigma = self.mean_field_cov.initialize(state_dim)

        # Baselines use no feedback gain in the mean-field covariance update.
        K_list = [np.zeros((self.riccati_solver.B.shape[1], state_dim)) for _ in range(T)]

        # Solve Riccati (for proposed)
        if method == "proposed":
            _, K_list = self.riccati_solver.solve()

        # Logs
        history = {
            "states": [],
            "energy": [],
            "coverage": [],
        }

        # -----------------------------
        # Time Loop
        # -----------------------------
        for k in range(T):

            # 1. Update disaster
            D = self.disaster_model.step(D)

            # 2. Measurements
            measurements = [
                self.measurement_model.measure(D, x)
                for x in uav_states
            ]

            # 3. Compute control
            if method == "proposed":
                K_k = K_list[k]
                controls = self.controller.compute_multi_uav_control(K_k, uav_states)

            elif method == "adam":
                controls = self.config["adam"].compute_multi_uav_control(uav_states, D)

            elif method == "javed":
                controls = self.config["javed"].compute_multi_uav_control(uav_states, D)

            elif method == "alawad":
                controls = self.config["alawad"].compute_multi_uav_control(uav_states, D)

            else:
                raise ValueError("Unknown method")

            # 4. Update UAV states and energy
            new_states = []
            new_energies = []

            for i in range(N):
                x_next = self.uav_dynamics.step_state(uav_states[i], controls[i])
                E_next = self.uav_dynamics.step_energy(energies[i], controls[i])

                new_states.append(x_next)
                new_energies.append(E_next)

            uav_states = new_states
            energies = new_energies

            # 5. Mean-field update
            Sigma = self.mean_field_cov.update(Sigma, K_list[k])

            # 6. Communication + consensus
            A = self.graph.compute_adjacency(uav_states)
            Sigma_list = [Sigma.copy() for _ in range(N)]
            Sigma_list = self.consensus.update(Sigma_list, A)

            # 7. Compute cost
            self.cost_function.compute_stage_cost(
                D_k=D,
                D_hat_k=D,
                controls=controls,
                uav_states=uav_states,
                latency=1.0,
                risk_function=self.cost_function.default_risk_function
            )

            # 8. Log data
            history["states"].append(uav_states)
            history["energy"].append(energies)

        return history
