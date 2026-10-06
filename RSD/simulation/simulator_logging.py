import numpy as np
import pandas as pd
import os


class Simulator:

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

        # Create data folders
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)

    # -----------------------------
    # RUN SIMULATION WITH LOGGING
    # -----------------------------
    def run(self, method="proposed"):

        T = self.config["T"]
        N = self.config["num_uavs"]
        state_dim = self.config["state_dim"]

        D = self.disaster_model.initialize_state(self.config["grid_size"])
        uav_states = [
            self.uav_dynamics.initialize_state(state_dim)
            for _ in range(N)
        ]
        energies = [self.config["E0"]] * N

        Sigma = self.mean_field_cov.initialize(state_dim)

        if method == "proposed":
            _, K_list = self.riccati_solver.solve()

        # -----------------------------
        # RAW LOG STORAGE
        # -----------------------------
        raw_log = []

        history = {
            "states": [],
            "energy": []
        }

        # -----------------------------
        # TIME LOOP
        # -----------------------------
        for k in range(T):

            D = self.disaster_model.step(D)

            # Measurements (not stored but used if needed)
            _ = [
                self.measurement_model.measure(D, x)
                for x in uav_states
            ]

            # CONTROL
            if method == "proposed":
                K_k = K_list[k]
                controls = self.controller.compute_multi_uav_control(K_k, uav_states)

            elif method == "adam":
                controls = self.config["adam"].compute_multi_uav_control(uav_states, D)

            elif method == "javed":
                controls = self.config["javed"].compute_multi_uav_control(uav_states, D)

            elif method == "alawad":
                controls = self.config["alawad"].compute_multi_uav_control(uav_states, D)

            # UPDATE STATES
            new_states = []
            new_energies = []

            for i in range(N):
                x_next = self.uav_dynamics.step_state(uav_states[i], controls[i])
                E_next = self.uav_dynamics.step_energy(energies[i], controls[i])

                new_states.append(x_next)
                new_energies.append(E_next)

                # 🔥 RAW LOG ENTRY
                raw_log.append([
                    k,
                    i,
                    float(x_next[0]),
                    float(E_next)
                ])

            uav_states = new_states
            energies = new_energies

            # Mean-field
            if method == "proposed":
                Sigma = self.mean_field_cov.update(Sigma, K_list[k])

            # Consensus
            A = self.graph.compute_adjacency(uav_states)
            Sigma_list = [Sigma.copy() for _ in range(N)]
            _ = self.consensus.update(Sigma_list, A)

            # Cost
            self.cost_function.compute_stage_cost(
                D_k=D,
                D_hat_k=D,
                controls=controls,
                uav_states=uav_states,
                latency=1.0,
                risk_function=self.cost_function.default_risk_function
            )

            history["states"].append(uav_states)
            history["energy"].append(energies)

        # -----------------------------
        # SAVE RAW DATA
        # -----------------------------
        df_raw = pd.DataFrame(
            raw_log,
            columns=["time", "uav_id", "position", "energy"]
        )

        raw_path = f"data/raw/{method}.csv"
        df_raw.to_csv(raw_path, index=False)

        print(f"[Saved] Raw data → {raw_path}")

        return history

