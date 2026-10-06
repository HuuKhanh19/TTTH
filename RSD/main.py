# main.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Dynamics
from dynamics.disaster_model import DisasterModel
from dynamics.uav_dynamics import UAVDynamics
from dynamics.measurement_model import MeasurementModel

# Control
from control.cost_function import CostFunction
from control.riccati_solver import RiccatiSolver
from control.controller import Controller

# Mean-field
from mean_field.covariance_update import MeanFieldCovariance
from mean_field.mean_field_update import MeanField

# Distributed
from distributed.communication_graph import CommunicationGraph
from distributed.consensus import Consensus

# Baselines
from baselines.adam_ga import AdamGABaseline
from baselines.javed_clustering import JavedClusteringBaseline
from baselines.alawad_swarm import AlawadSwarmBaseline

# Simulation
from simulation.simulator import Simulator

# Evaluation
from evaluation.metrics import Metrics
from evaluation.summary_table import SummaryTable

# Visualization
from visualization.plot_coverage import PlotCoverage
from visualization.plot_energy import PlotEnergy
from visualization.plot_path_length import PlotPathLength
from visualization.plot_connectivity import PlotConnectivity
from visualization.plot_robustness import PlotRobustness
from visualization.plot_radar import PlotRadar
from visualization.plot_improvement import PlotImprovement


def main(profile="default", theta_override=None, seed=42):

    np.random.seed(seed)

    # -----------------------------
    # CONFIGURATION
    # -----------------------------
    config = {
        "T": 50,
        "num_uavs": 10,
        "state_dim": 1,
        "grid_size": 50,
        "E0": 100.0,
        "state_weight": 1.0,
        "control_weight": 1.0,
        "process_noise": 0.1,
        "theta": 0.5,
        "communication_radius": 2.0,
        "consensus_gamma": 0.1,
        "minimum_energy": 0.0,
        "alpha": 0.1,
        "beta": 0.01,
    }

    if profile == "paper-table9-1d":
        config.update({
            "T": 100,
            "state_weight": 1.0,
            "control_weight": 0.1,
            "process_noise": 0.01,
            "theta": 0.1,
            "communication_radius": 12.0,
            "consensus_gamma": 0.15,
            "minimum_energy": 0.1 * config["E0"],
        })
    elif profile != "default":
        raise ValueError(f"Unknown profile: {profile}")

    if theta_override is not None:
        config["theta"] = theta_override
    if config["theta"] < 0:
        raise ValueError("theta must be nonnegative")
    effective_config = config.copy()

    # System matrices (example)
    A = np.eye(config["state_dim"])
    B = np.eye(config["state_dim"])
    Q = config["state_weight"] * np.eye(config["state_dim"])
    R = config["control_weight"] * np.eye(config["state_dim"])
    W = config["process_noise"] * np.eye(config["state_dim"])

    theta = config["theta"]

    # -----------------------------
    # INITIALIZE MODULES
    # -----------------------------
    disaster_A = np.eye(config["grid_size"])
    disaster_W = config["process_noise"] * np.eye(config["grid_size"])
    disaster_model = DisasterModel(disaster_A, disaster_W)
    uav_dynamics = UAVDynamics(
        A, B, W,
        alpha=config["alpha"], beta=config["beta"],
        E_min=config["minimum_energy"],
    )
    measurement_model = MeasurementModel(R)

    cost_function = CostFunction(
        {"lambda1":1, "lambda2":1, "lambda3":1, "lambda4":1},
        theta
    )

    adjustment = "paper_eq21" if profile == "paper-table9-1d" else "legacy"
    riccati_solver = RiccatiSolver(A, B, Q, R, W, theta, config["T"], adjustment)
    controller = Controller()

    mean_field_cov = MeanFieldCovariance(A, B, W)
    mean_field = MeanField(Q, R)

    graph = CommunicationGraph(communication_radius=config["communication_radius"])
    consensus = Consensus(gamma=config["consensus_gamma"])

    # Baselines
    adam = AdamGABaseline()
    javed = JavedClusteringBaseline()
    alawad = AlawadSwarmBaseline()

    config["adam"] = adam
    config["javed"] = javed
    config["alawad"] = alawad

    simulator = Simulator(
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
    )

    metrics = Metrics(config["grid_size"])
    summary = SummaryTable()

    # -----------------------------
    # RUN ALL METHODS
    # -----------------------------
    methods = ["proposed", "adam", "javed", "alawad"]

    results_history = {}
    results_metrics = {}

    for method in methods:
        print(f"Running {method}...")

        history = simulator.run(method=method)
        results_history[method] = history

        results_metrics[method] = metrics.compute_all(history, graph)

    # -----------------------------
    # GENERATE SUMMARY TABLE
    # -----------------------------
    df = summary.generate(results_metrics)
    summary.save(df)
    metrics_df = pd.DataFrame.from_dict(results_metrics, orient="index")
    metrics_df.index.name = "method"
    metrics_df.to_csv("data/tables/method_metrics.csv")

    manifest = {"profile": profile, "seed": seed, "effective_config": effective_config,
                "risk_adjustment": adjustment}
    if profile == "paper-table9-1d":
        margins = [np.min(np.real(np.linalg.eigvals(
            np.eye(config["state_dim"]) - 2 * theta * W @ P
        ))) for P in riccati_solver.P[1:]]
        radii = [np.max(np.abs(np.linalg.eigvals(A - B @ K)))
                 for K in riccati_solver.K]
        manifest["diagnostics"] = {
            "minimum_admissibility_eigenvalue": float(min(margins)),
            "maximum_closed_loop_spectral_radius": float(max(radii)),
        }
    Path("data/tables/run_config.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    # -----------------------------
    # PLOTS
    # -----------------------------
    PlotCoverage(config["grid_size"]).plot(results_history)
    PlotEnergy().plot(results_history)
    PlotPathLength().plot(results_history)
    PlotConnectivity(graph).plot(results_history)

    PlotRadar().plot(results_metrics)
    PlotImprovement().plot(df)

    print("\nAll simulations completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RSD simulation")
    parser.add_argument("--profile", choices=["default", "paper-table9-1d"], default="default")
    parser.add_argument("--theta", type=float)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(profile=args.profile, theta_override=args.theta, seed=args.seed)
