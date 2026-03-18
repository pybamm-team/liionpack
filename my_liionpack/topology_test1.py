import numpy as np
import matplotlib.pyplot as plt
import pybamm

from main_src import run_pack


def current_spread_metric(history):
    """
    Returns:
        spread_abs : max(I_cell) - min(I_cell) at each time step
        spread_rel : (max - min) / mean(|I|) at each time step
    """
    I = np.array(history["I_cell"], dtype=float)  # shape = (N_cells, N_time)
    max_I = np.max(I, axis=0)
    min_I = np.min(I, axis=0)
    mean_abs_I = np.mean(np.abs(I), axis=0)

    spread_abs = max_I - min_I
    spread_rel = spread_abs / np.maximum(mean_abs_I, 1e-12)
    return spread_abs, spread_rel


def summarize_case(name, history):
    spread_abs, spread_rel = current_spread_metric(history)

    I = np.array(history["I_cell"], dtype=float)
    final_currents = I[:, -1]
    mean_abs = np.mean(np.abs(final_currents))
    final_abs_spread = np.max(final_currents) - np.min(final_currents)
    final_rel_spread = final_abs_spread / max(mean_abs, 1e-12)

    avg_rel_spread = np.mean(spread_rel)
    max_rel_spread = np.max(spread_rel)

    print(f"\n--- {name} ---")
    print(f"Final cell currents [A]: {np.round(final_currents, 4)}")
    print(f"Final absolute spread [A]: {final_abs_spread:.6f}")
    print(f"Final relative spread [-]: {final_rel_spread:.6f}")
    print(f"Average relative spread over run [-]: {avg_rel_spread:.6f}")
    print(f"Maximum relative spread over run [-]: {max_rel_spread:.6f}")

    return {
        "final_abs_spread": final_abs_spread,
        "final_rel_spread": final_rel_spread,
        "avg_rel_spread": avg_rel_spread,
        "max_rel_spread": max_rel_spread,
    }


def run_case(terminals, label):
    """
    Runs one simple 4P discharge case with a chosen terminal topology.
    """
    model = pybamm.lithium_ion.SPMe()
    params = pybamm.ParameterValues("Prada2013")  # LFP parameter set

    num_cells_parallel = 4
    initial_soc = 0.90

    # Make topology effect visible
    r_busbar = 50 * 2.5e-5  # = 0.00125 ohm between adjacent rail nodes
    r_terminal = 1e-10

    rate = 1.0
    capacity = params["Nominal cell capacity [A.h]"]
    total_current = rate * capacity * num_cells_parallel

    delta_t = 10  # seconds
    discharge_time = 12 * 60  # 12 minutes; short and simple

    experiment = pybamm.Experiment(
        [
            f"Discharge at {total_current} A for {discharge_time} seconds or until 2.5 V",
        ],
        period=f"{delta_t} seconds",
    )

    sols, history = run_pack(
        model=model,
        parameters=params,
        experiment=experiment,
        num_cells_parallel=num_cells_parallel,
        r_busbar=r_busbar,
        r_terminal=r_terminal,
        terminals=terminals,
        initial_soc=initial_soc,
        save_memory=True,
        solver=pybamm.CasadiSolver(mode="safe"),
        Ri=0.01,  # initial/default resistance guess used before updating
    )

    return sols, history


def plot_histories(histories, labels):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")

    for history, label in zip(histories, labels):
        time = np.array(history["time"], dtype=float)
        I = np.array(history["I_cell"], dtype=float)
        V = np.array(history["V_cell"], dtype=float)
        Vpack = np.array(history["V_pack"], dtype=float)

        spread_abs, spread_rel = current_spread_metric(history)

        # Cell currents
        for i in range(I.shape[0]):
            axes[0, 0].plot(time, I[i], label=f"{label} - Cell {i + 1}")
        axes[0, 0].set_title("Cell currents")
        axes[0, 0].set_ylabel("Current [A]")
        axes[0, 0].grid(True, alpha=0.3)

        # Pack voltage
        axes[0, 1].plot(time, Vpack, linewidth=2, label=label)
        axes[0, 1].set_title("Pack voltage")
        axes[0, 1].set_ylabel("Voltage [V]")
        axes[0, 1].grid(True, alpha=0.3)

        # Current spread
        axes[1, 0].plot(time, spread_abs, linewidth=2, label=label)
        axes[1, 0].set_title("Absolute current spread")
        axes[1, 0].set_xlabel("Time [s]")
        axes[1, 0].set_ylabel("max(I) - min(I) [A]")
        axes[1, 0].grid(True, alpha=0.3)

        # Relative spread
        axes[1, 1].plot(time, spread_rel, linewidth=2, label=label)
        axes[1, 1].set_title("Relative current spread")
        axes[1, 1].set_xlabel("Time [s]")
        axes[1, 1].set_ylabel("(max-min)/mean(|I|) [-]")
        axes[1, 1].grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 1].legend()
    axes[1, 0].legend()
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def main():
    # Case 1: asymmetric topology
    _, history_left = run_case(terminals="left", label="left")

    # Case 2: more symmetric topology
    _, history_leftright = run_case(terminals="left-right", label="left-right")

    # Print numerical summary
    stats_left = summarize_case("Topology = left", history_left)
    stats_leftright = summarize_case("Topology = left-right", history_leftright)

    # Simple qualitative validation statement
    print("\n=== Simple qualitative validation check ===")
    if stats_left["avg_rel_spread"] > stats_leftright["avg_rel_spread"]:
        print(
            "PASS: 'left' topology gives larger current imbalance than 'left-right'.\n"
            "This is the expected qualitative behavior."
        )
    else:
        print(
            "WARNING: current imbalance did not reduce for 'left-right'.\n"
            "Check terminal handling, busbar resistance level, and plotting/history logic."
        )

    # Plot results
    plot_histories(
        histories=[history_left, history_leftright],
        labels=["left", "left-right"],
    )


if __name__ == "__main__":
    main()
