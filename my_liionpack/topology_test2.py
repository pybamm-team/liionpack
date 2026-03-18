import pickle
import numpy as np
import matplotlib.pyplot as plt
import pybamm

from my_liionpack.main_src import run_pack
from my_liionpack.params_and_ocps.base_battery_param import (
    param_base,
    param_adv,
    discret_points,
)


# ------------------------------------------------------------
# Load pre-made charged models
# ------------------------------------------------------------
with open("my_liionpack/models/Charged_adv_model.pkl", "rb") as f:
    adv_model_charged = pickle.load(f)

with open("my_liionpack/models/Charged_base_model.pkl", "rb") as f:
    base_model_charged = pickle.load(f)


# ------------------------------------------------------------
# Experiment settings
# ------------------------------------------------------------
rate = 3
delta_t = 60  # seconds
num_cells_parallel = 4

r_busbar = 2.5e-5
r_terminal = 1e-6

topology_1 = "left"
topology_2 = "left-right"

capacity = param_base["Nominal cell capacity [A.h]"]
total_current = rate * capacity * num_cells_parallel

experiment = pybamm.Experiment(
    [
        "Rest for 3 minutes",
        f"Discharge at {total_current} A for {60 * 0.5 / rate} minutes",
        "Rest for 6 minutes",
    ],
    period=f"{delta_t} seconds",
)


# ------------------------------------------------------------
# Run helper
# ------------------------------------------------------------
def run_one_case(model, params, terminals):
    sols, history = run_pack(
        model=model,
        parameters=params,
        experiment=experiment,
        num_cells_parallel=num_cells_parallel,
        r_busbar=r_busbar,
        r_terminal=r_terminal,
        terminals=terminals,
        var_pts=discret_points,
        Ri=0.005,
    )
    return sols, history


# ------------------------------------------------------------
# History utilities
# ------------------------------------------------------------
def get_time(history):
    return np.array(history["time"], dtype=float)


def get_cell_currents(history):
    return np.array(history["I_cell"], dtype=float)


def get_cell_voltages(history):
    return np.array(history["V_cell"], dtype=float)


def get_pack_voltage(history):
    return np.array(history["V_pack"], dtype=float)


def get_pack_current(history):
    """
    Reconstruct pack current from cell currents.
    For a parallel pack, sum of branch currents = pack current.
    """
    I_cells = get_cell_currents(history)
    return np.sum(I_cells, axis=0)


# ------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------
def plot_case_on_axes(ax_current, ax_voltage, history, style="-", prefix=""):
    t = get_time(history)
    I_cells = get_cell_currents(history)
    V_cells = get_cell_voltages(history)
    I_pack = get_pack_current(history)
    V_pack = get_pack_voltage(history)

    # # Pack current
    # ax_current.plot(
    #     t, I_pack,
    #     linestyle=style,
    #     linewidth=2.5,
    #     color="k",
    #     label=f"{prefix} Pack"
    # )

    # Each cell current
    for i in range(I_cells.shape[0]):
        ax_current.plot(
            t,
            I_cells[i],
            linestyle=style,
            linewidth=1.7,
            color=f"C{i}",
            label=f"{prefix} Cell {i + 1}",
        )

    # # Pack voltage
    # ax_voltage.plot(
    #     t, V_pack,
    #     linestyle=style,
    #     linewidth=2.5,
    #     color="k",
    #     label=f"{prefix} Pack"
    # )

    # Each cell voltage
    for i in range(V_cells.shape[0]):
        ax_voltage.plot(
            t,
            V_cells[i],
            linestyle=style,
            linewidth=1.7,
            color=f"C{i}",
            label=f"{prefix} Cell {i + 1}",
        )


def make_comparison_figure(history_a, history_b, label_a, label_b, title):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax_current = axes[0]
    ax_voltage = axes[1]

    plot_case_on_axes(
        ax_current, ax_voltage, history_a, style="--", prefix=f"{label_a} |"
    )
    plot_case_on_axes(
        ax_current, ax_voltage, history_b, style="-", prefix=f"{label_b} |"
    )

    ax_current.set_title(f"{title} - Currents")
    ax_current.set_ylabel("Current [A]")
    ax_current.grid(True, alpha=0.4)
    ax_current.legend(frameon=False, fontsize=8, ncol=2)

    ax_voltage.set_title(f"{title} - Voltages")
    ax_voltage.set_xlabel("Time [s]")
    ax_voltage.set_ylabel("Voltage [V]")
    ax_voltage.grid(True, alpha=0.4)
    ax_voltage.legend(frameon=False, fontsize=8, ncol=2)

    plt.tight_layout()
    return fig


# ------------------------------------------------------------
# Optional simple summary
# ------------------------------------------------------------
def print_summary(name, history):
    I_cells = get_cell_currents(history)
    V_cells = get_cell_voltages(history)
    I_pack = get_pack_current(history)
    V_pack = get_pack_voltage(history)

    print(f"\n--- {name} ---")
    print("Final cell currents [A]:", np.round(I_cells[:, -1], 4))
    print("Final pack current [A]:", round(I_pack[-1], 4))
    print("Final cell voltages [V]:", np.round(V_cells[:, -1], 4))
    print("Final pack voltage [V]:", round(V_pack[-1], 4))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("\nRunning 4 cases...\n")

    # 1) Base + left
    _, hist_base_left = run_one_case(
        model=base_model_charged,
        params=param_base,
        terminals=topology_1,
    )

    # 2) Adv + left
    _, hist_adv_left = run_one_case(
        model=adv_model_charged,
        params=param_adv,
        terminals=topology_1,
    )

    # 3) Base + left-right
    _, hist_base_lr = run_one_case(
        model=base_model_charged,
        params=param_base,
        terminals=topology_2,
    )

    # 4) Adv + left-right
    _, hist_adv_lr = run_one_case(
        model=adv_model_charged,
        params=param_adv,
        terminals=topology_2,
    )

    print_summary(f"Base | {topology_1}", hist_base_left)
    print_summary(f"Adv  | {topology_1}", hist_adv_left)
    print_summary(f"Base | {topology_2}", hist_base_lr)
    print_summary(f"Adv  | {topology_2}", hist_adv_lr)

    # --------------------------------------------------------
    # Figure 1: for topology_1 compare Base vs Advanced
    # --------------------------------------------------------
    make_comparison_figure(
        hist_base_left,
        hist_adv_left,
        label_a="Base",
        label_b="Adv",
        title=f"Topology = {topology_1}",
    )

    # --------------------------------------------------------
    # Figure 2: for topology_2 compare Base vs Advanced
    # --------------------------------------------------------
    make_comparison_figure(
        hist_base_lr,
        hist_adv_lr,
        label_a="Base",
        label_b="Adv",
        title=f"Topology = {topology_2}",
    )

    # --------------------------------------------------------
    # Figure 3: for Base compare topology_1 vs topology_2
    # --------------------------------------------------------
    make_comparison_figure(
        hist_base_left,
        hist_base_lr,
        label_a=topology_1,
        label_b=topology_2,
        title="Model = Base",
    )

    # --------------------------------------------------------
    # Figure 4: for Adv compare topology_1 vs topology_2
    # --------------------------------------------------------
    make_comparison_figure(
        hist_adv_left,
        hist_adv_lr,
        label_a=topology_1,
        label_b=topology_2,
        title="Model = Adv",
    )

    plt.show()


if __name__ == "__main__":
    main()
