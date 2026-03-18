import numpy as np


def plot_results(history, num_cells, fig, axes, linestyle="--", lab=None):
    axes = axes.flatten()

    time = np.array(history["time"])

    # Plot Currents
    for i in range(num_cells):
        color = f"C{i}"
        if lab is not None:
            label = f"{lab} - Cell {i + 1}"
        axes[0].plot(
            time + time[1],
            history["I_cell"][i],
            label=label,
            color=color,
            linestyle=linestyle,
        )
        axes[1].plot(
            time + time[1],
            history["V_cell"][i],
            label=label,
            color=color,
            linestyle=linestyle,
        )
        axes[2].plot(
            time,
            history["R_internal"][i],
            label=label,
            color=color,
            linestyle=linestyle,
        )
        axes[3].plot(
            time + time[1],
            history["SOC"][i],
            label=label,
            color=color,
            linestyle=linestyle,
        )

    # axes[1].plot(time, history["terminal_voltage"], 'o', label='Mine Terminal V', alpha=0.7)

    # axes[0].set_title("Cell Currents")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Current [A]")
    axes[0].grid(True, alpha=0.5)

    # axes[1].set_title("Terminal Voltages")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Voltage [V]")
    axes[1].grid(True, alpha=0.5)

    # Plot Resistances
    # axes[2].set_title("Internal Resistance (Linearized)")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Resistance [Ohm]")

    axes[2].grid(True, alpha=0.5)

    # axes[3].set_title("Cell State of Charge")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylabel("SOC (%)")
    axes[3].grid(True, alpha=0.5)
    axes[3].legend(frameon=False)
