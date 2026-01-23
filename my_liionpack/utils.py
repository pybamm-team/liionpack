import pybamm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time

def update_internal_resistance(sims, default_R=0.02):
    """
    Updates 'R' using R = |(OCV - V) / I|
    Returns default_R if the simulation has not started.
    """
    R_values = []
    OCVs = []
    for sim in sims:
        # Fetch last entries
        solution = sim.solution
        V = solution["Voltage [V]"].entries[-1]
        I = solution["Current [A]"].entries[-1]
        # OCV = solution["Battery open-circuit voltage [V]"].entries[-1]
        OCV = solution["Surface open-circuit voltage [V]"].entries[-1]
        # OCV_bulk = solution["Bulk open-circuit voltage [V]"].entries[-1]

        del solution
        # Avoid division by zero if current is 0
        if abs(I) < 1e-6:
            new_R = default_R
        else:
            new_R = abs((OCV - V) / I)
        R_values.append(max(1e-6, new_R))
        OCVs.append(OCV)

    return R_values, OCVs

def solve_circuit_u_config(R_values, OCV_values, total_current, r_busbar, r_conn = 1e-5):
    """
    Solves the resistor network for a U-configuration (ladder network) with imposed current.
    
    This function replaces the generic MNA solver with a topology-specific 
    linear algebra solution (Ax = B), reducing complexity and overhead.

    Args:
        R_values (array-like): Internal resistances of each cell (Ohms).
        OCV_values (array-like): Open-circuit voltages of each cell (Volts).
        total_current (float): Total current flowing into the pack (Amps).
                               Positive = Discharging, Negative = Charging.
        r_busbar (float): Resistance of the busbar connecting adjacent cells (Ohms).
                          Assumes symmetry (top and bottom rails have same resistance).

    Returns:
        np.ndarray: Vector of individual cell currents [I_0, I_1, ...].
    """
    # Ensure inputs are numpy arrays
    R = np.array(R_values)
    OCV = np.array(OCV_values)
    N = len(R)

    # Initialize System Matrices: A * I = B
    # We have N unknowns (currents) and N equations.
    A = np.zeros((N, N))
    B = np.zeros(N)

    # 1. Fill KVL Equations (Rows 0 to N-2)
    # These represent the voltage loops between adjacent cells.
    for k in range(N - 1):
        # Current flowing out of cell k causes a drop across R_k
        A[k, k] = R[k] + r_conn
        
        # Current flowing out of cell k+1 causes a 'gain' relative to the loop
        # plus the drop across the two busbar segments (top and bottom)
        A[k, k+1] = -(R[k+1] + 2 * r_busbar)
        
        # Any current from cells further down the line (k+2 onwards) also 
        # flows through the busbar segments between k and k+1.
        if k + 2 < N:
            A[k, k+2:] = -2 * r_busbar
            
        # The RHS is the difference in Open Circuit Voltages
        B[k] = OCV[k] - OCV[k+1]

    # 2. Fill KCL Equation (Last Row)
    # The sum of all currents must equal the total pack current.
    A[N-1, :] = 1
    B[N-1] = total_current

    # 3. Solve the linear system
    try:
        cell_currents = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        raise ValueError("Singular matrix: Check for zero resistances or invalid inputs.")

    return cell_currents

def generate_protocol_from_experiment(experiment):
    """

    Args:
        experiment (pybamm.Experiment):
            The experiment to generate the protocol from.

    Returns:
        protocol (list):
            a sequence of terminal currents to apply at each timestep
        terminations (list):
            a sequence voltage terminations for each step

    """
    protocol = []
    terminations = []
    step_types = []
    for i, step in enumerate(experiment.steps):
        proto = []
        t = step.duration
        dt = step.period
        termination = step.termination
        step_type = type(step).__name__.lower()
        if step_type not in ["current", "power"]:
            raise ValueError("Only current and power operations are supported")
        else:
            if not isinstance(step.value, pybamm.Interpolant):
                I = step.value
                proto.extend([I] * int(np.round(t, 5) / np.round(dt, 5)))
                if i == 0:
                    # Include initial state when not drive cycle, first op
                    proto = [proto[0]] + proto
            else:
                proto.extend(step.value.y.tolist())
            if len(termination) > 0:
                for term in termination:
                    if isinstance(
                        term, pybamm.experiment.step.step_termination.VoltageTermination
                    ):
                        terminations.append(term.value)
            else:
                terminations.append([])

        protocol.append(proto)
        step_types.append(step_type)

    return protocol, terminations, step_types

def run_pack(model, 
             params,
             rate, 
             total_time, 
             time_steps,
            #  experiment,
             num_cells_parallel, 
             r_busbar,
             initial_soc= 1.0,
             save_memory=False,
             solver = pybamm.IDAKLUSolver(atol=1e-4, rtol=1e-4),):
    
    params.update({"Current function [A]": "[input]"})
    capacity = params["Nominal cell capacity [A.h]"]
    total_current = rate * capacity * num_cells_parallel  # Amperes (Discharge is positive in this convention)

    v_cut_lower = params["Lower voltage cut-off [V]"]
    v_cut_higher = params["Upper voltage cut-off [V]"]
    # change cut off voltages to avoid the pybamm solver stopping too early
    params.update({"Lower voltage cut-off [V]": v_cut_lower - 0.5,
                   "Upper voltage cut-off [V]": v_cut_higher + 0.5})
    
    # Simulation settings
    dt = total_time / time_steps
    # Initialize simulations
    sims = [pybamm.Simulation(model, parameter_values=params,
                            solver = solver.copy(),
                            ) for _ in range(num_cells_parallel)]
    # Storage for plotting
    history = {
        "time": [],
        "currents": [[] for _ in range(num_cells_parallel)],
        "voltages": [[] for _ in range(num_cells_parallel)],
        "resistances": [[] for _ in range(num_cells_parallel)]
    }

    print(f"Starting simulation for {num_cells_parallel} cells in parallel...")
    # Initialize all simulations
    for i in range(num_cells_parallel):
        sims[i].build(initial_soc=initial_soc,
                      inputs={"Current function [A]": total_current / num_cells_parallel})
        sims[i].step(dt=0.1, 
                    inputs={"Current function [A]": total_current / num_cells_parallel})

    sols = [sim.solution for sim in sims]

    current_time = 0
    for t in range(time_steps):
        print(f"Time step {t+1}/{time_steps} at t={current_time:.2f}s", end='\r')
        R_values, OCVs = update_internal_resistance(sims)
        # Solve Circuit
        currents = solve_circuit_u_config(R_values, OCVs, total_current, r_busbar)
        voltages = np.array([])
        for i in range(num_cells_parallel):

            if save_memory:
                sims[i].step(
                        dt=dt, 
                        t_eval = [0, dt], # veen more extreme save of time and data.
                        save = False, # True saves more data but it gets slower at each step
                        inputs={"Current function [A]": currents[i]}
                        )
            else:
                sims[i].step(
                        dt=dt, 
                        inputs={"Current function [A]": currents[i]}
                        )
            
            voltages = np.append(voltages, sims[i].solution["Voltage [V]"].entries[-1])

            history["resistances"][i].append(R_values[i])
            history["currents"][i].append(currents[i])
            history["voltages"][i].append(voltages[i])
        
        history["time"].append(current_time)
        if np.any(voltages <= v_cut_lower):
            print(f"\nLower cut-off voltage at time {current_time:.2f}s. Stopping simulation.")
            return sols, history

        if np.any(voltages >= v_cut_higher):
            print(f"\nUpper cut-off voltage at time {current_time:.2f}s. Stopping simulation.")
            return sols, history
            # When we reach a resting point we can modify the hysteresis decay rate to 0 to stop the drift.
            # We need to set it up as a input so that it can be quickly changed here.
            # d. Store Data
        current_time += dt

        sols = [sim.solution for sim in sims]

    return sols, history