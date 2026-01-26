from scipy.optimize import root
import casadi
import scipy as sp
import numpy as np
import pybamm

def update_internal_resistance(sims, history=None):
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
        OCV = solution["Surface open-circuit voltage [V]"].entries[-1]
        del solution
        # Avoid division by zero if current is 0. Use the previous R value in that case
        # It's physically reasonable to assume R doesn't change if no current is flowing
        if abs(I) < 1e-6:
            if len(history["resistances"][sims.index(sim)]) > 0:
                new_R = history["resistances"][sims.index(sim)][-1]
            else:
                new_R = 0.03  # Default initial resistance if no history
        else:
            new_R = abs((OCV - V) / I)
        R_values.append(max(1e-6, new_R))
        OCVs.append(OCV)

    return R_values, OCVs

def solve_circuit_u_config_linear(R_values, OCV_values, total_current, r_busbar, r_conn = 1e-5):
    """
    Solves the resistor network for a U-configuration (ladder network) with imposed current.
    
    This function replaces the generic MNA solver with a topology-specific 
    linear algebra solution (Ax = B), reducing complexity and overhead.

    Args:
        R_values (array-like): Internal resistances of each cell (Ohms).
        OCV_values (array-like): Open-circuit cell_voltages of each cell (Volts).
        total_current (float): Total current flowing into the pack (Amps).
                               Positive = Discharging, Negative = Charging.
        r_busbar (float): Resistance of the busbar connecting adjacent cells (Ohms).
                          Assumes symmetry (top and bottom rails have same resistance).
        r_conn (float): Resistance of the connection between each cell and the busbar (Ohms).

    Returns:
        np.ndarray: Vector of individual cell currents [I_0, I_1, ...].
    """
    # Ensure inputs are numpy arrays
    R = np.array(R_values)
    OCV = np.array(OCV_values)
    N = len(R)

    # Initialize System Matrices: A * I = B
    A = np.zeros((N, N))
    B = np.zeros(N)

    # 1. Fill KVL Equations (Rows 0 to N-2)
    for k in range(N - 1):
        # Current flowing out of cell k causes a drop across R_k
        A[k, k] = R[k] + r_conn

        A[k, k+1] = -(R[k+1] + 2 * r_busbar)
        
        if k + 2 < N:
            A[k, k+2:] = -2 * r_busbar
            
        B[k] = OCV[k] - OCV[k+1]

    # 2. Fill KCL Equation (Last Row)
    A[N-1, :] = 1
    B[N-1] = total_current

    # 3. Solve the linear system
    try:
        cell_currents = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        raise ValueError("Singular matrix: Check for zero resistances or invalid inputs.")

    return cell_currents

def solve_circuit_u_config_non_lin(sims, total_current, r_busbar, r_conn=1e-5):
    """
    Solves the U-configuration pack using a numerical root solver (Newton-Hybrid),
    strictly replicating the circuit topology defined in the linear matrix solver.
    
    Args:
        R_values (array): Internal resistances (Ohms).
        OCV_values (array): Open Circuit Voltages.
        total_current (float): Total pack current.
    """
    N = len(sims)

    def equations(I_vector):
        sims_trial = sims.copy()  # To avoid modifying original sims
        # sims_trial = [sim.copy() for sim in sims]
        V_cells = np.zeros(N)

        for i in range(N):
            sims_trial[i].step(
                dt=2e-9, 
                # t_eval = [0, 0.1], # even more extreme save of time and data.
                save = False, # True saves more data but it gets slower at each step
                inputs={"Current function [A]": I_vector[i]}
                )
            sol = sims_trial[i].solution
            V_cells[i] = sol["Voltage [V]"].entries[-1]
            del sol

        residuals = np.zeros(N)
        
        # 1. Calculate Voltages
        v_self = V_cells - I_vector * r_conn  # Adjusted to match actual voltages
        v_neighbor = V_cells

        # 2. Calculate Busbar Currents
        I_bus_flow = np.cumsum(I_vector[::-1])[::-1][1:] 

        # 3. KVL Loops (Indices 0 to N-2)
        v_bus_drops = 2 * r_busbar * I_bus_flow
        
        # We use v_self for k and v_neighbor for k+1 to strictly match the linear matrix rows
        residuals[:N-1] = v_self[:N-1] - v_neighbor[1:] + v_bus_drops

        # 4. KCL Equation (Last Index)
        residuals[N-1] = np.sum(I_vector) - total_current
        
        return residuals

    # Initial Guess: Equal distribution
    x0 = np.ones(N) * (total_current / N)

    # Solve
    sol = root(equations, x0, method='hybr')

    if not sol.success:
        print(f"Warning: Non-linear solver failed ({sol.message}). Returning equal split.")
        return np.ones(N) * (total_current / N)

    cell_currents = sol.x
    
    return cell_currents

def calculate_currents(t, total_current, r_busbar, r_conn, 
                       R_values, OCVs, sims,
                       use_linear_solver, history):
    # 1. The Check Condition
    # If current is applied we can use the linear solver
    if abs(total_current) > 1e-1:
        use_linear_solver = True
    elif abs(total_current) <= 1e-1 and t < 3:
        # During rest we need to check the resistance stability over last 3 steps
        use_linear_solver = False
    else:
        last_3_R = np.array([res[-3:] for res in history["resistances"]])
        # Calculate variation metrics per cell
        r_max = np.max(last_3_R, axis=1)
        r_min = np.min(last_3_R, axis=1)
        variation = (r_max - r_min) / (r_min + 1e-9)                
        # 3. The Switch Condition
        # We only use linear if ALL cells are stable within 10% (0.10)
        if np.all(variation <= 0.1):
            use_linear_solver = True
    
    if use_linear_solver:
        currents = solve_circuit_u_config_linear(
            R_values, OCVs, 
            total_current, r_busbar, r_conn
        )
    else:
        currents = solve_circuit_u_config_non_lin(
                                    sims, 
                                    total_current, r_busbar, r_conn)
    return currents
            
