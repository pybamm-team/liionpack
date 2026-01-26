from utils import *
from solve_circuit import *

def run_pack(model, 
             parameters,
             experiment,
             num_cells_parallel, 
             r_busbar = 1e-4,
             r_conn = 1e-5,
             r_terminal = 1e-5,
             initial_soc= 1,
             save_memory=False,
             solver = pybamm.IDAKLUSolver(atol=1e-4, rtol=1e-4),
             var_pts = None):
    
    sims, v_cut_lower, v_cut_higher, _ = setup_sims_and_params(
        model, parameters, solver, var_pts, num_cells_parallel)

    (total_current_in_each_section, 
     termination_conditions, _, dt) = generate_protocol_from_experiment(experiment)

    # Store for plotting
    history = create_hystory_dict(num_cells_parallel)

    print(f"Starting simulation for {num_cells_parallel} cells in parallel...")
    # Initialize all simulations
    for i in range(num_cells_parallel):
        sims[i].build(initial_soc=initial_soc,
                      inputs={"Current function [A]": total_current_in_each_section[0][0] / num_cells_parallel})
        
        sims[i].step(dt=1e-4, 
                     t_eval = [0, 1e-4], # even more extreme save of time and data.
                     save = False, # True saves more data but it gets slower at each step
                     inputs={"Current function [A]": total_current_in_each_section[0][0] / num_cells_parallel})

    updated_time = 0
    for s in range(len(total_current_in_each_section)):
        print(f"Starting section {s+1}/{len(total_current_in_each_section)}...")
        # If there is a termination condition on top of the time duration only
        if termination_conditions[s]: 
            if total_current_in_each_section[s][0] < 0:
                print(f"Termination: charge until {termination_conditions[s]} V")
                v_cut_higher = termination_conditions[s]
            elif total_current_in_each_section[s][0] > 0:
                print(f"Termination: discharge until {termination_conditions[s]} V")
                v_cut_lower = termination_conditions[s]
            

        for t in range(len(total_current_in_each_section[s])):
            total_current = total_current_in_each_section[s][t]
            use_linear_solver = False
            print(f"Time step {t+1}/{len(total_current_in_each_section[s])} at t={updated_time:.2f}s", end='\r')

            R_values, OCVs = update_internal_resistance(sims, history)
            currents = calculate_currents(
                       t, total_current, r_busbar, r_conn,
                       R_values, OCVs, sims,
                       use_linear_solver, history)
            
            for i in range(num_cells_parallel):
                if save_memory:
                    sims[i].step(
                            dt=dt, 
                            t_eval = [0, dt], # even more extreme save of time and data.
                            save = False, # True saves more data but it gets slower at each step
                            inputs={"Current function [A]": currents[i]}
                            )
                else:
                    sims[i].step(
                            dt=dt, 
                            inputs={"Current function [A]": currents[i]}
                            )

                history["resistances"][i].append(R_values[i])
                history["currents"][i].append(currents[i])
                history["cell_voltages"][i].append(sims[i].solution["Voltage [V]"].entries[-1])
                history["temperatures"][i].append(sims[i].solution["Cell temperature [C]"].entries[-1])

            terminal_volt = sims[0].solution["Voltage [V]"].entries[-1] - total_current * r_terminal * 2
            history["terminal_voltage"].append(terminal_volt)
            history["time"].append(updated_time)

            if terminal_volt <= v_cut_lower:
                print(f"\nLower cut-off voltage at time {updated_time:.2f}s. Starting next section.")
                break

            if terminal_volt >= v_cut_higher:
                print(f"\nUpper cut-off voltage at time {updated_time:.2f}s. Starting next section.")
                break
            updated_time += dt

    sols = [sim.solution for sim in sims]

    return sols, history