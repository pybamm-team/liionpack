from utils import *
from solve_circuit import *

def run_pack(model, 
             parameters,
             experiment,
             num_cells_parallel, 
             r_busbar = 1e-4,
             r_conn = 1e-5,
             r_terminal = 1e-5,
             terminals = "left",
             initial_soc = None,
             save_memory=False,
             solver = pybamm.IDAKLUSolver(atol=1e-4, rtol=1e-4),
             var_pts = None,
             Ri = 0.01):
    
    # Setup simulations and parameters.
    sims, v_cut_lower, v_cut_higher, _ = setup_sims_and_params(
        model, parameters, solver, var_pts, num_cells_parallel)

    # transfrom the protocol in natural language into vectors of currents and termination conditions
    (total_current_in_each_section, 
     termination_conditions, _, dt) = generate_protocol_from_experiment(experiment)

    # Store for plotting
    history = create_hystory_dict(num_cells_parallel)

    print(f"Starting simulation for {num_cells_parallel} cells in parallel...")
    # Initialize all simulations
    for i in range(num_cells_parallel):

        if initial_soc is not None: # if inital soc is given.
            sims[i].build(initial_soc=initial_soc,
                        inputs={"Current function [A]": total_current_in_each_section[0][0] / num_cells_parallel})
        else: # otherwise use whatever was in the model
            sims[i].build(inputs={"Current function [A]": total_current_in_each_section[0][0] / num_cells_parallel})

        # mini step to initialize everything properly and get internal resistances
        sims[i].step(dt=1e-4, 
                     t_eval = [0, 1e-4], # even more extreme save of time and data.
                     save = False, # True saves more data but it gets slower at each step
                     inputs={"Current function [A]": total_current_in_each_section[0][0] / num_cells_parallel})

    updated_time = 0
    for s in range(len(total_current_in_each_section)):
        print(f"Starting section {s+1}/{len(total_current_in_each_section)}...")
        v_cut_lower = None
        v_cut_higher = None
        # If there is a termination condition on top of the time duration only
        section_profile = total_current_in_each_section[s]
        section_termination = termination_conditions[s]

        section_current = next((val for val in section_profile if abs(val) > 1e-12), 0.0)

        if section_termination is not None:
            if section_termination["type"] == "pack_voltage":
                if section_current < 0:
                    print(f"Termination: charge until pack terminal voltage reaches {section_termination['value']} V")
                    v_cut_higher = section_termination["value"]
                elif section_current > 0:
                    print(f"Termination: discharge until pack terminal voltage reaches {section_termination['value']} V")
                    v_cut_lower = section_termination["value"]
            

        for t in range(len(section_profile)):
            total_current = section_profile[t]
            use_linear_solver = False
            print(f"Time step {t+1}/{len(total_current_in_each_section[s])} at t={updated_time:.2f}s", end='\r')

            R_values, OCVs = update_internal_resistance(sims, Ri, history)
            circuit_out = calculate_currents(
                t=t,
                total_current=total_current,
                r_busbar=r_busbar,
                r_conn=r_conn,
                R_values=R_values,
                OCVs=OCVs,
                sims=sims,
                use_linear_solver=use_linear_solver,
                history=history,
                positive_terminal="NT",
                negative_terminal="PT",
                r_terminal_contact=r_terminal,
                terminals=terminals,
            )

            currents = circuit_out["I_cells"]
            pack_voltage = circuit_out["pack_voltage"]
            
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
                            save = True,
                            inputs={"Current function [A]": currents[i]}
                            )

                history["R_internal"][i].append(R_values[i])
                history["I_cell"][i].append(currents[i])
                history["V_cell"][i].append(sims[i].solution["Voltage [V]"].entries[-1])
                history["T_cell"][i].append(sims[i].solution["Cell temperature [C]"].entries[-1])
                history["SOC"][i].append(get_soc(sims[i]))

            # terminal_volt = sims[0].solution["Voltage [V]"].entries[-1] - total_current * r_terminal * 2
            history["V_pack"].append(pack_voltage)
            history["time"].append(updated_time)

            updated_time += dt #If code breaks, displayed time be one step ahead.

            # print(f"time = {updated_time:.2f} s")
            # print(f"total_current = {total_current}")
            # print(f"pack_voltage = {pack_voltage}")
            # print(f"v_cut_lower = {v_cut_lower}")
            # print(f"v_cut_higher = {v_cut_higher}")

            if t > 0:
                if v_cut_lower is not None and pack_voltage <= v_cut_lower:
                    print(f"\nLower cut-off voltage at time {updated_time:.2f}s. Starting next section.")
                    break

                if v_cut_higher is not None and pack_voltage >= v_cut_higher:
                    print(f"\nUpper cut-off voltage at time {updated_time:.2f}s. Starting next section.")
                    break
            # updated_time += dt

    sols = [sim.solution for sim in sims]

    return sols, history