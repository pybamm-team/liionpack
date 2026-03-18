import numpy as np
from scipy.optimize import root


# ==========================================================
# Internal resistance estimation
# ==========================================================
def update_internal_resistance(sims, Ri, history=None):
    """
    Updates 'R' using R = |(OCV - V) / I|.
    If current is ~0, reuse last R (or Ri default).
    """
    R_values = []
    OCVs = []
    for idx, sim in enumerate(sims):
        sol = sim.solution
        V = sol["Voltage [V]"].entries[-1]
        I = sol["Current [A]"].entries[-1]
        OCV = sol["Surface open-circuit voltage [V]"].entries[-1]

        if abs(I) < 1e-6:
            if history is not None and len(history["R_internal"][idx]) > 0:
                new_R = history["R_internal"][idx][-1]
            else:
                new_R = Ri
        else:
            new_R = abs((OCV - V) / I)

        R_values.append(max(1e-9, float(new_R)))
        OCVs.append(float(OCV))

    return R_values, OCVs


# ==========================================================
# Build nodes
# ==========================================================
def _build_nodes(N, terminals="left"):
    nodes = {}
    k = 0

    nodes["NT"] = k  # positive external terminal in the netlist convention
    k += 1
    nodes["PT"] = k  # negative external terminal in the netlist convention
    k += 1

    use_true_middle = (terminals == "middle" and N % 2 == 0)
    if use_true_middle:
        nodes["TM"] = k  # top midpoint node
        k += 1
        nodes["BM"] = k  # bottom midpoint node
        k += 1

    for i in range(N):
        nodes[f"T{i}"] = k
        k += 1
    for i in range(N):
        nodes[f"X{i}"] = k
        k += 1
    for i in range(N):
        nodes[f"Y{i}"] = k
        k += 1
    for i in range(N):
        nodes[f"B{i}"] = k
        k += 1

    return nodes


# ==========================================================
# Resolve terminal contact locations
# Supported:
#   "left"
#   "right"
#   "left-right"
#   "right-left"
#   "middle"
#   [k_pos, k_neg]
# ==========================================================
def _resolve_terminal_contact_nodes(N, terminals):
    if terminals == "left":
        k_pos = 0
        k_neg = 0

    elif terminals == "right":
        k_pos = N - 1
        k_neg = N - 1

    elif terminals == "left-right":
        k_pos = 0
        k_neg = N - 1

    elif terminals == "right-left":
        k_pos = N - 1
        k_neg = 0

    elif terminals == "middle":
        # Practical default middle:
        # for N=4 -> index 1
        # use [2, 2] explicitly if you want the other center node
        k_pos = (N - 1) // 2
        k_neg = (N - 1) // 2

    elif isinstance(terminals, (list, tuple, np.ndarray)) and len(terminals) == 2:
        k_pos = int(terminals[0])
        k_neg = int(terminals[1])

    else:
        raise ValueError(
            "Unsupported terminal configuration. "
            "Use 'left', 'right', 'left-right', 'right-left', 'middle', or [k_pos, k_neg]."
        )

    if not (0 <= k_pos < N and 0 <= k_neg < N):
        raise ValueError(f"Terminal indices out of range: [{k_pos}, {k_neg}] for N={N}")

    return f"T{k_pos}", f"B{k_neg}"


# ==========================================================
# Shared rail / terminal stamping
# ==========================================================
def _append_busbars_and_terminal_contacts(net, N, r_busbar, r_terminal_contact, terminals):
    use_true_middle = (terminals == "middle" and N % 2 == 0)

    # --------------------------------------------------
    # TRUE MIDDLE CONTACT FOR EVEN N
    # --------------------------------------------------
    if use_true_middle:
        left_mid = N // 2 - 1
        right_mid = N // 2

        # ---------- top rail ----------
        for i in range(left_mid):
            net.append(("R", f"T{i}", f"T{i+1}", float(r_busbar)))

        net.append(("R", f"T{left_mid}", "TM", float(r_busbar) / 2.0))
        net.append(("R", "TM", f"T{right_mid}", float(r_busbar) / 2.0))

        for i in range(right_mid, N - 1):
            net.append(("R", f"T{i}", f"T{i+1}", float(r_busbar)))

        net.append(("R", "NT", "TM", float(r_terminal_contact)))

        # ---------- bottom rail ----------
        for i in range(left_mid):
            net.append(("R", f"B{i}", f"B{i+1}", float(r_busbar)))

        net.append(("R", f"B{left_mid}", "BM", float(r_busbar) / 2.0))
        net.append(("R", "BM", f"B{right_mid}", float(r_busbar) / 2.0))

        for i in range(right_mid, N - 1):
            net.append(("R", f"B{i}", f"B{i+1}", float(r_busbar)))

        net.append(("R", "BM", "PT", float(r_terminal_contact)))

    # --------------------------------------------------
    # ALL OTHER CASES (existing node-based contacts)
    # --------------------------------------------------
    else:
        top_contact_node, bottom_contact_node = _resolve_terminal_contact_nodes(N, terminals)

        net.append(("R", "NT", top_contact_node, float(r_terminal_contact)))
        net.append(("R", bottom_contact_node, "PT", float(r_terminal_contact)))

        for i in range(N - 1):
            net.append(("R", f"T{i}", f"T{i+1}", float(r_busbar)))

        for i in range(N - 1):
            net.append(("R", f"B{i}", f"B{i+1}", float(r_busbar)))

    return net


# ==========================================================
# Linear netlist builder: OCV + Ri with MNA
# ==========================================================
def _build_liionpack_like_netlist(
    R_values,
    OCV_values,
    I_pack,
    r_busbar,
    r_conn,
    r_terminal_contact=1e-10,
    terminals="left",
):
    N = len(R_values)
    net = []

    # pack current source: PT -> NT
    net.append(("I", "PT", "NT", float(I_pack)))
    _append_busbars_and_terminal_contacts(net, N, r_busbar, r_terminal_contact, terminals)

    # per-cell elements
    for i in range(N):
        Ri = float(max(1e-9, R_values[i]))
        OCV = float(OCV_values[i])

        net.append(("R", f"X{i}", f"T{i}", float(r_conn)))
        net.append(("V", f"Y{i}", f"X{i}", OCV))
        net.append(("R", f"B{i}", f"Y{i}", Ri))

    return net


# ==========================================================
# Nonlinear netlist builder: Rc + measured cell voltage source
# ==========================================================
def _build_voltage_driven_netlist(
    V_cell_values,
    I_pack,
    r_busbar,
    r_conn,
    r_terminal_contact=1e-10,
    terminals="left",
):
    """
    Build a netlist where each cell is represented by:
        busbar node T_i -- Rc -- X_i -- ideal cell voltage source -- B_i

    The ideal cell voltage source uses the PyBaMM terminal voltage directly,
    while Rc, busbars and terminal contacts are kept in the external network.
    """
    N = len(V_cell_values)
    net = []

    net.append(("I", "PT", "NT", float(I_pack)))
    _append_busbars_and_terminal_contacts(net, N, r_busbar, r_terminal_contact, terminals)

    for i in range(N):
        V_cell = float(V_cell_values[i])
        net.append(("R", f"X{i}", f"T{i}", float(r_conn)))
        net.append(("V", f"B{i}", f"X{i}", V_cell))

    return net


# ==========================================================
# Full MNA solve with voltage sources
# ==========================================================
def _solve_mna_full(netlist, nodes, ground="PT"):
    node_count = len(nodes)

    vsrcs = [(n1, n2, val) for (typ, n1, n2, val) in netlist if typ == "V"]
    m = len(vsrcs)

    # Only keep nodes that are actually present in the current netlist.
    # This is essential for the nonlinear voltage-driven netlist, where Y{i}
    # nodes are defined in the master node table but are intentionally unused.
    # Keeping unused nodes in the MNA system creates zero rows/cols and makes
    # the matrix singular.
    active_node_names = {ground}
    for typ, n1, n2, _ in netlist:
        active_node_names.add(n1)
        active_node_names.add(n2)

    g = nodes[ground]
    node_map = {}
    k = 0
    for name, idx in nodes.items():
        if idx == g:
            continue
        if name not in active_node_names:
            continue
        node_map[name] = k
        k += 1

    n = len(node_map)

    A = np.zeros((n + m, n + m), dtype=float)
    z = np.zeros(n + m, dtype=float)

    def add_conductance(a, b, gval):
        ai = node_map.get(a, None)
        bi = node_map.get(b, None)

        if ai is not None:
            A[ai, ai] += gval
        if bi is not None:
            A[bi, bi] += gval
        if ai is not None and bi is not None:
            A[ai, bi] -= gval
            A[bi, ai] -= gval

    def add_current_source(a, b, I):
        ai = node_map.get(a, None)
        bi = node_map.get(b, None)
        if ai is not None:
            z[ai] -= I
        if bi is not None:
            z[bi] += I

    for typ, n1, n2, val in netlist:
        if typ == "R":
            gval = 1.0 / float(val)
            add_conductance(n1, n2, gval)
        elif typ == "I":
            add_current_source(n1, n2, float(val))

    for k_src, (p, n_node, Vs) in enumerate(vsrcs):
        row = n + k_src

        p_i = node_map.get(p, None)
        n_i = node_map.get(n_node, None)

        if p_i is not None:
            A[p_i, row] += 1.0
            A[row, p_i] += 1.0
        if n_i is not None:
            A[n_i, row] -= 1.0
            A[row, n_i] -= 1.0

        z[row] = float(Vs)

    x = np.linalg.solve(A, z)

    V = np.zeros(node_count, dtype=float)
    for name, red_idx in node_map.items():
        V[nodes[name]] = x[red_idx]
    V[g] = 0.0

    I_vsrc = x[n:] if m > 0 else np.array([])
    return V, I_vsrc


# ==========================================================
# Pack terminal voltage
# ==========================================================
def _compute_pack_terminal_voltage(V, nodes, positive_terminal="NT", negative_terminal="PT"):
    return float(-V[nodes[positive_terminal]] + V[nodes[negative_terminal]])


# ==========================================================
# Extract cell currents through Rc
# ==========================================================
def _extract_cell_currents_from_rc(V, nodes, r_conn, N):
    I_cells = np.zeros(N, dtype=float)
    Rc = float(r_conn)

    for i in range(N):
        vt = V[nodes[f"T{i}"]]
        vx = V[nodes[f"X{i}"]]
        I_cells[i] = (vt - vx) / Rc

    return I_cells


# ==========================================================
# Linear MNA solver wrapper
# ==========================================================
def _solve_circuit_linear_mna(
    total_current,
    r_busbar,
    r_conn,
    R_values,
    OCVs,
    positive_terminal="NT",
    negative_terminal="PT",
    r_terminal_contact=1e-10,
    terminals="left",
):
    N = len(R_values)
    nodes = _build_nodes(N, terminals=terminals)

    netlist = _build_liionpack_like_netlist(
        R_values=R_values,
        OCV_values=OCVs,
        I_pack=total_current,
        r_busbar=r_busbar,
        r_conn=r_conn,
        r_terminal_contact=r_terminal_contact,
        terminals=terminals,
    )

    V, _ = _solve_mna_full(netlist, nodes, ground=negative_terminal)
    I_cells = _extract_cell_currents_from_rc(V, nodes, r_conn, N)
    pack_voltage = _compute_pack_terminal_voltage(
        V,
        nodes,
        positive_terminal=positive_terminal,
        negative_terminal=negative_terminal,
    )

    return {
        "I_cells": I_cells,
        "pack_voltage": pack_voltage,
        "node_voltages": V,
        "nodes": nodes,
    }


# ==========================================================
# Nonlinear MNA solver wrapper
# ==========================================================
def _solve_circuit_non_linear_mna(
    total_current,
    r_busbar,
    r_conn,
    sims,
    R_values,
    OCVs,
    positive_terminal="NT",
    negative_terminal="PT",
    r_terminal_contact=1e-10,
    terminals="left",
):
    """
    Nonlinear solve using the old root-solver idea, but with the new configurable MNA network.

    Important:
    This intentionally uses a shallow list copy of sims and tiny trial steps, matching the old logic.
    That means trial evaluations can still perturb the underlying simulation states.
    """
    N = len(sims)
    nodes = _build_nodes(N, terminals=terminals)

    linear_out = _solve_circuit_linear_mna(
        total_current=total_current,
        r_busbar=r_busbar,
        r_conn=r_conn,
        R_values=R_values,
        OCVs=OCVs,
        positive_terminal=positive_terminal,
        negative_terminal=negative_terminal,
        r_terminal_contact=r_terminal_contact,
        terminals=terminals,
    )
    x0 = linear_out["I_cells"]

    def equations(I_vector):
        sims_trial = sims.copy()  # intentional shallow copy to match the old behavior
        V_cells = np.zeros(N, dtype=float)

        for i in range(N):
            sims_trial[i].step(
                dt=2e-9,
                save=True,
                inputs={"Current function [A]": float(I_vector[i])},
            )
            sol_trial = sims_trial[i].solution
            V_cells[i] = sol_trial["Voltage [V]"].entries[-1]

        netlist = _build_voltage_driven_netlist(
            V_cell_values=V_cells,
            I_pack=total_current,
            r_busbar=r_busbar,
            r_conn=r_conn,
            r_terminal_contact=r_terminal_contact,
            terminals=terminals,
        )

        V_net, _ = _solve_mna_full(netlist, nodes, ground=negative_terminal)
        I_net = _extract_cell_currents_from_rc(V_net, nodes, r_conn, N)

        return I_net - I_vector

    solution_currents = root(equations, x0, method="hybr")

    if not solution_currents.success:
        print(
            f"Warning: Non-linear solver failed ({solution_currents.message}). "
            "Returning linear MNA solution."
        )
        return linear_out

    final_currents = solution_currents.x

    # Re-evaluate once at the converged current vector to get pack voltage/node voltages
    sims_trial = sims.copy()  # intentional shallow copy to match the old behavior
    V_cells = np.zeros(N, dtype=float)
    for i in range(N):
        sims_trial[i].step(
            dt=2e-9,
            save=True,
            inputs={"Current function [A]": float(final_currents[i])},
        )
        sol_trial = sims_trial[i].solution
        V_cells[i] = sol_trial["Voltage [V]"].entries[-1]

    netlist = _build_voltage_driven_netlist(
        V_cell_values=V_cells,
        I_pack=total_current,
        r_busbar=r_busbar,
        r_conn=r_conn,
        r_terminal_contact=r_terminal_contact,
        terminals=terminals,
    )
    V_net, _ = _solve_mna_full(netlist, nodes, ground=negative_terminal)
    pack_voltage = _compute_pack_terminal_voltage(
        V_net,
        nodes,
        positive_terminal=positive_terminal,
        negative_terminal=negative_terminal,
    )

    return {
        "I_cells": final_currents,
        "pack_voltage": pack_voltage,
        "node_voltages": V_net,
        "nodes": nodes,
    }


# ==========================================================
# Solver switch logic
# ==========================================================
def calculate_currents(
    t,
    total_current,
    r_busbar,
    r_conn,
    R_values,
    OCVs,
    sims,
    use_linear_solver,
    history,
    positive_terminal="NT",
    negative_terminal="PT",
    r_terminal_contact=1e-10,
    terminals="left",
):
    if abs(total_current) > 1e-1:
        use_linear_solver = True
    elif abs(total_current) <= 1e-1 and t < 3:
        use_linear_solver = False
    else:
        use_linear_solver = False
        if history is not None and "R_internal" in history:
            enough_history = all(len(res) >= 3 for res in history["R_internal"])
            if enough_history:
                last_3_R = np.array([res[-3:] for res in history["R_internal"]], dtype=float)
                r_max = np.max(last_3_R, axis=1)
                r_min = np.min(last_3_R, axis=1)
                variation = (r_max - r_min) / (r_min + 1e-9)
                if np.all(variation <= 0.1):
                    use_linear_solver = True

    if use_linear_solver:
        return _solve_circuit_linear_mna(
            total_current=total_current,
            r_busbar=r_busbar,
            r_conn=r_conn,
            R_values=R_values,
            OCVs=OCVs,
            positive_terminal=positive_terminal,
            negative_terminal=negative_terminal,
            r_terminal_contact=r_terminal_contact,
            terminals=terminals,
        )

    return _solve_circuit_non_linear_mna(
        total_current=total_current,
        r_busbar=r_busbar,
        r_conn=r_conn,
        sims=sims,
        R_values=R_values,
        OCVs=OCVs,
        positive_terminal=positive_terminal,
        negative_terminal=negative_terminal,
        r_terminal_contact=r_terminal_contact,
        terminals=terminals,
    )
