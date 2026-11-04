"""Heuristic baselines of the paper: MS MINTOC, SUR NLP and Adjoint NLP.

Each one solves continuous relaxations of the benchmark model with Gurobi and returns a
switching sequence; run_revision_benchmarks.run_baselines costs that sequence with the same
fixed-sequence QP as every other method.
"""
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np

import miGoBuG_revision_patch as P


# ======================================================
# HEURISTIC BASELINES (SUR, Hybrid PMP, MS MINTOC)
# ======================================================
# Bounds of the benchmark model (miGoBuG_revision_patch.Case); the drivers pass the case values.
BASELINE_BOUNDS = dict(P_s_ub=500.0, P_g_ub=2000.0, P_dg_lb=0.0, P_dg_ub=2000.0, U_ub=4000.0)


def _relaxed_baseline_model(name, p_net, E_init, E_target, E_min, E_max, P_dg_init, S, Z, B, Delta,
                            eta_s, eta_dg, eta_g, X_g, base_grid_costs, dt, bounds=None, n_blocks=None, blocks=None):
    """Continuous relaxation of the benchmark model: same bounds, efficiencies and objective,
    topology weights x_c on the simplex, and the grid cost in perspective form per grid node,
    0.5*Delta_i*p_g^2 <= c_g * x_node.  The energy bounds are soft (penalized slacks), so every
    relaxation is feasible; the switching sequence it yields is evaluated with hard bounds.
    With `blocks`, one topology weight is shared by all intervals of a block (MS MINTOC)."""
    b = dict(BASELINE_BOUNDS, **(bounds or {}))
    N, ns, nz, ng, nk = len(p_net), len(S), len(Z), len(Delta), X_g.shape[0]
    blocks = blocks or [(t, t + 1) for t in range(N)]
    nb = len(blocks)
    blk = np.empty(N, dtype=int)
    for kb, (a, c) in enumerate(blocks):
        blk[a:c] = kb
    m = gp.Model(name, env=P.gurobi_env(0))
    p_dis = m.addVars(ns, N, lb=0.0)
    p_ch = m.addVars(ns, N, lb=0.0)
    for j in range(ns):
        for t in range(N):
            p_dis[j, t].ub = b["P_s_ub"] * eta_s[j]
            p_ch[j, t].ub = b["P_s_ub"] / eta_s[j]
    p_s = m.addVars(ns, N, lb=-b["P_s_ub"], ub=b["P_s_ub"])
    p_g = m.addVars(ng, N, lb=-b["P_g_ub"], ub=b["P_g_ub"])
    u = m.addVars(nz, N, lb=-b["U_ub"], ub=b["U_ub"])
    e = m.addVars(ns, N + 1, lb=-GRB.INFINITY)
    p_dg = m.addVars(nz, N + 1, lb=b["P_dg_lb"], ub=b["P_dg_ub"])
    x_c = m.addVars(nk, nb, lb=0.0, ub=1.0)
    x_node = m.addVars(ng, N, lb=0.0, ub=1.0)
    c_g = m.addVars(ng, N, lb=0.0)
    s_min, s_max = m.addVars(ns, N + 1, lb=0), m.addVars(ns, N + 1, lb=0)

    m.addConstrs((gp.quicksum(x_c[k, kb] for k in range(nk)) == 1.0 for kb in range(nb)))
    m.addConstrs((e[j, 0] == E_init[j] for j in range(ns)))
    m.addConstrs((p_dg[j, 0] == P_dg_init[j] for j in range(nz)))
    m.addConstrs((e[j, N] == E_target[j] for j in range(ns)))
    m.addConstrs((p_s[j, t] == p_dis[j, t] / eta_s[j] - p_ch[j, t] * eta_s[j] for j in range(ns) for t in range(N)))
    e_dyn = m.addConstrs((e[j, t + 1] == e[j, t] - p_s[j, t] * dt for j in range(ns) for t in range(N)))
    pdg_dyn = m.addConstrs((p_dg[j, t + 1] == p_dg[j, t] + u[j, t] * dt for j in range(nz) for t in range(N)))
    m.addConstrs((e[j, t] + s_min[j, t] >= E_min[j] for j in range(ns) for t in range(N + 1)))
    m.addConstrs((e[j, t] - s_max[j, t] <= E_max[j] for j in range(ns) for t in range(N + 1)))
    qcs = []
    for i in range(ng):
        for t in range(N):
            m.addConstr(x_node[i, t] == gp.quicksum(X_g[k, i] * x_c[k, blk[t]] for k in range(nk)))
            m.addConstr(p_g[i, t] <= b["P_g_ub"] * x_node[i, t])
            m.addConstr(p_g[i, t] >= -b["P_g_ub"] * x_node[i, t])
            qcs.append(m.addConstr(p_g[i, t] * p_g[i, t] <= (2.0 / Delta[i]) * c_g[i, t] * x_node[i, t]))
    for t in range(N):
        m.addConstr(p_net[t] == gp.quicksum(eta_dg[j] * p_dg[j, t] for j in range(nz)) +
                    gp.quicksum(p_dis[j, t] - p_ch[j, t] for j in range(ns)) +
                    gp.quicksum(eta_g[i] * p_g[i, t] for i in range(ng)))
    C_topo = X_g @ base_grid_costs
    base_obj = (gp.quicksum(0.5 * S[j] * (p_dis[j, t] ** 2 + p_ch[j, t] ** 2) for j in range(ns) for t in range(N)) +
                gp.quicksum(0.5 * Z[j] * p_dg[j, t] ** 2 for j in range(nz) for t in range(N)) +
                gp.quicksum(0.5 * B[j] * u[j, t] ** 2 for j in range(nz) for t in range(N)) +
                gp.quicksum(c_g[i, t] for i in range(ng) for t in range(N)) +
                gp.quicksum(C_topo[k] * x_c[k, kb] * (blocks[kb][1] - blocks[kb][0]) for k in range(nk) for kb in range(nb))) * dt
    slack_pen = gp.quicksum(1e6 * (s_min[j, t] + s_max[j, t]) for j in range(ns) for t in range(N + 1))
    m.setObjective(base_obj + slack_pen, GRB.MINIMIZE)
    V = dict(p_s=p_s, p_g=p_g, p_dg=p_dg, e=e, x_c=x_c, e_dyn=e_dyn, pdg_dyn=pdg_dyn, qcs=qcs, c_g=c_g, x_node=x_node, Delta=Delta,
             base_obj=base_obj, slack_pen=slack_pen, blocks=blocks, dt=dt, N=N, ns=ns, nz=nz, ng=ng, nk=nk, nb=nb)
    return m, V


def _fix_and_resolve(m, V, seq_blocks, t_start, label):
    """Fix the topology weights to the chosen sequence (one topology per block) and re-solve."""
    for kb in range(V["nb"]):
        for k in range(V["nk"]):
            V["x_c"][k, kb].lb = V["x_c"][k, kb].ub = 1.0 if k == seq_blocks[kb] else 0.0
    m.setObjective(V["base_obj"] + V["slack_pen"], GRB.MINIMIZE)
    m.optimize()
    t_total = time.time() - t_start
    if m.status != GRB.OPTIMAL:
        return {"success": False}
    N, ns, nz, ng = V["N"], V["ns"], V["nz"], V["ng"]
    m_star = np.zeros(N, dtype=int)
    for kb, (a, c) in enumerate(V["blocks"]):
        m_star[a:c] = seq_blocks[kb]
    print(f"{label} Success: Solved in {t_total:.2f}s")
    return {"success": True,
            "p_s": np.array([[V["p_s"][j, t].X for t in range(N)] for j in range(ns)]),
            "p_dg": np.array([[V["p_dg"][j, t].X for t in range(N)] for j in range(nz)]),
            "e": np.array([[V["e"][j, t].X for t in range(N + 1)] for j in range(ns)]),
            "p_g": np.array([[V["p_g"][i, t].X for t in range(N)] for i in range(ng)]) if ng > 0 else np.empty((0, N)),
            "m_star": m_star, "obj": m.ObjVal, "time": t_total}


def _qp_duals(m, V):
    """Duals of the dynamics at the relaxed solution: fix the topology weights there and write the
    grid cost as 0.5*Delta/x*p_g^2, an equivalent convex QP at that solution, whose duals are
    exact (the duals of the perspective QCP are only approximate and sometimes unavailable)."""
    x_val = {key: v.X for key, v in V["x_node"].items()}
    for v in V["x_c"].values():
        v.lb = v.ub = v.X
    for q in V["qcs"]:
        m.remove(q)
    grid = gp.QuadExpr()
    for (i, t), x in x_val.items():
        V["c_g"][i, t].ub = 0.0
        if x > 1e-9:
            grid += 0.5 * V["Delta"][i] / x * V["p_g"][i, t] * V["p_g"][i, t]
    m.setObjective(V["base_obj"] + V["slack_pen"] + grid * (V["dt"]), GRB.MINIMIZE)
    m.Params.QCPDual = 0
    m.optimize()


# ======================================================
# METHOD 3: SUR-SOS1 (combinatorial integral approximation by sum-up rounding)
# ======================================================
def solve_ems_sur_sos1(p_net, E_init, E_target, E_min, E_max, P_dg_init, S, Z, B, Delta, eta_s, eta_dg, eta_g, X_g, base_grid_costs, dt, bounds=None):
    print("\n--- Method 3: Starting SUR-SOS1 ---")
    t_start = time.time()
    m, V = _relaxed_baseline_model("SUR_relaxed", p_net, E_init, E_target, E_min, E_max, P_dg_init, S, Z, B, Delta,
                                   eta_s, eta_dg, eta_g, X_g, base_grid_costs, dt, bounds)
    m.optimize()
    if m.status != GRB.OPTIMAL:
        return {"success": False}
    N, nk = V["N"], V["nk"]
    x_rel = np.array([[V["x_c"][k, t].X for t in range(N)] for k in range(nk)])
    # SUR-SOS1: activate the topology with the largest accumulated deficit
    w = np.zeros((nk, N))
    for t in range(N):
        deficit = x_rel[:, :t + 1].sum(axis=1) - w[:, :t].sum(axis=1)
        w[np.argmax(deficit), t] = 1.0
    return _fix_and_resolve(m, V, np.argmax(w, axis=0), t_start, "SUR-SOS1")


# ======================================================
# METHOD 4: HYBRID PMP (direct NLP -> adjoints -> pointwise Hamiltonian minimization)
# ======================================================
def solve_ems_hybrid_pmp(p_net, E_init, E_target, E_min, E_max, P_dg_init, S, Z, B, Delta, eta_s, eta_dg, eta_g, X_g, base_grid_costs, dt, bounds=None):
    print("\n--- Method 4: Starting Hybrid PMP (Adjoints) ---")
    t_start = time.time()
    m, V = _relaxed_baseline_model("Hybrid_relaxed", p_net, E_init, E_target, E_min, E_max, P_dg_init, S, Z, B, Delta,
                                   eta_s, eta_dg, eta_g, X_g, base_grid_costs, dt, bounds)
    m.optimize()
    if m.status != GRB.OPTIMAL:
        return {"success": False}
    N, ns, nz, ng, nk = V["N"], V["ns"], V["nz"], V["ng"], V["nk"]
    C_topo = X_g @ base_grid_costs
    pdg_traj = np.array([[V["p_dg"][j, t].X for t in range(N)] for j in range(nz)])
    _qp_duals(m, V)                     # exact duals of the dynamics at the relaxed solution
    if m.status != GRB.OPTIMAL:
        return {"success": False}
    m_star = np.zeros(N, dtype=int)
    for t in range(N):
        # co-state = multiplier of the dynamics row x[t+1] - x[t] - dt*f = 0 (Gurobi Pi = dJ/drhs)
        lam_e = np.array([V["e_dyn"][j, t].Pi for j in range(ns)])
        lam_pdg = np.array([V["pdg_dyn"][j, t].Pi for j in range(nz)])
        req = p_net[t] - eta_dg @ pdg_traj[:, t]
        u_opt = -lam_pdg / B
        best, best_k = np.inf, 0
        for k in range(nk):
            act = X_g[k] > 0.5
            # stationary point of the topology Hamiltonian under the power balance
            # sum(eta_s*p_s) + sum(eta_g*p_g) = req (storage efficiency in discharge direction)
            denom = np.sum(eta_s ** 2 / S) + np.sum(eta_g[act] ** 2 / Delta[act])
            nu = (req - np.sum(eta_s * lam_e / S)) / denom
            p_s_opt = (lam_e + nu * eta_s) / S
            p_g_opt = np.where(act, nu * eta_g / Delta, 0.0)
            H = (0.5 * np.sum(S * p_s_opt ** 2) + 0.5 * np.sum(Delta * p_g_opt ** 2) + 0.5 * np.sum(B * u_opt ** 2) +
                 C_topo[k] - np.sum(lam_e * p_s_opt) + np.sum(lam_pdg * u_opt))
            if H < best:
                best, best_k = H, k
        m_star[t] = best_k
    return _fix_and_resolve(m, V, m_star, t_start, "Hybrid PMP")


# ======================================================
# METHOD 5: MS MINTOC (adaptive control grid + penalty homotopy)
# ======================================================
def solve_ems_ms_mintoc(p_net, E_init, E_target, E_min, E_max, P_dg_init, S, Z, B, Delta, eta_s, eta_dg, eta_g, X_g, base_grid_costs, dt, initial_steps_per_ms=4, bounds=None):
    print("\n--- Method 5: MS MINTOC (Adaptive Grid + Penalty Homotopy) ---")
    t_start = time.time()
    N, nk = len(p_net), X_g.shape[0]
    C_topo = X_g @ base_grid_costs
    blocks = [(i, min(i + initial_steps_per_ms, N)) for i in range(0, N, initial_steps_per_ms)]
    max_grid_iterations = 5
    for grid_iter in range(max_grid_iterations):
        m, V = _relaxed_baseline_model(f"MINTOC_relaxed_{grid_iter}", p_net, E_init, E_target, E_min, E_max, P_dg_init,
                                       S, Z, B, Delta, eta_s, eta_dg, eta_g, X_g, base_grid_costs, dt, bounds, blocks=blocks)
        m.optimize()
        if m.status != GRB.OPTIMAL:
            return {"success": False}
        x_val = np.array([[V["x_c"][k, kb].X for kb in range(V["nb"])] for k in range(nk)])
        frac = np.where(np.max(x_val * (1.0 - x_val), axis=0) > 1e-3)[0]
        if len(frac) == 0 or grid_iter == max_grid_iterations - 1:
            break
        new_blocks = []
        for kb, (a, c) in enumerate(blocks):          # bisect the blocks with fractional weights
            if kb in frac and c - a > 1:
                mid = (a + c) // 2
                new_blocks.extend([(a, mid), (mid, c)])
            else:
                new_blocks.append((a, c))
        blocks = new_blocks
    # penalty homotopy: sequential linearization of the concave penalty sum x(1-x)
    beta, beta_inc = m.ObjVal * 1e-3, 1.5
    for _ in range(30):
        x_val = np.array([[V["x_c"][k, kb].X for kb in range(V["nb"])] for k in range(nk)])
        if np.max(x_val * (1.0 - x_val)) < 1e-4:
            break
        x_lin = x_val.copy()
        for kb in range(V["nb"]):                       # break exact ties towards the cheapest topology
            if np.max(x_lin[:, kb]) - np.min(x_lin[:, kb]) < 1e-3:
                x_lin[np.argmin(C_topo), kb] += 0.05
        lin_pen = gp.quicksum((1.0 - 2.0 * x_lin[k, kb]) * V["x_c"][k, kb] * (blocks[kb][1] - blocks[kb][0])
                              for k in range(nk) for kb in range(V["nb"]))
        m.setObjective(V["base_obj"] + V["slack_pen"] + beta * lin_pen, GRB.MINIMIZE)
        m.optimize()
        if m.status != GRB.OPTIMAL:
            break
        beta *= beta_inc
    x_val = np.array([[V["x_c"][k, kb].X for kb in range(V["nb"])] for k in range(nk)])
    return _fix_and_resolve(m, V, np.argmax(x_val, axis=0), t_start, "MS MINTOC")
