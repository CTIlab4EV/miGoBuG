"""Benchmark model and mi-Alt-PMP continuation.

  * one benchmark model (ZOH transcription, convex MIQP) in which EVERY method is scored
  * MINLP and MILP-PWL solves with Gurobi: status, incumbent, best bound, MIP gap
  * evaluate_sequence(): fix a switching sequence, hard constraints, same objective
  * run_continuation(): the continuation of mi-Alt-PMP (miGoBuG.py), with logging
  * diagnostics of the solution and the regularization threshold of Appendix A

A Gurobi licence is read from GRB_WLSACCESSID / GRB_WLSSECRET / GRB_LICENSEID, or from gurobi.lic.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_bvp

import gurobipy as gp
from gurobipy import GRB

import miGoBuG as M

# Bounds that are HARD-CODED inside the numba kernels of miGoBuG.py.
# The benchmark must use the same values (checked in run_continuation).
PMP_HARDCODED = dict(P_s_ub=500.0, P_g_ub=2000.0, P_dg_lb=0.0,
                     P_dg_ub=2000.0, U_ub=4000.0)
TOL_SCHEDULE = ((5e3, 10.0), (1e-2, 5.0), (-np.inf, 15.0))  # as in miGoBuG.py


# ======================================================================
# Problem container
# ======================================================================
@dataclass
class Case:
    """All data of one benchmark instance (weights already scaled)."""
    p_net: np.ndarray            # (N,) net load at ZOH sample points [kW]
    dt: float                    # sample time [h]
    E_init: np.ndarray           # (s,)
    E_target: np.ndarray         # (s,)
    E_min: np.ndarray            # (s,)
    E_max: np.ndarray            # (s,)
    P_dg_init: np.ndarray        # (z,)
    S: np.ndarray                # (s,)  storage weights   (scaled)
    Z: np.ndarray                # (z,)  diesel weights    (scaled)
    B: np.ndarray                # (z,)  ramp weights, per-HOUR ramp (scaled)
    Delta: np.ndarray            # (g,)  grid weights      (scaled)
    eta_s: np.ndarray            # (s,)
    eta_dg: np.ndarray           # (z,)
    eta_g: np.ndarray            # (g,)
    X_g: np.ndarray              # (k, g) admissible topologies (0/1)
    base_grid_costs: np.ndarray  # (g,)  (scaled)
    P_s_ub: float = 500.0        # |storage-side power| bound
    P_g_ub: float = 2000.0
    P_dg_lb: float = 0.0
    P_dg_ub: float = 2000.0
    U_ub: float = 4000.0         # |ramp| bound [kW/h]
    scale: float = 1e5           # multiply objective by this for reporting

    @property
    def N(self): return len(self.p_net)
    @property
    def s(self): return len(self.S)
    @property
    def z(self): return len(self.Z)
    @property
    def g(self): return len(self.Delta)
    @property
    def k(self): return self.X_g.shape[0]
    @property
    def C_grd(self): return self.X_g @ self.base_grid_costs


# ======================================================================
# Gurobi environment (no credentials in source code)
# ======================================================================
def gurobi_env(output_flag: int = 0) -> gp.Env:
    keys = ("GRB_WLSACCESSID", "GRB_WLSSECRET", "GRB_LICENSEID")
    env = gp.Env(empty=True)
    if all(k in os.environ for k in keys):
        env.setParam("WLSACCESSID", os.environ["GRB_WLSACCESSID"])
        env.setParam("WLSSECRET", os.environ["GRB_WLSSECRET"])
        env.setParam("LICENSEID", int(os.environ["GRB_LICENSEID"]))
    env.setParam("OutputFlag", output_flag)
    env.start()
    return env


# ======================================================================
# The benchmark model (ZOH transcription)
# ======================================================================
def build_benchmark(case: Case, env: gp.Env, pwl_L: int | None = None,
                    fixed_seq: np.ndarray | None = None, name="bench",
                    perspective: bool = True, binaries: str = "auto", cg_ub: bool = False):
    """Convex MIQP (pwl_L=None) or MILP-PWL (pwl_L=L).

    Storage: bus-side charge/discharge split, storage-side power
    p_s = p_dis/eta - p_ch*eta bounded by P_s_ub (same bound as the PMP).
    Topology: one binary per admissible mode and interval (restricts the
    search to V, also when V is smaller than 2^g).  Grid powers: exact
    linear bounds  -P_g_ub*x_g <= p_g <= P_g_ub*x_g.

    perspective=True (MIQP only): the grid cost 1/2*D*p_g^2 is written in
    perspective form  c >= 1/2*D*p_g^2 / x_g  (rotated second-order cone
    p_g^2 <= (2/D)*c*x_g).  For binary x_g this is the SAME cost, so the
    optimum is unchanged, but the continuous relaxation is much tighter:
    a fractional switch can no longer buy grid power at a fraction of its
    quadratic cost.  This strengthens Gurobi's lower bound.

    binaries="switch": one binary per grid node and interval (g*N instead of
    k*N).  Exact when every on/off combination is admissible (X_g = {0,1}^g)
    and the topology cost is additive (C_grd = X_g @ base_grid_costs), which
    holds for Configurations II-V.  The continuous
    relaxation is the same (the simplex over topologies projects onto the
    cube [0,1]^g), so the root bound is unchanged; branching on switches
    gives balanced branches instead of weak "topology m off" branches.
    "topology": one binary per topology (used in the paper).  "auto": switch if exact, else topology.
    """
    N, s, z, g, k, dt = case.N, case.s, case.z, case.g, case.k, case.dt
    eta_s = case.eta_s
    m = gp.Model(name, env=env)

    ub_dis = np.repeat((case.P_s_ub * eta_s)[:, None], N, 1)
    ub_ch = np.repeat((case.P_s_ub / eta_s)[:, None], N, 1)
    p_dis = m.addMVar((s, N), lb=0.0, ub=ub_dis, name="p_dis")
    p_ch = m.addMVar((s, N), lb=0.0, ub=ub_ch, name="p_ch")
    e = m.addMVar((s, N + 1), lb=np.repeat(case.E_min[:, None], N + 1, 1),
                  ub=np.repeat(case.E_max[:, None], N + 1, 1), name="e")
    p_dg = m.addMVar((z, N + 1), lb=case.P_dg_lb, ub=case.P_dg_ub, name="p_dg")
    u = m.addMVar((z, N), lb=-case.U_ub, ub=case.U_ub, name="u")
    p_g = m.addMVar((g, N), lb=-case.P_g_ub, ub=case.P_g_ub, name="p_g") if g > 0 else None
    full = g > 0 and k == 2 ** g and np.allclose(
        np.sort(case.X_g @ (2.0 ** np.arange(g)[::-1])), np.arange(k))
    use_switch = binaries == "switch" or (binaries == "auto" and full)
    if use_switch and not full:
        raise ValueError("binaries='switch' needs all 2^g topologies admissible")
    if use_switch:
        xb = m.addMVar((g, N), vtype=GRB.BINARY, name="xb")
        code = (case.X_g @ (2 ** np.arange(g)[::-1])).astype(int)   # bit pattern -> code
        topo_of_code = np.empty(k, int)
        topo_of_code[code] = np.arange(k)                          # code -> topology index
        y = None
    else:
        y = m.addMVar((k, N), vtype=GRB.BINARY, name="y")

    Dinv, Deta = np.diag(1.0 / eta_s), np.diag(eta_s)
    p_s_store = Dinv @ p_dis - Deta @ p_ch                      # storage side
    m.addConstr(p_s_store <= case.P_s_ub)
    m.addConstr(p_s_store >= -case.P_s_ub)
    m.addConstr(e[:, 0] == case.E_init)
    m.addConstr(e[:, N] == case.E_target)
    m.addConstr(e[:, 1:] == e[:, :-1] - dt * p_s_store)
    m.addConstr(p_dg[:, 0] == case.P_dg_init)
    m.addConstr(p_dg[:, 1:] == p_dg[:, :-1] + dt * u)
    if not use_switch:
        m.addConstr(np.ones(k) @ y == 1.0)
    bal = case.eta_dg @ p_dg[:, :N] + np.ones(s) @ p_dis - np.ones(s) @ p_ch
    if g > 0:
        x_g = xb if use_switch else case.X_g.T @ y              # (g, N)
        m.addConstr(p_g <= case.P_g_ub * x_g)
        m.addConstr(p_g >= -case.P_g_ub * x_g)
        bal = bal + case.eta_g @ p_g
    m.addConstr(bal == case.p_net)

    if use_switch:
        topo_cost = dt * (case.base_grid_costs @ xb).sum()
    else:
        topo_cost = dt * (case.C_grd @ y).sum()
    persp = perspective and pwl_L is None and g > 0
    if persp:
        xg_var = m.addMVar((g, N), lb=0.0, ub=1.0, name="x_g")
        m.addConstr(xg_var == (xb if use_switch else case.X_g.T @ y))
        # cg_ub: c_g <= 1/2*Delta*P_g_ub^2 (valid: |p_g| <= P_g_ub, x <= 1); gives
        # Gurobi a bounded product when x_g is binary (switch formulation)
        cg = m.addMVar((g, N), lb=0.0, name="c_g",
                       ub=(np.repeat((0.5 * case.Delta * case.P_g_ub ** 2)[:, None], N, 1)
                           if cg_ub else GRB.INFINITY))
        for i in range(g):
            for t in range(N):
                m.addConstr(p_g[i, t] * p_g[i, t]
                            <= (2.0 / case.Delta[i]) * cg[i, t] * xg_var[i, t])
    if pwl_L is None:
        q = 0
        for j in range(s):
            q += 0.5 * case.S[j] * (p_dis[j, :] @ p_dis[j, :] + p_ch[j, :] @ p_ch[j, :])
        for j in range(z):
            q += 0.5 * case.Z[j] * (p_dg[j, :N] @ p_dg[j, :N])
            q += 0.5 * case.B[j] * (u[j, :] @ u[j, :])
        for i in range(g):
            if persp:
                q += cg[i, :].sum()
            else:
                q += 0.5 * case.Delta[i] * (p_g[i, :] @ p_g[i, :])
        m.setObjective(dt * q + topo_cost, GRB.MINIMIZE)
    else:
        m.setObjective(topo_cost, GRB.MINIMIZE)
        m.update()

        def pwl(var, w, lb, ub):
            xs = np.linspace(lb, ub, pwl_L + 1)
            m.setPWLObj(var, xs.tolist(), (0.5 * w * dt * xs ** 2).tolist())

        for j in range(s):
            for t in range(N):
                pwl(p_dis[j, t].item(), case.S[j], 0.0, ub_dis[j, t])
                pwl(p_ch[j, t].item(), case.S[j], 0.0, ub_ch[j, t])
        for j in range(z):
            for t in range(N):
                pwl(p_dg[j, t].item(), case.Z[j], case.P_dg_lb, case.P_dg_ub)
                pwl(u[j, t].item(), case.B[j], -case.U_ub, case.U_ub)
        for i in range(g):
            for t in range(N):
                pwl(p_g[i, t].item(), case.Delta[i], -case.P_g_ub, case.P_g_ub)

    if fixed_seq is not None:
        if use_switch:
            XB = case.X_g[np.asarray(fixed_seq, int)].T
            xb.lb, xb.ub = XB, XB
        else:
            Y = np.zeros((k, N))
            Y[np.asarray(fixed_seq, int), np.arange(N)] = 1.0
            y.lb, y.ub = Y, Y

    m.update()
    V = dict(p_dis=p_dis, p_ch=p_ch, e=e, p_dg=p_dg, u=u)
    if use_switch:
        V["xb"] = xb
        m._topo_of_code = topo_of_code
    else:
        V["y"] = y
    m._binaries = "switch" if use_switch else "topology"
    if g > 0:
        V["p_g"] = p_g
    return m, V


def true_objective(case: Case, v: dict) -> float:
    """Quadratic benchmark objective evaluated on solution values (scaled back)."""
    dt, N = case.dt, case.N
    J = 0.5 * np.sum(case.S[:, None] * (v["p_dis"] ** 2 + v["p_ch"] ** 2))
    J += 0.5 * np.sum(case.Z[:, None] * v["p_dg"][:, :N] ** 2)
    J += 0.5 * np.sum(case.B[:, None] * v["u"] ** 2)
    if case.g > 0:
        J += 0.5 * np.sum(case.Delta[:, None] * v["p_g"] ** 2)
    J += np.sum(case.C_grd[v["seq"]])
    return float(J * dt * case.scale)


def _collect(case, m, V, t_wall):
    out = dict(status=m.Status, runtime=t_wall, gurobi_runtime=m.Runtime,
               n_bin=int(m.NumBinVars), n_var=int(m.NumVars))
    if m.SolCount > 0:
        v = {kk: vv.X for kk, vv in V.items()}
        v.setdefault("p_g", np.zeros((0, case.N)))
        if "xb" in v:                       # switch formulation -> topology index
            bits = np.rint(v["xb"]).astype(int)
            g_ = bits.shape[0]
            v["seq"] = m._topo_of_code[(2 ** np.arange(g_)[::-1]) @ bits]
        else:
            v["seq"] = np.argmax(v["y"], axis=0)
        out.update(values=v, seq=v["seq"],
                   J_inc=m.ObjVal * case.scale,
                   J_true=true_objective(case, v),
                   simult_chdis_kW=float(np.max(np.minimum(v["p_ch"], v["p_dis"]))))
    if m.IsMIP:
        out["J_LB"] = m.ObjBound * case.scale
        out["node_count"] = m.NodeCount
        out["mip_gap"] = m.MIPGap if m.SolCount > 0 else math.inf
    return out


STATUS = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit",
          GRB.INFEASIBLE: "infeasible", GRB.INF_OR_UNBD: "inf_or_unbd",
          GRB.SUBOPTIMAL: "suboptimal",      # 13: numerical trouble, bound NOT reliable
          GRB.NUMERIC: "numeric", GRB.INTERRUPTED: "interrupted",
          GRB.MEM_LIMIT: "mem_limit"}         # SoftMemLimit reached: incumbent kept, bound not final
VALID_MIQP_STATUS = ("optimal", "time_limit")   # only these give a usable J_LB


def solve_benchmark(case: Case, time_limit=3600.0, mip_gap=1e-4, threads=None,
                    mip_focus=2, pwl_L=None, env=None, output_flag=0,
                    log_file=None, perspective=True, numeric_focus=0,
                    binaries="topology", extra_params=None, cg_ub=False):
    """MIQP (pwl_L=None) or MILP-PWL.  Returns status, J_inc, J_LB, gap,
    J_true (re-evaluated with the true quadratic objective), time.

    binaries: "topology" | "switch" | "auto" (see build_benchmark)."""
    env = env or gurobi_env(output_flag)
    m, V = build_benchmark(case, env, pwl_L=pwl_L, perspective=perspective,
                           name="MIQP" if pwl_L is None else f"MILP_PWL{pwl_L}",
                           binaries=binaries, cg_ub=cg_ub)
    for kk, vv in (extra_params or {}).items():
        m.setParam(kk, vv)
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    m.Params.MIPFocus = mip_focus
    if numeric_focus:
        m.Params.NumericFocus = numeric_focus
    if threads:
        m.Params.Threads = threads
    if log_file:
        m.Params.LogFile = str(log_file)
    progress = []                      # (time, incumbent, bound) when either changes

    def cb(model, where):
        if where == GRB.Callback.MIP:
            inc = model.cbGet(GRB.Callback.MIP_OBJBST)
            bnd = model.cbGet(GRB.Callback.MIP_OBJBND)
            if not progress or progress[-1][1] != inc or \
                    abs(progress[-1][2] - bnd) > 1e-6 * max(1.0, abs(bnd)):
                progress.append((model.cbGet(GRB.Callback.RUNTIME), inc, bnd))

    t0 = time.time()
    m.optimize(cb)
    out = _collect(case, m, V, time.time() - t0)
    out["progress"] = [(t, i * case.scale if i < GRB.INFINITY else None, b * case.scale)
                       for t, i, b in progress]
    out["status_str"] = STATUS.get(m.Status, str(m.Status))
    out["settings"] = dict(time_limit=time_limit, mip_gap=mip_gap,
                           threads=m.Params.Threads, mip_focus=mip_focus,
                           numeric_focus=numeric_focus, binaries=m._binaries, cg_ub=cg_ub,
                           extra_params=dict(extra_params or {}),
                           perspective=bool(perspective and pwl_L is None and case.g > 0),
                           version=".".join(map(str, gp.gurobi.version())))
    return out


def evaluate_sequence(case: Case, seq, env=None):
    """Fix the topology sequence and solve the remaining convex QP with
    HARD constraints in the benchmark model.  Infeasible -> reported as such
    (no elastic slacks)."""
    env = env or gurobi_env(0)
    m, V = build_benchmark(case, env, fixed_seq=np.asarray(seq, int), name="fixed")
    t0 = time.time()
    m.optimize()
    out = _collect(case, m, V, time.time() - t0)
    out["status_str"] = STATUS.get(m.Status, str(m.Status))
    out["feasible"] = m.Status == GRB.OPTIMAL
    return out


def certified_gap(J, J_LB):
    """(J - J_LB)/J in percent; valid upper bound on suboptimality."""
    return 100.0 * (J - J_LB) / J


# ======================================================================
# mi-Alt-PMP: continuation
# ======================================================================
def pmp_arrays(case: Case):
    """Weights as the PMP kernels expect them.

    The kernels integrate in index time tau = t/dt:  de/dtau = -dt*p_s but
    dp_z/dtau = u_tau (no dt).  The PMP therefore weights the PER-STEP ramp
    u_tau = dt*u_h, while the benchmark weights the per-hour ramp u_h.
    B/dt^2 is therefore passed, so both models penalize the same physical ramp.
    (All other terms only differ by the common factor dt.)
    """
    return dict(S=case.S, Z=case.Z, B=case.B / case.dt ** 2, Delta=case.Delta)


def _check_hardcoded(case):
    bad = {k: (getattr(case, k), v) for k, v in PMP_HARDCODED.items()
           if not np.isclose(getattr(case, k), v)}
    if bad:
        raise ValueError(f"Case bounds differ from values hard-coded in the PMP "
                         f"kernels (case, kernel): {bad}. Edit miGoBuG.py or the case.")


def _bvp_funs(case, W, eps):
    inv_eps = -1.0 / max(eps, 1e-2)
    eps_sm = eps * 1e4
    args = (case.p_net, case.E_min, case.E_max, W["S"], W["Z"], W["B"], W["Delta"],
            case.eta_s, case.eta_dg, case.eta_g, case.X_g, case.C_grd,
            inv_eps, eps_sm, case.dt)
    fun = lambda t, y: M.jitted_ode_n_sources(t, y, *args)
    jac = lambda t, y: M.jitted_jacobian_n_sources(t, y, *args)
    bc = lambda ya, yb: M.jitted_bc(ya, yb, case.E_init, case.E_target, case.P_dg_init)
    return fun, jac, bc


def run_continuation(case: Case, eps_array, snapshots=(), max_nodes=500, adaptive=False, max_halvings=6,
                     tol_schedule=TOL_SCHEDULE, max_consecutive_fail=None, resume=None):
    """Continuation of mi-Alt-PMP over eps_array, warm-starting every TPBVP from the last
    converged solution.

    adaptive=False: a failed step is skipped.  With max_consecutive_fail=n the attempt stops
    after n failed steps in a row (the continuation has stalled); the remaining targets count
    as failed and the last converged solution is returned.

    adaptive=True: on failure the log-step towards the current target is halved up to
    max_halvings times; the next target starts again with the full step.

    resume=(sol, eps): continue from an earlier converged solution at continuation value eps
    instead of from the initial guess; eps_array then holds only the remaining targets.

    tol_schedule: ((eps_pi threshold, tol), ...); the first entry whose threshold is below eps_pi
    applies.  max_nodes: cap on collocation nodes per solve; it makes failed solves cheap."""
    _check_hardcoded(case)
    W = pmp_arrays(case)
    s, z = case.s, case.z
    sched = tuple(tuple(x) for x in tol_schedule)
    t_nodes = np.linspace(0.0, case.N - 1.0, case.N)     # one node per ZOH sample
    y_guess = np.zeros((2 * s + 2 * z, case.N))
    y_guess[:s] = case.E_init[:, None]
    y_guess[s:s + z] = case.P_dg_init[:, None]
    log, snaps, sol_last, eps_last = [], {}, None, None
    if resume is not None:
        sol_last, eps_last = resume
        t_nodes, y_guess = sol_last.x, sol_last.y
    n_solves = n_refined = 0

    def attempt(eps, idx, refined, x0, y0):
        nonlocal n_solves
        n_solves += 1
        eps_sm = eps * 1e4
        tol = next(tl for thr, tl in sched if eps_sm > thr)
        fun, jac, bc = _bvp_funs(case, W, eps)
        ts = time.time()
        sol = solve_bvp(fun, bc, x0, y0, fun_jac=jac, tol=tol, max_nodes=max_nodes, bc_tol=tol)
        log.append(dict(step=idx, eps=eps, eps_pi=eps_sm, tol=tol,
                        success=bool(sol.success), refined=refined,
                        nodes=len(sol.x), niter=getattr(sol, "niter", np.nan),
                        max_rms=(float(np.max(sol.rms_residuals))
                                 if sol.rms_residuals is not None else np.nan),
                        message=str(sol.message), time=time.time() - ts))
        return sol

    t0 = time.time()
    eps_array = np.asarray(eps_array, float)
    failed_targets = 0
    stalled_at = None
    consecutive_fail = 0
    for idx, target in enumerate(eps_array):
        if sol_last is None or not adaptive:
            sol = attempt(target, idx, False, t_nodes, y_guess)
            if sol.success:
                t_nodes, y_guess, sol_last, eps_last = sol.x, sol.y, sol, target
                consecutive_fail = 0
            else:
                failed_targets += 1
                consecutive_fail += 1
                if (max_consecutive_fail and not adaptive and sol_last is not None
                        and consecutive_fail >= max_consecutive_fail):
                    stalled_at = eps_last
                    failed_targets += len(eps_array) - idx - 1
                    break
                continue
        else:
            log_step = math.log(target / eps_last)          # < 0
            halvings = 0
            while eps_last > target * (1 + 1e-12):
                e_try = target if halvings == 0 else max(target, eps_last * math.exp(log_step))
                refined = halvings > 0
                sol = attempt(e_try, idx, refined, t_nodes, y_guess)
                n_refined += refined
                if sol.success:
                    t_nodes, y_guess, sol_last, eps_last = sol.x, sol.y, sol, e_try
                else:
                    if halvings >= max_halvings:
                        break
                    log_step /= 2.0
                    halvings += 1
            if eps_last > target * (1 + 1e-12):
                failed_targets += 1
                consecutive_fail += 1
                if max_consecutive_fail and consecutive_fail >= max_consecutive_fail:
                    stalled_at = eps_last
                    failed_targets += len(eps_array) - idx - 1
                    break
                continue
            consecutive_fail = 0
        if idx in snapshots:
            snaps[idx] = (eps_last, sol_last)
    return dict(sol=sol_last, eps_last=eps_last, log=log, snapshots=snaps,
                time=time.time() - t0, failed_steps=failed_targets, n_steps=len(eps_array),
                n_solves=n_solves, n_refined=n_refined,
                stalled_at_eps_pi=(None if stalled_at is None else stalled_at * 1e4),
                weights=W)


def residual_report(case: Case, sol, eps, W):
    """Accuracy of a collocation solution, per mesh interval.
    solve_bvp's rms_residuals are RMS values of the RELATIVE residual
    r/(1+|f|) on each interval (all equations combined).  The maximum alone
    can be dominated by a few intervals at switching instants, so the
    distribution is reported, plus the boundary-condition residual in
    physical units (kWh for energies, kW for the initial diesel power)."""
    rms = np.asarray(sol.rms_residuals) if sol.rms_residuals is not None else np.array([np.nan])
    bc = _bvp_funs(case, W, eps)[2]
    bres = np.abs(bc(sol.y[:, 0], sol.y[:, -1]))
    s = case.s
    return dict(res_median=float(np.nanmedian(rms)), res_p95=float(np.nanpercentile(rms, 95)),
                res_max=float(np.nanmax(rms)),
                res_share_gt_1e_3=float(np.mean(rms > 1e-3)),
                res_share_gt_1e_1=float(np.mean(rms > 1e-1)),
                bc_res_energy_kWh=float(bres[:2 * s].max()),   # e(t0), e(tf)
                bc_res_all=float(bres.max()))


def extract_sequence(case: Case, sol, W):
    out = M.extract_pmp_trajectories(sol, np.arange(case.N, dtype=float), case.p_net,
                                     case.s, case.z, W["S"], W["Delta"], case.eta_s,
                                     case.eta_dg, case.eta_g, case.X_g, case.C_grd)
    return out["m_star"].astype(int), out


# ----------------------------------------------------------------------
# Per-node replica of jitted_ode_n_sources (numpy) for diagnostics
# ----------------------------------------------------------------------
def node_modes(case, W, pt, ycol, k_s_override=None):
    s, z, k = case.s, case.z, case.k
    S, Z, B, D = W["S"], W["Z"], W["B"], W["Delta"]
    pz = ycol[s:s + z]
    lam_e, lam_z = ycol[s + z:2 * s + z], ycol[2 * s + z:]
    dep = s - 1
    Pdg = float(case.eta_dg @ pz)
    ks = math.tanh(10.0 * (pt - Pdg)) if k_s_override is None else k_s_override
    eta_k = case.eta_s ** ks
    tau1 = np.sum(lam_e[:dep] * eta_k[:dep] / S[:dep])
    tau2 = np.sum(case.eta_s[:dep] ** (2 * ks) / S[:dep])
    tau_g = (case.X_g * case.eta_g) ** 2 @ (1.0 / D)
    H, slack_act, clipped, bal_res = np.zeros(k), np.zeros(k, bool), np.zeros(k, bool), np.zeros(k)
    u_opt = -lam_z / B
    for m in range(k):
        den = case.eta_s[dep] ** (2 * ks) + S[dep] * (tau2 + tau_g[m])
        trial = (eta_k[dep] * (pt - Pdg - tau1) + lam_e[dep] * (tau2 + tau_g[m])) / den
        Ps = float(np.clip(trial, -500.0, 500.0))
        slack_act[m] = Ps != trial
        lam_mod = lam_e[dep] if not slack_act[m] else \
            (Ps * den - eta_k[dep] * (pt - Pdg - tau1)) / (tau2 + tau_g[m] + 1e-12)
        Wm = (S[dep] * Ps - lam_mod) / eta_k[dep]
        pg_raw = Wm / D * case.X_g[m] * case.eta_g
        ps_raw = (lam_e[:dep] + eta_k[:dep] * Wm) / S[:dep]
        pg, ps = np.clip(pg_raw, -2000, 2000), np.clip(ps_raw, -500, 500)
        clipped[m] = np.any(pg != pg_raw) or np.any(ps != ps_raw)
        p_s_all = np.append(ps, Ps)
        bal_res[m] = pt - (eta_k @ p_s_all + Pdg + (case.X_g[m] * case.eta_g) @ pg)
        H[m] = (0.5 * S @ p_s_all ** 2 + 0.5 * D @ pg ** 2 + case.C_grd[m]
                - lam_e @ p_s_all)
    A0 = 0.5 * Z @ pz ** 2 + 0.5 * B @ np.clip(u_opt, -4000, 4000) ** 2 \
        + lam_z @ np.clip(u_opt, -4000, 4000)
    return dict(H=H, A0=A0, slack_active=slack_act, clipped=clipped,
                bal_res=bal_res, u_tau=u_opt, ks=ks)


def boltzmann(H, eps_pi):
    a = -(H - H.min()) / eps_pi
    w = np.exp(a)
    return w / w.sum()


def pmp_diagnostics(case: Case, sol, eps, W):
    """Diagnostics: share of nodes with clipping / active slack,
    power-balance residual of the selected mode, fractionality, violations."""
    s, z = case.s, case.z
    eps_pi = eps * 1e4
    t = sol.x
    pts = np.interp(t, np.arange(case.N), case.p_net)
    clip = slack = 0
    frac, bal, u_h = [], [], []
    for i in range(len(t)):
        r = node_modes(case, W, pts[i], sol.y[:, i])
        w = boltzmann(r["H"], eps_pi)
        mstar = int(np.argmax(w))
        clip += bool(r["clipped"][mstar])
        slack += bool(r["slack_active"][mstar])
        frac.append(1.0 - w.max())
        bal.append(abs(r["bal_res"][mstar]))
        u_h.append(np.max(np.abs(r["u_tau"])) / case.dt)     # per-hour ramp
    e = sol.y[:s]
    e_viol = max(0.0, float(np.max(e - case.E_max[:, None])),
                 float(np.max(case.E_min[:, None] - e)))
    pz = sol.y[s:s + z]
    pz_viol = max(0.0, float(np.max(pz - case.P_dg_ub)), float(np.max(case.P_dg_lb - pz)))
    return dict(nodes=len(t), share_free_clipped=clip / len(t),
                share_slack_active=slack / len(t), max_balance_residual_kW=max(bal),
                max_fractionality=max(frac), max_energy_violation_kWh=e_viol,
                max_diesel_violation_kW=pz_viol,
                max_ramp_kW_per_h=max(u_h), ramp_bound_kW_per_h=case.U_ub)


def _balance_multipliers(case, W, pt, lam, ks, grid):
    """Multiplier of the power balance of every topology on a grid of diesel powers, by bisection
    on the stationarity conditions with bounded controls.  Returns nu (G, k) and the mask (G, k)
    of the grid points at which a topology can meet the balance."""
    S, D = W["S"], W["Delta"]
    G, k, g = len(grid), case.k, case.g
    a = case.eta_s ** ks                                   # (s,)
    b = case.X_g * case.eta_g if g else np.zeros((k, 0))   # (k, g)
    Ps, Pg = case.P_s_ub, case.P_g_ub
    r = pt - case.eta_dg[0] * grid                         # (G,)
    cap = Ps * a.sum() + Pg * b.sum(axis=1)                # (k,)
    feas = np.abs(r)[:, None] <= cap[None, :] * (1 + 1e-12)  # (G, k)
    # bracket for nu: beyond these values every control is saturated
    with np.errstate(divide="ignore"):
        cand_hi = [((S * Ps - lam) / a).max()]
        cand_lo = [((-S * Ps - lam) / a).min()]
        if g:
            bp = np.where(b > 0, b, np.nan)
            cand_hi.append(np.nanmax(D * Pg / bp) if np.isfinite(np.nanmax(D * Pg / bp)) else 0.0)
            cand_lo.append(-cand_hi[-1])
    hi0, lo0 = max(cand_hi) + 1.0, min(cand_lo) - 1.0
    lo = np.full((G, k), lo0)
    hi = np.full((G, k), hi0)

    def flows(nu):                                         # nu: (G, k)
        p = np.clip((lam[None, None, :] + nu[..., None] * a) / S, -Ps, Ps)   # (G,k,s)
        pg = np.clip(nu[..., None] * b[None] / D, -Pg, Pg) if g else np.zeros((G, k, 0))
        return p, pg, p @ a + np.einsum("gki,ki->gk", pg, b)

    for _ in range(80):                                    # 2^-80 of the bracket
        mid = 0.5 * (lo + hi)
        low = flows(mid)[2] < r[:, None]
        lo = np.where(low, mid, lo)
        hi = np.where(low, hi, mid)
    return 0.5 * (lo + hi), feas


def threshold_bound(case: Case, sol, eps, W, n_grid=801):
    """Regularization threshold of Appendix A at one solution, z = 1.

    Comparison set K_u(t): diesel powers at which every topology meets the power
    balance with none of its storage or grid powers at a bound.  There each minimized
    mode Hamiltonian h_m is a convex quadratic in p_z, with h_m' = -eta_z * nu_m and
    nu_m (the multiplier of the balance) affine in p_z.  With
        H0'' = Z + phi''/eps_chi + E[h_m''] - Var(h_m')/eps_pi
    and Var(h_m') <= D(t)^2/4 (a variance never exceeds a quarter of the squared range),
    D(t) = eta_z * max_{p_z in K_u(t)} (max_m nu_m - min_m nu_m), H0 is convex on
    K_u(t) whenever eps_pi >= D(t)^2/(4 Z) for every node t.  Because each nu_m is affine
    on K_u(t), the maximum spread is attained at the end points of the interval; the grid
    locates them, and the distance to the true end point (at most one grid step) is
    added as a Lipschitz margin.  Also reports whether the trajectory itself is in K_u(t)."""
    assert case.z == 1, "written for one aggregated diesel unit"
    eps_pi = eps * 1e4
    grid = np.linspace(case.P_dg_lb, case.P_dg_ub, n_grid)
    h = grid[1] - grid[0]
    Zw = W["Z"][0]
    eta_z = float(case.eta_dg[0])
    S, Dl = W["S"], W["Delta"]
    s = case.s
    pts = np.interp(sol.x, np.arange(case.N), case.p_net)
    D_max, n_nodes, traj_in = 0.0, 0, 0
    for i in range(len(sol.x)):
        y = sol.y[:, i]
        lam = y[s + 1:2 * s + 1]
        pz = float(y[s])
        ks = math.tanh(10.0 * (pts[i] - eta_z * pz))
        nu, feasK = _balance_multipliers(case, W, pts[i], lam, ks, grid)
        a = case.eta_s ** ks
        b = case.X_g * case.eta_g if case.g else np.zeros((case.k, 0))
        p = (lam[None, None, :] + nu[..., None] * a) / S
        pg = nu[..., None] * b[None] / Dl if case.g else np.zeros(nu.shape + (0,))
        unsat = (np.abs(p) < case.P_s_ub).all(axis=2)
        if case.g:
            unsat &= ((np.abs(pg) < case.P_g_ub) | (b[None] == 0)).all(axis=2)
        n_nodes += 1
        j0 = int(np.argmin(np.abs(grid - pz)))
        A = feasK[j0]                                   # topologies feasible at the trajectory
        if not A.any():
            continue
        # K(t): interval of diesel powers around p_z*(t) on which the same topologies stay feasible
        same = (feasK == A[None, :]).all(axis=1)
        lo0 = hi0 = j0
        while lo0 > 0 and same[lo0 - 1]:
            lo0 -= 1
        while hi0 < n_grid - 1 and same[hi0 + 1]:
            hi0 += 1
        # J(t) within K(t): smallest interval containing the trajectory and the points at which no
        # feasible topology has a storage or grid power at a bound
        Ku = unsat[:, A].all(axis=1)
        Ku[:lo0] = False
        Ku[hi0 + 1:] = False
        idx = np.flatnonzero(Ku)
        lo, hi = (min(idx[0], j0), max(idx[-1], j0)) if idx.size else (j0, j0)
        traj_in += 1
        if A.sum() <= 1:
            continue
        J = np.zeros(n_grid, dtype=bool)
        J[lo:hi + 1] = True
        slopes = [S[j] / a[j] ** 2 for j in range(s)]
        if case.g:
            slopes += [Dl[j] / bj ** 2 for row in b for j, bj in enumerate(row) if bj > 0]
        lip = eta_z * max(slopes)                           # |d nu_m / d p_z| on K_u(t)
        nk = nu[J][:, A]
        spread = float(np.max(nk.max(axis=1) - nk.min(axis=1))) + 2.0 * lip * h
        D_max = max(D_max, eta_z * spread)
    eps_bar = D_max ** 2 / (4.0 * Zw)
    return dict(eps_pi=eps_pi, cert_eps_bar=eps_bar, cert_D_max=D_max,
                cert_share_traj_in_J=traj_in / max(n_nodes, 1), cert_holds=bool(eps_pi >= eps_bar))

# ======================================================================
# Grid-node check
# ======================================================================
def grid_envelope(Delta, eta_g, breakeven, X_g=None, y_max=2.5, n=2501):
    """Dominance check for grid-connection nodes.

    Delivering bus power y through node i costs c_i + 1/2*Delta_i*(y/eta_i)^2,
    with c_i = 1/2*Delta_i*(b_i*Pbar)^2.  For a topology S (set of connected
    nodes) the optimal split gives c_S + 1/2*a_S*y^2 with c_S = sum c_i and
    1/a_S = sum eta_i^2/Delta_i.  Normalizing by 1/2*Pbar^2, every topology is
    a parabola in y/Pbar that does not depend on the data.  A node is useful
    only if some topology containing it lies on the lower envelope.

    Returns (segments, share): segments = [(nodes, y_from, y_to)] of the
    envelope on [0, y_max]*Pbar, share[i] = length of y-range in which node
    i is connected on the envelope.  share[i] == 0 -> node i is dominated
    (it can never be part of the cheapest grid topology).  Storage and diesel
    are not included, so share > 0 is necessary, not sufficient, for use."""
    D, eta, b = (np.asarray(v, float) for v in (Delta, eta_g, breakeven))
    g = len(D)
    if X_g is None:
        X_g = np.array(np.meshgrid(*[[0.0, 1.0]] * g, indexing="ij")).reshape(g, -1).T
    rows = [r.astype(bool) for r in np.asarray(X_g) if r.sum() > 0]
    a, c = D / eta ** 2, D * b ** 2
    y = np.linspace(0.0, y_max, n)
    cost = np.array([c[r].sum() + y ** 2 / np.sum(1.0 / a[r]) for r in rows])
    arg = cost.argmin(axis=0)
    segs, i0 = [], 0
    for i in range(1, n + 1):
        if i == n or arg[i] != arg[i0]:
            segs.append((tuple(np.flatnonzero(rows[arg[i0]]) + 1), float(y[i0]), float(y[i - 1])))
            i0 = i
    share = [sum(t - f for S, f, t in segs if j + 1 in S) for j in range(g)]
    return segs, share


# ======================================================================
# One complete benchmark row (Tables R1, R2, VI)
# ======================================================================