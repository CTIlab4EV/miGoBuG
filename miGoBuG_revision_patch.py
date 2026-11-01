"""
miGoBuG_revision_patch.py
=========================
Add-on to miGoBuG.py for the TSG revision.  It does NOT change the
mi-Alt-PMP solver itself; it adds what the response letter promises:

  * one benchmark model (convex MIQP) used for EVERY method      -> R1.1, R3.4
  * Gurobi status / incumbent / best bound / MIP gap             -> R2.3
  * MILP with piecewise-linear costs (L segments)                -> R1.3
  * evaluate_sequence(): fix a switching sequence, hard
    constraints, same objective -> certified gap vs. J_LB        -> R3.4
  * continuation loop that records failed steps, mesh size,
    timing, and a tight-tolerance "polish" solve                 -> R2.1, R3.6
  * PMP diagnostics: clipping, slack active-set, balance
    residual, fractionality, energy/ramp violation               -> Table VI
  * numerical check of condition (C)                             -> Prop. 2
  * helpers for topology scaling (g nodes) and horizon scaling   -> R2.2, R1.4

Usage sketch (see benchmark_case at the bottom):

    import miGoBuG_revision_patch as P
    case = P.Case(p_net=..., dt=0.5, E_init=..., ...)
    row  = P.benchmark_case(case, eps_array=np.logspace(5, -10, 1000),
                            time_limit=3600, pwl_Ls=(8, 16, 32))

Gurobi credentials are read from environment variables
GRB_WLSACCESSID / GRB_WLSSECRET / GRB_LICENSEID (or the default
license file if they are not set).  Do not hard-code them.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field, replace

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
# The ONE benchmark model (ZOH transcription of (P1))
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
    holds for Configurations II-V and the topology study.  The continuous
    relaxation is the same (the simplex over topologies projects onto the
    cube [0,1]^g), so the root bound is unchanged; branching on switches
    gives balanced branches instead of weak "topology m off" branches.
    "topology": run3 formulation.  "auto": switch if exact, else topology.
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
                    binaries="topology", miqcp_method=None, mip_start_seq=None,
                    extra_params=None, cg_ub=False):
    """MIQP (pwl_L=None) or MILP-PWL.  Returns status, J_inc, J_LB, gap,
    J_true (re-evaluated with the true quadratic objective), time.

    binaries: "topology" (run3) | "switch" | "auto" (see build_benchmark).
    miqcp_method: Gurobi MIQCPMethod (None = default -1 automatic;
        0 = QCP relaxation at every node, 1 = linearized outer approximation).
    mip_start_seq: topology sequence used as MIP start (e.g. miGoBuG's).
        For CERTIFICATION runs only (a tighter J_LB); the timed comparison
        must run without it, otherwise Gurobi's time includes miGoBuG's work."""
    env = env or gurobi_env(output_flag)
    m, V = build_benchmark(case, env, pwl_L=pwl_L, perspective=perspective,
                           name="MIQP" if pwl_L is None else f"MILP_PWL{pwl_L}",
                           binaries=binaries, cg_ub=cg_ub)
    for kk, vv in (extra_params or {}).items():
        m.setParam(kk, vv)
    if miqcp_method is not None:
        m.Params.MIQCPMethod = miqcp_method
    if mip_start_seq is not None:
        seq0 = np.asarray(mip_start_seq, int)
        if "xb" in V:
            V["xb"].Start = case.X_g[seq0].T
        else:
            Y0 = np.zeros((case.k, case.N))
            Y0[seq0, np.arange(case.N)] = 1.0
            V["y"].Start = Y0
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
                           miqcp_method=miqcp_method, mip_start=mip_start_seq is not None,
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
# mi-Alt-PMP: continuation with bookkeeping + polish solve
# ======================================================================
def pmp_arrays(case: Case, fix_ramp_units=True):
    """Weights as the PMP kernels expect them.

    The kernels integrate in index time tau = t/dt:  de/dtau = -dt*p_s but
    dp_z/dtau = u_tau (no dt).  The PMP therefore weights the PER-STEP ramp
    u_tau = dt*u_h, while the benchmark weights the per-hour ramp u_h.
    With fix_ramp_units=True we pass B/dt^2 so both models penalize the same
    physical ramp.  (All other terms only differ by the common factor dt.)
    """
    B = case.B / case.dt ** 2 if fix_ramp_units else case.B
    return dict(S=case.S, Z=case.Z, B=B, Delta=case.Delta)


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


def run_continuation(case: Case, eps_array, fix_ramp_units=True, snapshots=(),
                     max_nodes=50000, verbose=False, adaptive=True, max_halvings=6,
                     step_control="persistent", mesh_reset=False, tol_fallback=None,
                     tol_schedule=None, bc_tol=None, mesh_density=1,
                     kappa=0.01, delta_state=0.05, h_max=math.log(10.0),
                     tau_stop=1e-6, eval_density=4, reject_factor=4.0, secant=False,
                     kl_measure="max", max_consecutive_fail=None, resume=None):
    """Same continuation as solve_ems_n_sources, plus bookkeeping and an
    adaptive step rule.  The rule is the same for every case.

    adaptive=False: original rule, a failed step is skipped and the next
    epsilon is tried from the last converged solution.  With
    max_consecutive_fail=n the attempt stops after n failed steps in a row
    (the continuation has stalled); the remaining targets count as failed and
    the last converged solution is returned.

    resume=(sol, eps): continue from an earlier converged solution `sol` at
    continuation value `eps` instead of from the initial guess; eps_array then
    holds only the remaining targets (all below eps).

    step_control="per_target" (run3): on failure the log-step towards the
    current scheduled epsilon is halved up to `max_halvings` times; the next
    scheduled epsilon starts again with the FULL step from the last converged
    solution.  Once stalled, every later target repeats max_halvings+1 failed
    solves from an ever larger distance: run3 spent >1000 s in such loops and
    never recovered.

    step_control="persistent" (default): a step size h (in log epsilon) is
    carried across targets.  Failure -> h/2; success -> h = min(2h, h_sched),
    where h_sched is the scheduled step.  If h drops below
    h_sched / 2**max_halvings, the continuation stops (stall) and the last
    converged solution is returned; the remaining targets count as failed.

    mesh_reset=True: before halving, retry the same epsilon once on the
    original uniform mesh (guess = last solution interpolated onto it).
    tol_fallback=x: before halving, retry once with collocation tolerance x
    (only if x is looser than the scheduled tolerance).  Both are logged.

    tol_schedule: ((eps_pi threshold, tol), ...) used instead of the global
    TOL_SCHEDULE; the first entry whose threshold is below eps_pi applies.
    bc_tol: absolute tolerance on the boundary conditions (kWh for energies,
    kW for the initial diesel power); None = solve_bvp default (= tol).
    mesh_density: initial uniform mesh has mesh_density*(N-1)+1 nodes
    (1 = one node per ZOH sample, as in run3).

    max_nodes: cap on collocation nodes per solve.  In run3 no converged solve
    used more than 164 nodes, while failed solves ran to 50000 nodes, which is
    what made failures expensive."""
    if step_control == "fisher":
        return _run_fisher(case, eps_array, fix_ramp_units, snapshots, max_nodes, verbose,
                           max_halvings, mesh_reset, tol_fallback, tol_schedule, bc_tol,
                           mesh_density, kappa, delta_state, h_max, tau_stop, eval_density,
                           reject_factor, secant, kl_measure)
    _check_hardcoded(case)
    W = pmp_arrays(case, fix_ramp_units)
    s, z = case.s, case.z
    sched = tuple(tuple(x) for x in (tol_schedule or TOL_SCHEDULE))
    n0 = int(mesh_density) * (case.N - 1) + 1
    t_uniform = np.linspace(0.0, case.N - 1.0, n0)
    t_nodes = t_uniform
    y_guess = np.zeros((2 * s + 2 * z, n0))
    y_guess[:s] = case.E_init[:, None]
    y_guess[s:s + z] = case.P_dg_init[:, None]
    log, snaps, sol_last, eps_last = [], {}, None, None
    if resume is not None:
        sol_last, eps_last = resume
        t_nodes, y_guess = sol_last.x, sol_last.y
    n_solves = n_refined = n_mesh_reset = n_tol_fallback = 0
    eps_array = np.asarray(eps_array, float)
    h_sched = abs(math.log(eps_array[1] / eps_array[0])) if len(eps_array) > 1 else 1.0
    h_min = h_sched / 2 ** max_halvings

    def attempt(eps, idx, refined, x0, y0, tol_override=None, kind="normal"):
        nonlocal n_solves
        n_solves += 1
        eps_sm = eps * 1e4
        tol = next(tl for thr, tl in sched if eps_sm > thr)
        if tol_override is not None:
            tol = tol_override
        fun, jac, bc = _bvp_funs(case, W, eps)
        ts = time.time()
        sol = solve_bvp(fun, bc, x0, y0, fun_jac=jac, tol=tol, max_nodes=max_nodes,
                        bc_tol=(bc_tol if bc_tol is not None else tol))
        log.append(dict(step=idx, eps=eps, eps_pi=eps_sm, tol=tol, kind=kind,
                        success=bool(sol.success), refined=refined,
                        nodes=len(sol.x), niter=getattr(sol, "niter", np.nan),
                        max_rms=(float(np.max(sol.rms_residuals))
                                 if sol.rms_residuals is not None else np.nan),
                        message=str(sol.message), time=time.time() - ts))
        return sol

    t0 = time.time()
    failed_targets = 0
    stalled_at = None
    h = h_sched
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
        elif step_control == "per_target":
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
                    stalled_at = eps_last                  # resumed attempt gives up quickly
                    failed_targets += len(eps_array) - idx - 1
                    break
                continue
            consecutive_fail = 0
        else:                                               # persistent
            if stalled_at is not None:
                failed_targets += 1
                continue
            while eps_last > target * (1 + 1e-12):
                e_try = max(target, eps_last * math.exp(-h))
                refined = h < h_sched * (1 - 1e-9)
                sol = attempt(e_try, idx, refined, t_nodes, y_guess)
                n_refined += refined
                if not sol.success and mesh_reset:
                    n_mesh_reset += 1
                    sol = attempt(e_try, idx, refined, t_uniform, sol_last.sol(t_uniform),
                                  kind="mesh_reset")
                if not sol.success and tol_fallback is not None:
                    tol_now = next(tl for thr, tl in sched if e_try * 1e4 > thr)
                    if tol_fallback > tol_now:
                        n_tol_fallback += 1
                        sol = attempt(e_try, idx, refined, t_nodes, y_guess,
                                      tol_override=tol_fallback, kind="tol_fallback")
                if sol.success:
                    t_nodes, y_guess, sol_last, eps_last = sol.x, sol.y, sol, e_try
                    h = min(h_sched, 2.0 * h)
                else:
                    h /= 2.0
                    if h < h_min * (1 - 1e-9):
                        stalled_at = eps_last
                        break
            if eps_last > target * (1 + 1e-12):
                failed_targets += 1
                continue
        if idx in snapshots:
            snaps[idx] = (eps_last, sol_last)
        if verbose and idx % 50 == 0:
            print(log[-1])
    t_total = time.time() - t0
    return dict(sol=sol_last, eps_last=eps_last, log=log, snapshots=snaps,
                time=t_total, failed_steps=failed_targets, n_steps=len(eps_array),
                n_solves=n_solves, n_refined=n_refined, n_mesh_reset=n_mesh_reset,
                n_tol_fallback=n_tol_fallback, tol_schedule=sched, bc_tol=bc_tol,
                mesh_density=mesh_density, max_nodes=max_nodes,
                stalled_at_eps_pi=(None if stalled_at is None else stalled_at * 1e4),
                weights=W)


def _run_fisher(case, eps_array, fix_ramp_units, snapshots, max_nodes, verbose,
                max_halvings, mesh_reset, tol_fallback, tol_schedule, bc_tol,
                mesh_density, kappa, delta_state, h_max, tau_stop, eval_density,
                reject_factor=4.0, secant=False, kl_measure="max"):
    """Equation-based (deterministic) epsilon schedule: constant thermodynamic speed.

    The topology weights are p_m(t) ~ exp(-beta*H_m(t)), beta = 1/eps_pi.  With
    the trajectory frozen, a step beta -> beta + dbeta changes them by
        KL(p_beta || p_beta+dbeta) = 1/2 * Var_p(H)(t) * dbeta^2 + O(dbeta^3).
    Requiring max_t KL <= kappa gives the predicted step
        dbeta = sqrt(2*kappa / max_t Var_t),   h = ln(1 + eps_pi*dbeta)   (in log eps).
    The trajectory also moves with eps (states/costates react), so after each
    accepted step the REALIZED max_t KL between the old and new weights is
    measured on a fixed grid, and the next step is
        h_next = min(h_pred, h_prev*sqrt(kappa/KL_real), 2*h_prev, h_max)
    (KL ~ h^2, the usual step-size controller).
    The penalty weight eps_chi = max(eps, 1e-2) and the costates also move the
    STATES.  Second controller (secant distance, Allgower & Georg): the realized
    change of the state trajectory, d = max_t max_i |x_new - x_old| / range_i
    (energies over E_max - E_min, diesel over its bound), must stay <= delta_state:
        h_next <= h_prev * delta_state / d        (d ~ h).
    A failed solve halves h (stop below h0/2**max_halvings), as before.
    Step REJECTION: a converged step whose realized KL exceeds
    reject_factor*kappa, or whose state change exceeds reject_factor*delta, is
    discarded and retried from the last accepted solution with
        h <- h * max(0.1, 0.9*min(sqrt(kappa/KL), delta/d))
    (as in adaptive ODE integrators).  Without it, one large accepted jump can
    move the continuation onto another solution branch (run "fast": a step
    with KL = 0.89 in IV/2, which then stalled).
    SECANT predictor (secant=True): the initial guess is the linear
    extrapolation in log(eps) of the last two accepted solutions, on the mesh
    of the last one; if that solve fails, the plain warm start is tried next.
    Stop at eps_array[-1], or earlier once every ZOH sample point has
    max_m p_m >= 1 - tau_stop (the binarization bound (R14) is met).
    Nothing is random: the same data give the same schedule.

    eps_array: only its first and last values are used (start, floor);
    `snapshots` indices are mapped to the first accepted eps <= eps_array[idx]."""
    _check_hardcoded(case)
    # "mean": horizon average of the per-point KL / variance (default).  The
    # states respond to the time integral of the weights; a switching instant
    # that moves by dtau changes the integrated weights by ~2*dtau whatever
    # eps_pi is, whereas the per-point KL next to it grows like (dtau/eps_pi)^2,
    # so a "max" target forces vanishing steps (or is violated) as eps_pi -> 0.
    agg = np.mean if kl_measure == "mean" else np.max
    W = pmp_arrays(case, fix_ramp_units)
    s, z = case.s, case.z
    eps_array = np.asarray(eps_array, float)
    eps0, eps_min = float(eps_array[0]), float(eps_array[-1])
    sched = tuple(tuple(x) for x in (tol_schedule or TOL_SCHEDULE))
    n0 = int(mesh_density) * (case.N - 1) + 1
    t_uniform = np.linspace(0.0, case.N - 1.0, n0)
    t_eval = np.linspace(0.0, case.N - 1.0, int(eval_density) * (case.N - 1) + 1)
    t_zoh = np.arange(case.N, dtype=float)
    y0 = np.zeros((2 * s + 2 * z, n0))
    y0[:s] = case.E_init[:, None]
    y0[s:s + z] = case.P_dg_init[:, None]
    log, accepted = [], []
    n_solves = n_fail = n_mesh_reset = n_tol_fallback = n_reject = n_secant_fail = 0
    t_rejected = 0.0
    sol_prev_acc, h_last_acc = None, None

    def attempt(eps, x0, yg, tol_override=None, kind="normal"):
        nonlocal n_solves
        n_solves += 1
        tol = tol_override if tol_override is not None else \
            next(tl for thr, tl in sched if eps * 1e4 > thr)
        fun, jac, bc = _bvp_funs(case, W, eps)
        ts = time.time()
        sol = solve_bvp(fun, bc, x0, yg, fun_jac=jac, tol=tol, max_nodes=max_nodes,
                        bc_tol=(bc_tol if bc_tol is not None else tol))
        log.append(dict(eps=eps, eps_pi=eps * 1e4, tol=tol, kind=kind, success=bool(sol.success),
                        nodes=len(sol.x), niter=getattr(sol, "niter", np.nan),
                        max_rms=(float(np.max(sol.rms_residuals))
                                 if sol.rms_residuals is not None else np.nan),
                        message=str(sol.message), time=time.time() - ts))
        return sol

    def solve_at(eps, x0, yg, sol_prev, y_sec=None):
        nonlocal n_mesh_reset, n_tol_fallback, n_secant_fail
        if y_sec is not None:
            sol = attempt(eps, x0, y_sec, kind="secant")
            if sol.success:
                return sol
            n_secant_fail += 1
        sol = attempt(eps, x0, yg)
        if not sol.success and mesh_reset and sol_prev is not None:
            n_mesh_reset += 1
            sol = attempt(eps, t_uniform, sol_prev.sol(t_uniform), kind="mesh_reset")
        if not sol.success and tol_fallback is not None:
            if tol_fallback > next(tl for thr, tl in sched if eps * 1e4 > thr):
                n_tol_fallback += 1
                sol = attempt(eps, x0, yg, tol_override=tol_fallback, kind="tol_fallback")
        return sol

    t0 = time.time()
    sol = solve_at(eps0, t_uniform, y0, None)
    if not sol.success:
        return dict(sol=None, eps_last=None, log=log, snapshots={}, time=time.time() - t0,
                    failed_steps=1, n_steps=0, n_solves=n_solves, n_refined=0,
                    n_mesh_reset=n_mesh_reset, n_tol_fallback=n_tol_fallback,
                    stalled_at_eps_pi=None, weights=W, schedule=[])
    sol_last, eps_last = sol, eps0
    p_last, var = topology_stats(case, W, sol_last, eps_last, t_eval)
    x_rng = np.concatenate([case.E_max - case.E_min, np.full(z, max(case.P_dg_ub - case.P_dg_lb, 1.0))])
    x_last = sol_last.sol(t_eval)[:s + z]
    accepted.append(dict(eps=eps_last, h=0.0, h_pred=np.nan, kl_real=0.0, kl_pointmax=0.0, d_state=0.0,
                         var_max_over_epspi2=float(var.max()) / (eps_last * 1e4) ** 2))
    h_prev = h_max
    h_floor = None
    stalled_at = None
    snap_targets = sorted({int(i) for i in snapshots if 0 <= int(i) < len(eps_array)})
    snaps, si = {}, 0

    def take_snapshots():
        nonlocal si
        while si < len(snap_targets) and eps_last <= eps_array[snap_targets[si]] * (1 + 1e-12):
            snaps[snap_targets[si]] = (eps_last, sol_last)
            si += 1

    take_snapshots()
    while eps_last > eps_min * (1 + 1e-12):
        eps_pi = eps_last * 1e4
        vmax = float(agg(var))
        h_pred = math.log1p(eps_pi * math.sqrt(2.0 * kappa / vmax)) if vmax > 0 else h_max
        h = min(h_pred, 2.0 * h_prev, h_max)
        if accepted[-1]["kl_real"] > 0:
            h = min(h, accepted[-1]["h"] * math.sqrt(kappa / accepted[-1]["kl_real"]))
        if accepted[-1].get("d_state", 0) > 0:
            h = min(h, accepted[-1]["h"] * delta_state / accepted[-1]["d_state"])
        if h_floor is None:
            h_floor = h / 2 ** max_halvings
        h = max(h, h_floor)
        while True:
            e_try = max(eps_min, eps_last * math.exp(-h))
            y_sec = None
            if secant and sol_prev_acc is not None and h_last_acc:
                r = math.log(eps_last / e_try) / h_last_acc
                y_sec = sol_last.y + r * (sol_last.y - sol_prev_acc.sol(sol_last.x))
            sol = solve_at(e_try, sol_last.x, sol_last.y, sol_last, y_sec)
            if not sol.success:
                n_fail += 1
                h /= 2.0
                if h < h_floor:
                    stalled_at = eps_last * 1e4
                    break
                continue
            p_new, var_new = topology_stats(case, W, sol, e_try, t_eval)
            with np.errstate(divide="ignore", invalid="ignore"):
                kl = np.sum(np.where(p_last > 0, p_last * (np.log(p_last)
                                                           - np.log(np.maximum(p_new, 1e-300))), 0.0), axis=1)
            kl_max = float(agg(kl))                      # the controlled measure
            kl_pointmax = float(np.max(kl))
            x_new = sol.sol(t_eval)[:s + z]
            d_state = float(np.max(np.abs(x_new - x_last) / x_rng[:, None]))
            too_big = kl_max > reject_factor * kappa or d_state > reject_factor * delta_state
            if too_big and h / 2.0 >= h_floor and e_try > eps_min * (1 + 1e-12):
                n_reject += 1
                t_rejected += log[-1]["time"]
                f = min(math.sqrt(kappa / kl_max) if kl_max > 0 else 1.0,
                        delta_state / d_state if d_state > 0 else 1.0)
                h *= max(0.1, 0.9 * f)
                continue
            break
        if stalled_at is not None:
            break
        var = var_new
        h_used = math.log(eps_last / e_try)
        x_last = x_new
        accepted.append(dict(eps=e_try, h=h_used, h_pred=h_pred, kl_real=kl_max,
                             kl_pointmax=kl_pointmax, d_state=d_state,
                             var_max_over_epspi2=float(var.max()) / (e_try * 1e4) ** 2))
        sol_prev_acc, h_last_acc = sol_last, h_used
        sol_last, eps_last, p_last, h_prev = sol, e_try, p_new, max(h_used, 1e-12)
        take_snapshots()
        if verbose and len(accepted) % 25 == 0:
            print(accepted[-1])
        # binarization reached at every ZOH sample point -> stop (R14)
        pz = boltzmann_rows(mode_H_vec(case, W, t_zoh, sol_last.sol(t_zoh)), eps_last * 1e4)
        if np.min(pz.max(axis=1)) >= 1.0 - tau_stop:
            break
    if si < len(snap_targets):                       # stopped early: last solution for the rest
        for idx in snap_targets[si:]:
            snaps[idx] = (eps_last, sol_last)
    return dict(sol=sol_last, eps_last=eps_last, log=log, snapshots=snaps,
                time=time.time() - t0, failed_steps=n_fail, n_steps=len(accepted),
                n_solves=n_solves, n_refined=n_fail, n_mesh_reset=n_mesh_reset,
                n_tol_fallback=n_tol_fallback, stalled_at_eps_pi=stalled_at, weights=W,
                schedule=accepted, n_rejected=n_reject, n_secant_fail=n_secant_fail,
                t_rejected=t_rejected, binarized=bool(stalled_at is None and eps_last > eps_min * (1 + 1e-12)))


def tighten_final(case: Case, cont: dict, tols=(0.3, 0.1), bc_tol=1e-2, max_nodes=50000):
    """Re-solve at the final epsilon with successively tighter tolerances,
    each warm-started from the previous one (continuation in the tolerance,
    not in epsilon).  Used when the continuation runs at a loose tolerance:
    polishing directly from tol 1.0 to 1e-3 failed in every case of the
    'fast' run.  Returns the tightest converged solution."""
    sol, eps = cont["sol"], cont["eps_last"]
    fun, jac, bc = _bvp_funs(case, cont["weights"], eps)
    t0, reached = time.time(), None
    for tl in tols:
        s2 = solve_bvp(fun, bc, sol.x, sol.y, fun_jac=jac, tol=tl, bc_tol=bc_tol, max_nodes=max_nodes)
        if not s2.success:
            break
        sol, reached = s2, tl
    return dict(sol=sol, tol_reached=reached, time=time.time() - t0)


def polish(case: Case, cont: dict, tol=1e-3, bc_tol=1e-3, max_nodes=50000):
    """Re-solve at the last epsilon with a tight tolerance (Response 2.1).
    The continuation itself uses tol = 5-15 and bc_tol = tol, which is far
    too loose to claim that (18)-(21) and (12) hold."""
    sol0, eps = cont["sol"], cont["eps_last"]
    fun, jac, bc = _bvp_funs(case, cont["weights"], eps)
    t0 = time.time()
    sol = solve_bvp(fun, bc, sol0.x, sol0.y, fun_jac=jac, tol=tol, bc_tol=bc_tol,
                    max_nodes=max_nodes)
    bc_res = np.abs(bc(sol.y[:, 0], sol.y[:, -1]))
    return dict(sol=sol, success=bool(sol.success), time=time.time() - t0,
                max_rms=float(np.max(sol.rms_residuals)), max_bc_res=float(bc_res.max()),
                message=sol.message)


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
    s, z, g, k = case.s, case.z, case.g, case.k
    S, Z, B, D = W["S"], W["Z"], W["B"], W["Delta"]
    e, pz = ycol[:s], ycol[s:s + z]
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


def mode_H_vec(case, W, t, Y):
    """Vectorized node_modes(...)["H"]: mode Hamiltonians H_m at many points.
    t: (n,) index times, Y: (2s+2z, n) states/costates.  Returns (n, k).
    Same formulas as node_modes (replica of the kernel), checked to 1e-10."""
    s, z, k = case.s, case.z, case.k
    S, D = W["S"], W["Delta"]
    pts = np.interp(t, np.arange(case.N), case.p_net)
    pz, lam_e = Y[s:s + z], Y[s + z:2 * s + z]
    dep = s - 1
    Pdg = case.eta_dg @ pz                                        # (n,)
    ks = np.tanh(10.0 * (pts - Pdg))                              # (n,)
    eta_k = case.eta_s[:, None] ** ks[None, :]                    # (s, n)
    tau1 = np.sum(lam_e[:dep] * eta_k[:dep] / S[:dep, None], axis=0)
    tau2 = np.sum(case.eta_s[:dep, None] ** (2 * ks[None, :]) / S[:dep, None], axis=0)
    tau_g = (case.X_g * case.eta_g) ** 2 @ (1.0 / D) if case.g else np.zeros(k)
    den = case.eta_s[dep] ** (2 * ks)[:, None] + S[dep] * (tau2[:, None] + tau_g[None, :])
    rhs = (pts - Pdg - tau1)[:, None]
    trial = (eta_k[dep][:, None] * rhs + lam_e[dep][:, None] * (tau2[:, None] + tau_g[None, :])) / den
    Ps = np.clip(trial, -500.0, 500.0)                            # (n, k)
    act = Ps != trial
    lam_mod = np.where(act, (Ps * den - eta_k[dep][:, None] * rhs) / (tau2[:, None] + tau_g[None, :] + 1e-12),
                       lam_e[dep][:, None])
    Wm = (S[dep] * Ps - lam_mod) / eta_k[dep][:, None]            # (n, k)
    ps = np.clip((lam_e[:dep, :, None] + eta_k[:dep, :, None] * Wm[None]) / S[:dep, None, None],
                 -500.0, 500.0)                                   # (dep, n, k)
    H = 0.5 * np.sum(S[:dep, None, None] * ps ** 2, axis=0) + 0.5 * S[dep] * Ps ** 2 \
        - np.sum(lam_e[:dep, :, None] * ps, axis=0) - lam_e[dep][:, None] * Ps + case.C_grd[None, :]
    if case.g:
        coef = (case.X_g * case.eta_g)[None, :, :] / D[None, None, :]   # (1, k, g)
        pg = np.clip(Wm[:, :, None] * coef, -2000.0, 2000.0)            # (n, k, g)
        H = H + 0.5 * np.sum(D[None, None, :] * pg ** 2, axis=2)
    return H


def boltzmann_rows(H, eps_pi):
    a = -(H - H.min(axis=1, keepdims=True)) / eps_pi
    w = np.exp(a)
    return w / w.sum(axis=1, keepdims=True)


def topology_stats(case, W, sol, eps, t_eval):
    """Per-point Boltzmann weights and Var_p(H) on a fixed grid (Fisher rule)."""
    H = mode_H_vec(case, W, t_eval, sol.sol(t_eval))
    p = boltzmann_rows(H, eps * 1e4)
    mean = np.sum(p * H, axis=1)
    var = np.maximum(np.sum(p * H ** 2, axis=1) - mean ** 2, 0.0)
    return p, var


def boltzmann(H, eps_pi):
    a = -(H - H.min()) / eps_pi
    w = np.exp(a)
    return w / w.sum()


def pmp_diagnostics(case: Case, sol, eps, W):
    """Numbers for Table VI: share of nodes with clipping / active slack,
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


def mode_H_exact(case, W, pt, ycol, ks):
    """EXACT minimized mode Hamiltonians H*_m (bounded controls, power balance
    enforced), +inf where mode m cannot meet the balance.  Used for the (C)
    check, because condition (C) concerns the true minimized Hamiltonian, not
    the clipped approximation of the kernel.

    For fixed direction ks, each mode solves the separable convex QP
        min  sum_j 1/2 S_j p_j^2 - lam_j p_j + sum_i 1/2 D_i pg_i^2
        s.t. sum_j a_j p_j + sum_i b_i pg_i = r,   box bounds,
    with a_j = eta_j^ks, b_i = X_mi eta_g,i, r = P_ld - eta_z' p_z.
    Stationarity gives p_j = clip((lam_j + nu a_j)/S_j), pg_i = clip(nu b_i/D_i);
    the balance is monotone in nu and solved by bisection."""
    s, z = case.s, case.z
    S, D = W["S"], W["Delta"]
    pz, lam = ycol[s:s + z], ycol[s + z:2 * s + z]
    r = pt - float(case.eta_dg @ pz)
    a = case.eta_s ** ks
    H = np.full(case.k, np.inf)
    for m in range(case.k):
        b = case.X_g[m] * case.eta_g
        cap = case.P_s_ub * a.sum() + case.P_g_ub * b.sum()
        if abs(r) > cap * (1 + 1e-12):
            continue
        def flows(nu):
            p = np.clip((lam + nu * a) / S, -case.P_s_ub, case.P_s_ub)
            pg = np.clip(nu * b / D, -case.P_g_ub, case.P_g_ub) if case.g else np.zeros(0)
            return p, pg, a @ p + b @ pg
        lo, hi = -1.0, 1.0
        while flows(lo)[2] > r:
            lo *= 2.0
        while flows(hi)[2] < r:
            hi *= 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if flows(mid)[2] < r:
                lo = mid
            else:
                hi = mid
        p, pg, _ = flows(0.5 * (lo + hi))
        H[m] = 0.5 * S @ p ** 2 - lam @ p + 0.5 * D @ pg ** 2 + case.C_grd[m]
    return H


def _H0_grid_vectorized(case, W, pt, lam, ks, grid, eps_pi, return_nu=False):
    """Exact minimized mode Hamiltonians on a whole grid of diesel powers at
    once (same math as mode_H_exact, vectorized over grid x modes).
    Returns H0 on the grid (+inf where no mode is feasible) and a mask of
    grid points where ALL modes are feasible."""
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
    nu = 0.5 * (lo + hi)                                   # multiplier of the balance, (G, k)
    p, pg, _ = flows(nu)
    Hm = (0.5 * (p ** 2) @ S - p @ lam
          + (0.5 * np.einsum("gki,i->gk", pg ** 2, D) if g else 0.0)
          + case.C_grd[None, :])
    Hm = np.where(feas, Hm, np.inf)
    anyf = feas.any(axis=1)
    H0 = np.full(G, np.inf)
    if anyf.any():
        Hf = Hm[anyf]
        hmin = Hf.min(axis=1)
        w = np.exp(-(Hf - hmin[:, None]) / eps_pi)
        H0[anyf] = 0.5 * W["Z"][0] * grid[anyf] ** 2 + hmin - eps_pi * np.log(w.sum(axis=1))
    if return_nu:
        return H0, feas.all(axis=1), nu, feas, Hm
    return H0, feas.all(axis=1)


def certify_condition_C(case: Case, sol, eps, W, n_grid=801):
    """Sufficient bound for condition (C), z = 1 (Appendix A).

    Comparison set K_u(t): diesel powers at which every topology meets the power
    balance with none of its storage or grid powers at a bound.  There each minimized
    mode Hamiltonian h_m is a convex quadratic in p_z, with h_m' = -eta_z * nu_m and
    nu_m (the multiplier of the balance) affine in p_z.  With
        H0'' = Z + phi''/eps_chi + E[h_m''] - Var(h_m')/eps_pi
    and Var(h_m') <= D(t)^2/4 (a variance never exceeds a quarter of the squared range),
    D(t) = eta_z * max_{p_z in K_u(t)} (max_m nu_m - min_m nu_m), condition (C) holds on
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
        _, allK, nu, feasK, _ = _H0_grid_vectorized(case, W, pts[i], lam, ks, grid, eps_pi, return_nu=True)
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

def check_condition_C(case: Case, sol, eps, W, n_grid=41, node_stride=1):
    """Numerical check of (C) for one aggregated diesel unit (z = 1):
    second differences of the EXACT minimized Hamiltonian H0 in p_z on a grid
    over [P_dg_lb, P_dg_ub], storage direction held at its trajectory value.
    The energy part (penalty) is convex and has no cross terms with p_z.

    Two regions are reported:
      * "common": diesel powers at which EVERY topology can meet the power
        balance.  This is the convex set K(t) used in Proposition 2.
      * "full": wherever at least one topology is feasible.  At the edge of a
        topology's feasible range its entropy weight drops to zero and H0
        jumps by about eps_pi*ln(k/(k-1)), which is never convex.
    Also reports the share of trajectory nodes whose diesel power is in K(t)."""
    assert case.z == 1, "grid check written for one aggregated diesel unit"
    eps_pi, inv_eps = eps * 1e4, 1.0 / max(eps, 1e-2)
    grid = np.linspace(case.P_dg_lb, case.P_dg_ub, n_grid)
    h = grid[1] - grid[0]
    Zw = W["Z"][0]
    s = case.s
    pts = np.interp(sol.x, np.arange(case.N), case.p_net)
    sig = lambda x: 0.5 * (1.0 + np.tanh(12.5 * x))
    pen = inv_eps * (sig(grid[1:-1] - case.P_dg_ub) + sig(case.P_dg_lb - grid[1:-1]))
    res = {r: dict(cmin=np.inf, neg=0, tot=0) for r in ("common", "full")}
    in_K = n_nodes = 0
    for i in range(0, len(sol.x), node_stride):
        y = sol.y[:, i]
        lam = y[s + 1:2 * s + 1]
        pz_traj = y[s]
        ks = math.tanh(10.0 * (pts[i] - float(case.eta_dg[0] * pz_traj)))
        n_nodes += 1
        _, allK = _H0_grid_vectorized(case, W, pts[i], lam, ks, np.array([pz_traj]), eps_pi)
        in_K += bool(allK[0])
        H0, allfin = _H0_grid_vectorized(case, W, pts[i], lam, ks, grid, eps_pi)
        fin3 = np.isfinite(H0[2:]) & np.isfinite(H0[1:-1]) & np.isfinite(H0[:-2])
        com3 = allfin[2:] & allfin[1:-1] & allfin[:-2]
        if not fin3.any():
            continue
        with np.errstate(invalid="ignore"):
            c_all = (H0[2:] - 2 * H0[1:-1] + H0[:-2]) / h ** 2 + pen
        tolc = 1e-9 * np.max(np.abs(H0[np.isfinite(H0)])) / h ** 2 + 1e-6 * Zw
        for name, mask in (("common", com3), ("full", fin3)):
            if mask.any():
                c = c_all[mask]
                R = res[name]
                R["cmin"] = min(R["cmin"], float(c.min()))
                R["neg"] += int(np.sum(c < -tolc))
                R["tot"] += int(c.size)
    out = dict(eps_pi=eps_pi, share_traj_in_K=in_K / max(n_nodes, 1))
    for name, R in res.items():
        out[f"{name}_min_curv_over_Z"] = R["cmin"] / Zw
        out[f"{name}_share_negative"] = R["neg"] / max(R["tot"], 1)
        out[f"{name}_points"] = R["tot"]
        out[f"{name}_holds"] = R["neg"] == 0
    out["holds"] = out["common_holds"]
    return out


# ======================================================================
# Scaling helpers
# ======================================================================
def grid_envelope(Delta, eta_g, breakeven, X_g=None, y_max=2.5, n=2501):
    """Dominance check for grid-connection nodes (Table R6).

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


def make_grid_nodes(g, Delta0, eta0=0.95, price0=1.0, rng=None,
                    eta_spread=0.02, price_spread=0.3, admissible=None):
    """Synthetic grid connections for the topology-scaling study (Table VII).
    Returns X_g (all 2^g combos unless `admissible` filters rows), eta_g,
    Delta, base_grid_costs.  Report the drawn values in the paper."""
    rng = np.random.default_rng(0) if rng is None else rng
    X = np.array(np.meshgrid(*[[0.0, 1.0]] * g, indexing="ij")).reshape(g, -1).T
    if admissible is not None:
        X = X[[admissible(row) for row in X]]
    eta = np.clip(eta0 + eta_spread * rng.uniform(-1, 1, g), 0.8, 0.99)
    Delta = Delta0 * (1 + 0.2 * rng.uniform(-1, 1, g))
    cost = price0 * (1 + price_spread * rng.uniform(-1, 1, g))
    return X, eta, Delta, cost


def truncate_horizon(case: Case, N: int) -> Case:
    """Horizon scaling at fixed resolution (terminal energy = initial)."""
    return replace(case, p_net=case.p_net[:N].copy())


def resample(case: Case, factor: int) -> Case:
    """Change resolution at fixed horizon: factor>1 refines (linear interp),
    factor<0 coarsens by averaging |factor| samples."""
    p = case.p_net
    if factor > 1:
        tf = np.arange(len(p) * factor) / factor
        p_new, dt = np.interp(tf, np.arange(len(p)), p), case.dt / factor
    else:
        f = -factor
        n = len(p) // f
        p_new, dt = p[:n * f].reshape(n, f).mean(1), case.dt * f
    return replace(case, p_net=p_new, dt=dt)


# ======================================================================
# One complete benchmark row (Tables R1, R2, VI)
# ======================================================================
def benchmark_case(case: Case, eps_array, time_limit=3600.0, pwl_Ls=(8, 16, 32),
                   threads=None, fix_ramp_units=True, check_C=True, env=None):
    env = env or gurobi_env(0)
    row = {}

    # --- miGoBuG -------------------------------------------------------
    cont = run_continuation(case, eps_array, fix_ramp_units=fix_ramp_units)
    if cont["sol"] is None:
        row["miGoBuG_status"] = "no converged step"
        return row
    pol = polish(case, cont)
    sol_use = pol["sol"] if pol["success"] else cont["sol"]
    seq, _ = extract_sequence(case, sol_use, cont["weights"])
    ev = evaluate_sequence(case, seq, env=env)
    row.update(miGoBuG_time_continuation=cont["time"], miGoBuG_time_polish=pol["time"],
               miGoBuG_time_fixedQP=ev["runtime"],
               miGoBuG_time_total=cont["time"] + pol["time"] + ev["runtime"],
               failed_steps=cont["failed_steps"], n_steps=cont["n_steps"],
               final_nodes=len(sol_use.x), polish_success=pol["success"],
               polish_max_rms=pol["max_rms"], polish_max_bc=pol["max_bc_res"],
               miGoBuG_feasible=ev["feasible"],
               J_miGoBuG_ZOH=ev.get("J_true", np.nan),
               newton_iters=[r["niter"] for r in cont["log"] if r["success"]])
    row.update(pmp_diagnostics(case, sol_use, cont["eps_last"], cont["weights"]))
    if check_C:
        row["condition_C_final"] = check_condition_C(case, sol_use, cont["eps_last"],
                                                     cont["weights"], node_stride=4)

    # --- MIQP ------------------------------------------------------------
    mi = solve_benchmark(case, time_limit=time_limit, threads=threads, env=env)
    row.update(MIQP_status=mi["status_str"], MIQP_time=mi["runtime"],
               J_inc=mi.get("J_inc", np.nan), J_LB=mi["J_LB"],
               MIQP_gap_pct=100 * mi["mip_gap"],
               MIQP_simult_chdis_kW=mi.get("simult_chdis_kW", np.nan),
               MIQP_settings=mi["settings"])
    if ev["feasible"]:
        row["gap_vs_inc_pct"] = certified_gap(ev["J_true"], mi["J_inc"])
        row["gap_certified_pct"] = certified_gap(ev["J_true"], mi["J_LB"])
        row["fixedQP_simult_chdis_kW"] = ev["simult_chdis_kW"]

    # --- MILP-PWL --------------------------------------------------------
    for L in pwl_Ls:
        r = solve_benchmark(case, time_limit=time_limit, threads=threads, pwl_L=L, env=env)
        row[f"PWL{L}_status"] = r["status_str"]
        row[f"PWL{L}_time"] = r["runtime"]
        row[f"PWL{L}_J_true"] = r.get("J_true", np.nan)
        if "J_true" in r:
            row[f"PWL{L}_gap_certified_pct"] = certified_gap(r["J_true"], mi["J_LB"])
    return row