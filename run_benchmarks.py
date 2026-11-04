"""Runs the benchmark of the paper and writes the raw data to one results folder.

    python run_benchmarks.py --step-pref config/per_case_steps.csv --out results/main
    python run_benchmarks.py --configs III --cases 3 --migobug-only --defer-fixedqp \
        --step-pref config/per_case_steps.csv --out results/quickstart

Every case is solved by miGoBuG, the MINLP, the MILP-PWL benchmarks (--pwl-configs) and the
heuristic baselines.  Every switching sequence is scored in the same benchmark model.
Finished rows are skipped when a run is restarted with the same --out.

Output: main.csv (one row per case), config.json, environment.json, grid_envelope.txt,
gurobi_logs/, progress/ (incumbent and bound over time), conv/ (continuation logs), traj/,
step_preference.csv, and threshold.csv with --threshold.

A Gurobi licence is read from GRB_WLSACCESSID / GRB_WLSSECRET / GRB_LICENSEID or gurobi.lic.
Every MINLP or MILP solve that does not finish costs up to one hour (--miqp-limit, --pwl-limit).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import baselines
import benchmark as P

# ======================================================================
# 1. PARAMETERS  (written to config.json in every output folder)
# ======================================================================
CFG = dict(
    data_file="data/load.csv",         # net-load profile of the test system [kW], 30-min intervals
    days_per_case=3,                   # 3 days x 48 samples = 144 steps (72 h)
    dt=0.5,                            # h
    scale_factor=1e5,

    # storage (two BESS)
    cap=[1500.0, 550.0], soc_min=0.3, soc_max=0.9, soc_init=0.6,
    eta_s=[0.975, 0.975],
    # diesel
    eta_dg=[0.95], P_dg_init_kw=200.0 / 0.95,
    # weights (before scaling)
    S=[0.25, 0.15], Z=[0.6], B=[0.05],
    # bounds: must equal the values hard-coded in the PMP kernels
    P_s_ub=500.0, P_g_ub=2000.0, P_dg_lb=0.0, P_dg_ub=2000.0, U_ub=4000.0,

    # Grid configurations.  Fixed connection cost of node i:  c_i = 0.5*Delta_i*(b_i*mean(P_net))^2,
    # so connecting node i pays off once the power it would carry exceeds about b_i*mean(P_net).
    # The nodes trade fixed cost against energy cost, so that no node dominates and the optimal
    # topology changes with the load level (checked at start-up, see grid_envelope.txt):
    #   G1 main feeder   Delta 0.65  eta 0.95  b 0.60
    #   G2 weak tie      Delta 0.95  eta 0.93  b 0.45   cheap to connect, expensive per kWh^2
    #   G3 strong tie    Delta 0.45  eta 0.97  b 0.80   expensive to connect, cheap per kWh^2
    #   G4 backbone tie  Delta 0.38  eta 0.98  b 0.95   most expensive to connect, cheapest per kWh^2
    configs={
        "I":   dict(g=0, Delta=[], eta_g=[], breakeven=[]),
        "II":  dict(g=1, Delta=[0.65], eta_g=[0.95], breakeven=[0.60]),
        "III": dict(g=2, Delta=[0.65, 0.95], eta_g=[0.95, 0.93], breakeven=[0.60, 0.45]),
        "IV":  dict(g=3, Delta=[0.65, 0.95, 0.45], eta_g=[0.95, 0.93, 0.97],
                    breakeven=[0.60, 0.45, 0.80]),
        "V":   dict(g=4, Delta=[0.65, 0.95, 0.45, 0.38], eta_g=[0.95, 0.93, 0.97, 0.98],
                    breakeven=[0.60, 0.45, 0.80, 0.95]),
    },

    # mi-Alt-PMP: geometric schedule np.logspace(start, stop, steps) of eps, with eps_pi = 1e4*eps.
    eps_schedule=dict(start=5.0, stop=-10.0),
    # A failed solve is skipped and the next eps starts from the last converged solution.
    # tol_schedule: solve_bvp tolerance, first entry with eps_pi > threshold.  An attempt stops
    # after max_consecutive_fail failed steps in a row; failed solves stop at max_nodes mesh nodes.
    continuation=dict(tol_schedule=[[5e3, 10.0], [1e-2, 5.0], [-1e300, 15.0]],
                      max_nodes=500, max_consecutive_fail=70),
    # Restart rule: the continuation starts with start_steps (start_steps_by_k per number of
    # topologies) or with the count from --step-pref.  If it ends above eps_pi_target, it is first
    # continued from its last converged solution with step halving (resume_*), then restarted with
    # step_increment more steps from a snapshot restart_backoff_decades before the stall, as long
    # as the step count stays <= max_steps.  The first attempt at or below the target is kept (else
    # the one ending at the smallest eps_pi); the time of all other attempts is in t_restart_attempts
    # and in t_total.  The decision uses only eps, never Gurobi.
    restart_rule=dict(eps_pi_target=2e-5, start_steps=1000, start_steps_by_k={1: 500},
                      step_increment=50, max_steps=1400, resume_max_halvings=6,
                      resume_max_failed_targets=3, restart_backoff_decades=2.0),
    threshold_every=50,                # continuation steps between snapshots of the Appendix A threshold

    # Gurobi
    miqp_time_limit=3600.0, pwl_time_limit=3600.0, mip_gap=1e-4, mip_focus=2,
    # Node files move Gurobi's node tree to disk beyond 2 GB; this changes only where nodes are
    # stored, not the search.
    miqp_extra_params=dict(NodefileStart=2.0, NodefileDir="/tmp/gurobi_nodes"),
    numeric_focus_retry=2,             # re-solve with NumericFocus=2 if status is not optimal/time_limit
    # MIQP formulation.  Tested on 240 small random instances against the best feasible solution
    # of all variants:
    #   topology binaries, NumericFocus 0: bound above the optimum in 6/240 (up to 0.25 %),
    #        i.e., Gurobi claimed "optimal" for a worse solution
    #   switch binaries + c_g upper bound:       1/240 (0.046 %), 40 % faster
    #   switch + c_g ub + NumericFocus 3:        0/240, time = topology binaries
    # The reported results use topology binaries with the perspective grid cost; every reported
    # bound is compared with the best feasible solution of all methods (bound_consistent in
    # main.csv).  miqp_binaries, miqp_cg_ub and miqp_numeric_focus select the variant.
    miqp_binaries="topology", miqp_cg_ub=False, miqp_numeric_focus=0,
    threads=None, pwl_Ls=[8, 16, 32], pwl_configs=["III"],
)
STEP_PREF = {}                         # --step-pref: tag -> row of a step-preference file


# ======================================================================
# 2. DATA
# ======================================================================
def load_days(cfg):
    """Net-load profile of the test system (data/load.csv), split into days."""
    df = pd.read_csv(cfg["data_file"], float_precision="round_trip")
    return [dict(day=int(d), p_net=g.sort_values("step")["p_net"].to_numpy(float))
            for d, g in df.groupby("day", sort=True)]


def net_load(days):
    return np.concatenate([d["p_net"] for d in days])


def case_load(cfg, days, i):
    """Net load of dataset i = 1..10."""
    kd = cfg["days_per_case"]
    return net_load(days[(i - 1) * kd: i * kd])


def make_case(cfg, p_net, config):
    sc = cfg["scale_factor"]
    cap = np.array(cfg["cap"])
    gcfg = cfg["configs"][config]
    g = gcfg["g"]
    X_g = (np.array(np.meshgrid(*[[0.0, 1.0]] * g, indexing="ij")).reshape(g, -1).T
           if g > 0 else np.zeros((1, 0)))
    Delta = np.array(gcfg["Delta"], float)
    costs = 0.5 * Delta * (np.array(gcfg["breakeven"]) * np.mean(p_net)) ** 2
    return P.Case(
        p_net=np.asarray(p_net, float), dt=cfg["dt"],
        E_init=cap * cfg["soc_init"], E_target=cap * cfg["soc_init"],
        E_min=cap * cfg["soc_min"], E_max=cap * cfg["soc_max"],
        P_dg_init=np.array([cfg["P_dg_init_kw"]]),
        S=np.array(cfg["S"]) / sc, Z=np.array(cfg["Z"]) / sc, B=np.array(cfg["B"]) / sc,
        Delta=Delta / sc,
        eta_s=np.array(cfg["eta_s"]), eta_dg=np.array(cfg["eta_dg"]),
        eta_g=np.array(gcfg["eta_g"], float), X_g=np.asarray(X_g, float),
        base_grid_costs=np.asarray(costs, float) / sc,
        P_s_ub=cfg["P_s_ub"], P_g_ub=cfg["P_g_ub"], P_dg_lb=cfg["P_dg_lb"],
        P_dg_ub=cfg["P_dg_ub"], U_ub=cfg["U_ub"], scale=sc)


# ======================================================================
# 3. LOGGING HELPERS
# ======================================================================
class Table:
    """Append-only CSV with resume support."""

    def __init__(self, path, key_cols):
        self.path, self.key_cols = Path(path), key_cols
        self.done = set()
        if self.path.exists():
            df = pd.read_csv(self.path)
            for _, r in df.iterrows():
                self.done.add(tuple(str(r[c]) for c in key_cols))
        self.cols = list(pd.read_csv(self.path, nrows=0).columns) if self.path.exists() else None

    def has(self, key):
        return tuple(str(k) for k in key) in self.done

    def add(self, row):
        row = {k: (json.dumps(v, default=_js) if isinstance(v, (dict, list, tuple)) else v)
               for k, v in row.items()}
        if self.cols is None:
            self.cols = list(row.keys())
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, self.cols).writeheader()
        new = [k for k in row if k not in self.cols]
        if new:                                    # widen the file if new columns appear
            df = pd.read_csv(self.path)
            for k in new:
                df[k] = np.nan
            self.cols += new
            df[self.cols].to_csv(self.path, index=False)
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, self.cols).writerow({k: row.get(k, "") for k in self.cols})
        self.done.add(tuple(str(row[c]) for c in self.key_cols))


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ======================================================================
# 4. miGoBuG
# ======================================================================
def _continuation(case, cfg, eps):
    cs = cfg["continuation"]
    # snapshots for restarts and for the threshold of Appendix A
    every = cfg["threshold_every"]
    if not cfg.get("threshold"):
        every = max(every, len(eps) // 20)
    snaps = tuple(range(0, len(eps), every)) + (len(eps) - 1,)
    return P.run_continuation(case, eps, snapshots=snaps, max_nodes=cs["max_nodes"],
                              tol_schedule=cs["tol_schedule"],
                              max_consecutive_fail=cs["max_consecutive_fail"])


def _continuation_with_restarts(case, cfg, out, tag):
    """Restart rule of CFG['restart_rule'].  Starts from the step count of this case in
    --step-pref, if any; the time of the attempts that led to it is then taken from that file."""
    es, cs, rule = cfg["eps_schedule"], cfg["continuation"], cfg["restart_rule"]
    pref = STEP_PREF.get(tag)
    start = rule["start_steps_by_k"].get(case.k, rule["start_steps"])
    L0 = int(pref["steps"]) if pref is not None else int(start)
    t_prior = float(pref["t_restart_attempts"]) if pref is not None else 0.0
    attempts, best, base = [], None, None

    def resume(c, eps_L, L):
        """Continue a stalled run from its last converged solution with step halving."""
        if not (c["sol"] is not None and c["eps_last"] * 1e4 > rule["eps_pi_target"]):
            return c
        rest = eps_L[eps_L < c["eps_last"] * (1 - 1e-12)]
        if not len(rest):
            return c
        c2 = P.run_continuation(case, rest, adaptive=True, max_halvings=rule["resume_max_halvings"],
                                max_nodes=cs["max_nodes"], tol_schedule=cs["tol_schedule"],
                                max_consecutive_fail=rule["resume_max_failed_targets"],
                                resume=(c["sol"], c["eps_last"]))
        say(f"   steps={L}: stalled at eps_pi={c['eps_last'] * 1e4:.2e}; resumed with halving -> "
            f"{c2['eps_last'] * 1e4:.2e}  {c2['time']:.1f}s")
        return dict(c2, time=c["time"] + c2["time"], log=c["log"] + c2["log"],
                    snapshots=c["snapshots"], failed_steps=c["failed_steps"] + c2["failed_steps"],
                    n_solves=c["n_solves"] + c2["n_solves"], n_refined=c2["n_refined"])

    n_max = max(0, (int(rule["max_steps"]) - L0) // int(rule["step_increment"]))
    for i in range(n_max + 1):
        L = L0 + i * int(rule["step_increment"])
        eps_L = np.logspace(es["start"], es["stop"], L)
        snap = None
        if i > 0 and base is not None and base["sol"] is not None:
            # partial restart: start the longer schedule from the first run's snapshot
            # restart_backoff_decades of eps before its stall point
            back = rule["restart_backoff_decades"]
            cand = [v for v in base["snapshots"].values() if v[0] >= base["eps_last"] * 10.0 ** back]
            snap = min(cand, key=lambda v: v[0]) if cand else None
        if snap is not None:
            rest = eps_L[eps_L < snap[0] * (1 - 1e-12)]
            c = P.run_continuation(case, rest, max_nodes=cs["max_nodes"], tol_schedule=cs["tol_schedule"],
                                   max_consecutive_fail=cs["max_consecutive_fail"], resume=(snap[1], snap[0]))
            c = dict(c, snapshots=base["snapshots"])
            say(f"   steps={L}: partial restart from eps_pi={snap[0] * 1e4:.2e} -> {c['eps_last'] * 1e4:.2e}  {c['time']:.1f}s")
            c = resume(c, eps_L, L)
            if c["eps_last"] * 1e4 > rule["eps_pi_target"]:      # full restart if that fails
                t_part = c["time"]
                cf = resume(_continuation(case, cfg, eps_L), eps_L, L)
                c = dict(cf, time=cf["time"] + t_part)
                say(f"   steps={L}: full restart -> {c['eps_last'] * 1e4:.2e}  (partial {t_part:.1f}s + full {cf['time']:.1f}s)")
        else:
            c = _continuation(case, cfg, eps_L)
            if i == 0:
                base = c
            c = resume(c, eps_L, L)
        eps_pi = c["eps_last"] * 1e4 if c["sol"] is not None else np.inf
        attempts.append(dict(steps=L, eps_pi_last=eps_pi, time=c["time"], failed_steps=c["failed_steps"]))
        if best is None or eps_pi < best[1]["eps_pi_last"]:
            best = (c, attempts[-1])
        say(f"   steps={L}: eps_pi_last={eps_pi:.2e}  {c['time']:.1f}s")
        if eps_pi <= rule["eps_pi_target"]:
            break
    cont, chosen = best
    t_other = sum(a["time"] for a in attempts if a is not chosen) + t_prior
    restart = dict(steps_used=chosen["steps"], n_attempts=len(attempts), t_restart_attempts=t_other,
                   restart_attempts=json.dumps(attempts, default=_js))
    log_path = out / "step_preference.csv"
    new = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, ["tag", "k", "steps", "eps_pi_last", "n_attempts", "t_restart_attempts", "attempts"])
        if new:
            w.writeheader()
        w.writerow(dict(tag=tag, k=case.k, steps=chosen["steps"], eps_pi_last=chosen["eps_pi_last"],
                        n_attempts=len(attempts), t_restart_attempts=t_other,
                        attempts=restart["restart_attempts"]))
    return cont, restart


def run_migobug(case, cfg, out, tag, env):
    """Continuation + switching sequence + fixed-sequence QP + diagnostics."""
    cont, restart = _continuation_with_restarts(case, cfg, out, tag)
    pd.DataFrame(cont["log"]).to_csv(out / "conv" / f"{tag}.csv", index=False)
    if cont["sol"] is None:
        return dict(miGoBuG_status="no converged step"), cont, None
    sol = cont["sol"]
    seq, traj = P.extract_sequence(case, sol, cont["weights"])
    if cfg.get("defer_fixedqp"):
        # no Gurobi licence needed; the sequence is saved in traj/*.npz, J and t_fixedQP stay empty
        ev = dict(runtime=np.nan, feasible=None, values=None)
    else:
        ev = P.evaluate_sequence(case, seq, env=env)
    ok = [r for r in cont["log"] if r["success"]]
    eps_min = 10.0 ** cfg["eps_schedule"]["stop"]
    row = dict(
        miGoBuG_status="ok",
        t_continuation=cont["time"], t_fixedQP=ev["runtime"],
        t_total=(cont["time"] + restart["t_restart_attempts"]
                 + (0.0 if cfg.get("defer_fixedqp") else ev["runtime"])),
        **restart,
        t_failed_solves=float(sum(r["time"] for r in cont["log"] if not r["success"])),
        steps=cont["n_steps"], failed_steps=cont["failed_steps"],
        stalled_at_eps_pi=cont["stalled_at_eps_pi"],
        reached_eps_min=bool(cont["eps_last"] <= eps_min * (1 + 1e-9)),
        final_tol=next((r["tol"] for r in reversed(cont["log"]) if r["success"]), np.nan),
        bvp_solves=cont["n_solves"], refined_solves=cont["n_refined"],
        eps_last=cont["eps_last"], eps_pi_last=cont["eps_last"] * 1e4,
        final_nodes=len(sol.x), max_nodes_seen=max(r["nodes"] for r in ok),
        newton_avg=float(np.nanmean([r["niter"] for r in ok])),
        newton_max=float(np.nanmax([r["niter"] for r in ok])),
        seq_feasible=ev["feasible"], J_miGoBuG_ZOH=ev.get("J_true", np.nan),
        fixedQP_simult_chdis_kW=ev.get("simult_chdis_kW", np.nan),
        n_switches=int(np.sum(np.diff(seq) != 0)),
    )
    row.update(P.pmp_diagnostics(case, sol, cont["eps_last"], cont["weights"]))
    row.update({f"final_{k}": v for k, v in
                P.residual_report(case, sol, cont["eps_last"], cont["weights"]).items()})
    np.savez_compressed(out / "traj" / f"{tag}_miGoBuG.npz",
                        t=sol.x, y=sol.y, seq=seq,
                        p_s=traj["p_s"], p_g=traj["p_g"], p_dg=traj["p_dg"], e=traj["e"],
                        **({f"fixedQP_{k}": v for k, v in ev["values"].items()}
                           if ev.get("values") else {}))
    return row, cont, seq


def threshold_profile(case, cont, tag):
    """Threshold of Appendix A at every snapshot of the continuation."""
    return [dict(tag=tag, step=idx, eps=cont["snapshots"][idx][0],
                 **P.threshold_bound(case, cont["snapshots"][idx][1], cont["snapshots"][idx][0], cont["weights"]))
            for idx in sorted(cont["snapshots"])]


# ======================================================================
# 5. GUROBI BENCHMARKS AND BASELINES
# ======================================================================
def run_miqp(case, cfg, out, tag, env, pwl_L=None, limit=None):
    """MINLP (pwl_L=None) or MILP-PWL, with its log, progress and trajectory."""
    log = out / "gurobi_logs" / f"{tag}_{'MIQP' if pwl_L is None else f'PWL{pwl_L}'}.log"
    form = dict(binaries=cfg["miqp_binaries"], cg_ub=cfg["miqp_cg_ub"])
    nf0 = cfg["miqp_numeric_focus"] if pwl_L is None else 0
    r = P.solve_benchmark(case, time_limit=limit or cfg["miqp_time_limit"],
                          mip_gap=cfg["mip_gap"], threads=cfg["threads"],
                          mip_focus=cfg["mip_focus"], pwl_L=pwl_L, env=env, log_file=log,
                          numeric_focus=nf0, extra_params=cfg["miqp_extra_params"], **form)
    nf = cfg["numeric_focus_retry"]
    # mem_limit (SoftMemLimit) is not numerical trouble: a re-solve would hit the same limit
    if nf and nf > nf0 and r["status_str"] not in P.VALID_MIQP_STATUS and r["status_str"] != "mem_limit":
        # e.g. status 13 (SUBOPTIMAL): the bound exceeded feasible solutions, i.e., it was not valid
        say(f"   status {r['status_str']} -> re-solving with NumericFocus={nf}")
        first = r["status_str"]
        r = P.solve_benchmark(case, time_limit=limit or cfg["miqp_time_limit"],
                              mip_gap=cfg["mip_gap"], threads=cfg["threads"],
                              mip_focus=cfg["mip_focus"], pwl_L=pwl_L, env=env,
                              log_file=log.with_name(log.stem + "_NF.log"), numeric_focus=nf,
                              extra_params=cfg["miqp_extra_params"], **form)
        r["first_status"] = first
    with open(out / "progress" / f"{log.stem}.json", "w") as f:
        json.dump(r.get("progress", []), f, default=_js)
    if pwl_L is None and r.get("values"):
        np.savez_compressed(out / "traj" / f"{tag}_MIQP.npz", **r["values"])
    return r


def run_baselines(case, env, methods=("MINTOC", "SUR", "Adjoint")):
    """Heuristic baselines (baselines.py), costed in the benchmark model."""
    args = (case.p_net, case.E_init, case.E_target, case.E_min, case.E_max, case.P_dg_init,
            case.S, case.Z, case.B, case.Delta, case.eta_s, case.eta_dg, case.eta_g,
            case.X_g, case.base_grid_costs, case.dt)
    bounds = dict(P_s_ub=case.P_s_ub, P_g_ub=case.P_g_ub, P_dg_lb=case.P_dg_lb,
                  P_dg_ub=case.P_dg_ub, U_ub=case.U_ub)
    out = {}
    for name, fn in [("MINTOC", baselines.solve_ems_ms_mintoc), ("SUR", baselines.solve_ems_sur_sos1),
                     ("Adjoint", baselines.solve_ems_hybrid_pmp)]:
        if name not in methods:
            continue
        try:
            t0 = time.time()
            r = fn(*args, bounds=bounds)
            t = time.time() - t0
            if not r.get("success"):
                out[name] = dict(status="failed", time=t)
                continue
            ev = P.evaluate_sequence(case, r["m_star"], env=env)
            out[name] = dict(status="ok" if ev["feasible"] else "sequence infeasible",
                             time=t, J=ev.get("J_true", np.nan),
                             own_obj_with_penalty=r["obj"] * case.scale)
        except Exception as e:                       # log, never stop the run
            out[name] = dict(status=f"error: {e!r}")
    return out


def gap(J, J_LB):
    return P.certified_gap(J, J_LB) if np.isfinite(J) and np.isfinite(J_LB) else np.nan


def _bound_check(row):
    """Gurobi's best bound must not exceed the cost of ANY feasible solution found by any method.
    A violation means the bound is not valid (numerical trouble).
    bound_consistent = True is a necessary condition only, not a proof."""
    lb = row.get("MIQP_J_LB")
    if lb is None or not np.isfinite(lb):
        return
    costs = [row.get(c) for c in ("MIQP_J_inc", "J_miGoBuG_ZOH", "MINTOC_J", "SUR_J", "Adjoint_J")]
    costs += [row.get(f"PWL{L}_J_seqQP") for L in (8, 16, 32)]
    costs += [row.get(f"PWL{L}_J_true") for L in (8, 16, 32)]
    costs = [float(c) for c in costs if c is not None and np.isfinite(c)]
    if costs:
        best = min(costs)
        row["best_feasible_any"] = best
        row["bound_consistent"] = bool(lb <= best * (1 + 1e-6))
        row["bound_minus_best_pct"] = 100 * (lb - best) / best


def pwl_columns(case, r, L, J_LB, env):
    """MILP-PWL row entries.  J_true: PWL dispatch re-costed with the quadratic objective.
    seqQP: PWL switching sequence + fixed-sequence QP, the same post-processing as miGoBuG ZOH."""
    c = {f"PWL{L}_status": r["status_str"], f"PWL{L}_time": r["runtime"],
         f"PWL{L}_J_true": r.get("J_true", np.nan),
         f"PWL{L}_gap_certified_pct": gap(r.get("J_true", np.nan), J_LB)}
    if r.get("seq") is not None:
        ev = P.evaluate_sequence(case, r["seq"], env=env)
        Jq = ev.get("J_true", np.nan) if ev["feasible"] else np.nan
        c.update({f"PWL{L}_J_seqQP": Jq, f"PWL{L}_time_seqQP": r["runtime"] + ev["runtime"],
                  f"PWL{L}_gap_seqQP_certified_pct": gap(Jq, J_LB)})
    return c


# ======================================================================
# 6. BENCHMARK
# ======================================================================
def run_case(T, TC, cfg, config, ci, p_net, out, env, with_baselines):
    tag = f"main_{config}_case{ci}"
    say(f"[main] {tag}")
    case = make_case(cfg, p_net, config)
    row = dict(config=config, case=ci, N=case.N, g=case.g, k=case.k,
               grid_costs=(case.base_grid_costs * case.scale).tolist())
    mg, cont, seq = run_migobug(case, cfg, out, tag, env)
    row.update(mg)
    if seq is not None and case.g > 0:
        Xs = case.X_g[np.asarray(seq, int)]
        row["usage_miGoBuG"] = Xs.mean(axis=0).round(4).tolist()      # share of time each node is on
        row["multi_node_share_miGoBuG"] = float(np.mean(Xs.sum(axis=1) >= 2))
    if cfg.get("threshold") and cont["snapshots"]:
        for r in threshold_profile(case, cont, tag):
            if not TC.has((r["tag"], r["step"])):
                TC.add(r)
    if cfg.get("migobug_only"):
        T.add(row)
        say(f"   done (miGoBuG only): J={row.get('J_miGoBuG_ZOH', np.nan):.6g}  "
            f"miGoBuG={row.get('t_total', np.nan):.1f}s  reached_eps_min={row.get('reached_eps_min')}")
        return
    mi = run_miqp(case, cfg, out, tag, env)
    row.update(MIQP_status=mi["status_str"], MIQP_time=mi["runtime"],
               MIQP_J_inc=mi.get("J_inc", np.nan), MIQP_J_LB=mi["J_LB"],
               MIQP_gap_pct=100 * mi["mip_gap"], MIQP_nodes=mi.get("node_count"),
               MIQP_n_bin=mi["n_bin"], MIQP_n_var=mi["n_var"],
               MIQP_simult_chdis_kW=mi.get("simult_chdis_kW", np.nan),
               MIQP_t_first_inc=(mi["progress"][0][0] if mi.get("progress") else np.nan),
               MIQP_settings=mi["settings"], MIQP_first_status=mi.get("first_status"))
    if mi.get("seq") is not None and case.g > 0:
        Xm = case.X_g[np.asarray(mi["seq"], int)]
        row["MIQP_usage"] = Xm.mean(axis=0).round(4).tolist()
        row["MIQP_multi_node_share"] = float(np.mean(Xm.sum(axis=1) >= 2))
    Jm = row.get("J_miGoBuG_ZOH", np.nan)
    row.update(gap_vs_inc_pct=gap(Jm, row["MIQP_J_inc"]),
               gap_certified_pct=gap(Jm, row["MIQP_J_LB"]))
    if config in cfg["pwl_configs"]:
        for L in cfg["pwl_Ls"]:
            say(f"   PWL L={L}")
            r = run_miqp(case, cfg, out, tag, env, pwl_L=L, limit=cfg["pwl_time_limit"])
            row.update(pwl_columns(case, r, L, mi["J_LB"], env))
    if with_baselines:
        for name, b in run_baselines(case, env).items():
            row.update({f"{name}_status": b.get("status"), f"{name}_time": b.get("time"),
                        f"{name}_J": b.get("J", np.nan),
                        f"{name}_own_obj": b.get("own_obj_with_penalty", np.nan),
                        f"{name}_gap_certified_pct": gap(b.get("J", np.nan), mi["J_LB"])})
    _bound_check(row)
    T.add(row)
    say(f"   done: gap_cert={row.get('gap_certified_pct'):.4g}%  bound_consistent={row.get('bound_consistent')}  "
        f"miGoBuG={row.get('t_total', np.nan):.1f}s  MIQP={mi['runtime']:.1f}s ({mi['status_str']})")


def document(cfg, days, out):
    import numba, scipy
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=1, default=_js)
    env = dict(python=sys.version, platform=platform.platform(),
               processor=platform.processor(), cpu_count=os.cpu_count(),
               numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__,
               numba_threads=numba.get_num_threads(),
               gurobi=".".join(map(str, P.gp.gurobi.version())),
               data_file=cfg["data_file"], days=len(days),
               started=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(out / "environment.json", "w") as f:
        json.dump(env, f, indent=1)


def check_grid_nodes(cfg, out):
    """Every grid node must be able to be part of the cheapest grid topology
    (P.grid_envelope).  Writes grid_envelope.txt; warns about dominated nodes."""
    lines = []
    for name, gc in cfg["configs"].items():
        if gc["g"] == 0:
            continue
        segs, share = P.grid_envelope(gc["Delta"], gc["eta_g"], gc["breakeven"])
        lines.append(f"Config {name}: " + " | ".join(
            f"{'+'.join(f'G{j}' for j in S)} [{a:.2f},{b:.2f}]" for S, a, b in segs))
        for j, sh in enumerate(share):
            if sh <= 0:
                lines.append(f"   [WARN] node G{j + 1} of Config {name} is dominated "
                             f"(never on the cost envelope)")
    (out / "grid_envelope.txt").write_text("\n".join(lines) + "\n")
    for l in lines:
        say(l)


def warm_up(cfg, p):
    """JIT-compile the numba kernels so compile time is not counted."""
    for config in ("I", "III"):
        P.run_continuation(make_case(cfg, p[:12], config), np.array([1e5, 1e3]))


def parse_range(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=["I", "II", "III", "IV", "V"])
    ap.add_argument("--cases", default="1-10", help="e.g. 1-10 or 1,3,5")
    ap.add_argument("--out", default=None, help="output folder (reuse it to resume)")
    ap.add_argument("--miqp-limit", type=float, help="time limit of the MINLP [s] (default 3600)")
    ap.add_argument("--pwl-limit", type=float, help="time limit of each MILP-PWL [s] (default 3600)")
    ap.add_argument("--pwl-configs", nargs="*", help="configurations with MILP-PWL (default III)")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--no-baselines", action="store_true", help="skip the heuristic baselines")
    ap.add_argument("--migobug-only", action="store_true", help="run only miGoBuG and its fixed-sequence QP")
    ap.add_argument("--defer-fixedqp", action="store_true",
                    help="with --migobug-only: skip the fixed-sequence QP (no Gurobi licence needed)")
    ap.add_argument("--threshold", action="store_true",
                    help="evaluate the regularization threshold of Appendix A along the continuation")
    ap.add_argument("--step-pref", default=None,
                    help="start each case from the step count in this file (config/per_case_steps.csv)")
    ap.add_argument("--start-steps", type=int, default=None, help="first step count for every case")
    ap.add_argument("--max-steps", type=int, default=None, help="largest step count tried")
    a = ap.parse_args()
    if a.defer_fixedqp and not a.migobug_only:
        ap.error("--defer-fixedqp requires --migobug-only")

    cfg = dict(CFG)
    if a.miqp_limit: cfg["miqp_time_limit"] = a.miqp_limit
    if a.pwl_limit: cfg["pwl_time_limit"] = a.pwl_limit
    if a.pwl_configs is not None: cfg["pwl_configs"] = a.pwl_configs
    if a.threads: cfg["threads"] = a.threads
    cfg.update(threshold=a.threshold, migobug_only=a.migobug_only, defer_fixedqp=a.defer_fixedqp)
    rule = dict(cfg["restart_rule"])
    if a.start_steps:
        rule.update(start_steps=a.start_steps, start_steps_by_k={})
    if a.max_steps: rule["max_steps"] = a.max_steps
    cfg["restart_rule"] = rule
    if a.step_pref:
        for r in csv.DictReader(open(a.step_pref)):
            STEP_PREF[r["tag"]] = r

    out = Path(a.out or f"results/run_{time.strftime('%Y%m%d_%H%M%S')}")
    for sub in ("", "conv", "traj", "gurobi_logs", "progress"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    Path(cfg["miqp_extra_params"]["NodefileDir"]).mkdir(parents=True, exist_ok=True)
    say(f"output -> {out}")

    days = load_days(cfg)
    n_cases = len(days) // cfg["days_per_case"]
    cases = [(i, case_load(cfg, days, i)) for i in parse_range(a.cases) if 1 <= i <= n_cases]
    check_grid_nodes(cfg, out)
    document(cfg, days, out)
    env = P.gurobi_env(0)
    say("warming up numba ...")
    warm_up(cfg, cases[0][1])

    t0 = time.time()
    T = Table(out / "main.csv", ["config", "case"])
    TC = Table(out / "threshold.csv", ["tag", "step"])
    for config in a.configs:
        for ci, p_net in cases:
            if T.has((config, ci)):
                continue
            try:
                run_case(T, TC, cfg, config, ci, p_net, out, env, not a.no_baselines)
            except Exception:
                traceback.print_exc()
                with open(out / "errors.log", "a") as f:
                    f.write(f"\n[{time.ctime()}] {config}/{ci}\n{traceback.format_exc()}")
    say(f"finished in {(time.time() - t0) / 3600:.2f} h -> {out}")


if __name__ == "__main__":
    main()
