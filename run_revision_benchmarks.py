"""
run_revision_benchmarks.py
==========================
Runs the experiments of the paper and logs the raw data.

    Experiment      Output (results/<run>/)
    -------------   ---------------------------------
    main            main.csv, condC.csv, conv/*.csv, traj/*.npz
    horizon         horizon.csv
    topology        topology.csv
    sensitivity     sensitivity.csv
    always          config.json, environment.json,
                    data_profiles.csv, data_stats.csv

Every Gurobi run writes its full log to gurobi_logs/ and its incumbent/bound
progress to progress/.  Rows are appended as soon as they finish, so the script can be
stopped and restarted: finished rows are skipped (--resume, default on).

Requirements: miGoBuG.py and miGoBuG_revision_patch.py in the same folder.
Gurobi license: set GRB_WLSACCESSID, GRB_WLSSECRET, GRB_LICENSEID
(or have a gurobi.lic).

Examples
    python run_revision_benchmarks.py                       # everything
    python run_revision_benchmarks.py --exp main --configs III --cases 1-3
    python run_revision_benchmarks.py --exp horizon topology --out results/run1

Configurations: I (isolated), II (1 grid node), III (2), IV (3 nodes, k=8),
V (4 nodes, k=16).  See CFG["configs"] for the node parameters.

Runtime warning: with the defaults (MIQP 3600 s, PWL MILPs 3600 s, PWL only
for Configuration III) every Gurobi run that does not finish costs one hour.
Use --miqp-limit / --pwl-limit / --pwl-configs to budget it.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

import miGoBuG as M
import miGoBuG_revision_patch as P
import baselines

# ======================================================================
# 1. PARAMETERS  (written to config.json in every output folder)
# ======================================================================
CFG = dict(
    data_file="data/load.csv",        # net-load profile of the test system [kW], 30-min intervals
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

    # grid configurations.
    #
    # Fixed connection cost of node i:  c_i = 0.5*Delta_i*(b_i*mean(P_net))^2
    # (break-even rule: connecting node i pays off once the
    # power it would carry exceeds about b_i*mean(P_net)).
    #
    # IV and V (larger-scale cases) use nodes with a deliberate
    # TRADE-OFF between fixed and energy cost, so that no node dominates and
    # the optimal topology changes with the load level:
    #   G1 main feeder   Delta 0.65  eta 0.95  b 0.60   (reference, as in II)
    #   G2 weak tie      Delta 0.95  eta 0.93  b 0.45   cheap to connect, expensive per kWh^2
    #   G3 strong tie    Delta 0.45  eta 0.97  b 0.80   expensive to connect, cheap per kWh^2
    #   G4 backbone tie  Delta 0.38  eta 0.98  b 0.95   most expensive to connect, cheapest per kWh^2
    # An earlier node set used G4 = (0.80, 0.96, 0.55): both its fixed cost and its cost per
    # kWh^2 exceed G1's, so it was strictly dominated (never used; Config V
    # was Config IV).  P.grid_envelope() checks this at start-up: every node
    # must lie on the lower cost envelope of the grid topologies.  With the
    # new G4 the envelope is G2 | G1 | G3 | G4 | G3+G4 | G1+G3+G4 over
    # y/Pbar in [0, 2.5].  Check `MIQP_usage` in main.csv afterwards.
    configs={
        "I":   dict(g=0, Delta=[], eta_g=[], breakeven=[]),
        "II":  dict(g=1, Delta=[0.65], eta_g=[0.95], breakeven=[0.60]),
        # III = G1 + weak tie G2 (nested design: II < III < IV < V).  The old
        # placeholder [0.65,0.65]/[0.95,0.95]/[0.60,0.72] made G2 dominated
        # (never used).
        "III": dict(g=2, Delta=[0.65, 0.95], eta_g=[0.95, 0.93], breakeven=[0.60, 0.45]),
        "IV":  dict(g=3, Delta=[0.65, 0.95, 0.45], eta_g=[0.95, 0.93, 0.97],
                    breakeven=[0.60, 0.45, 0.80]),
        "V":   dict(g=4, Delta=[0.65, 0.95, 0.45, 0.38], eta_g=[0.95, 0.93, 0.97, 0.98],
                    breakeven=[0.60, 0.45, 0.80, 0.95]),
    },

    # mi-Alt-PMP
    eps_schedule=dict(start=5.0, stop=-10.0, steps=1000),   # np.logspace(5,-10,1000)
    fix_ramp_units=True,               # pass B/dt^2 to the PMP (see patch)
    adaptive_continuation=True,        # halve the log-step after a failed step
    step_control="persistent",         # step size carried across targets (run3: "per_target")
    max_halvings=6,                    # stop when the step is below scheduled step / 2**max_halvings
    mesh_reset=False,                  # retry a failed step once on the uniform mesh
    tol_fallback=None,                 # retry a failed step once with this (looser) tolerance
    cont_max_nodes=5000,               # default; overridden per k by cont_by_k below
    #
    # Continuation settings per number of topologies k (the number of steps can
    # be set per case with --step-pref).  Config I: k=1, II: 2, III: 4, IV: 8, V: 16;
    # the topology study uses k=32.  Keys missing for a k are inherited from
    # the largest listed k below it.
    #   steps        continuation steps (np.logspace(start, stop, steps));
    #                run3 used 1000 everywhere.  Rule: 1000*max(1, log2 k).
    #   tol_schedule solve_bvp tol per eps_pi bracket [[threshold, tol], ...]:
    #                the first entry with eps_pi > threshold applies.
    #                run3: [[5e3, 10], [1e-2, 5], [-inf, 15]] -> residuals up to 14.
    #   bc_tol       absolute tolerance on the boundary conditions (kWh / kW);
    #                run3: = tol (i.e. 5-15 kWh).
    #   tol_fallback if a tight solve fails, retry once with this looser tol
    #                before halving the step (counted in n_tol_fallback; the
    #                tol of the final solution is reported as final_tol).
    #   max_nodes    mesh-node cap per solve (run3: 50000, but no converged solve
    #                used > 164; a tighter tol needs more nodes).
    #   mesh_density initial mesh = mesh_density*(N-1)+1 uniform nodes (run3: 1).
    #   step_control "fisher" (default): deterministic, equation-based schedule
    #                (patch: _run_fisher).  Each step changes the topology weights
    #                by at most kappa nats (KL) at every point, and the state
    #                trajectory by at most delta_state (fraction of its range).
    #                "steps" is then ignored; only start/stop of eps_schedule are used.
    #                "persistent": geometric schedule with `steps` steps.
    #   kappa, delta_state  the two targets (same for every k: no size tuning)
    # Profiles (--profile), applied on top of cont_by_k for EVERY k:
    #   accurate: tol 1.0 / 0.1 during the continuation, mesh 2N (smoke: stalls
    #             cured, residual max < 0.09, but 10-15x slower on easy cases)
    #   fast:     tol 1.0 during the continuation, mesh N; at the final eps the
    #             solution is re-solved at tol 0.3 and 0.1 (P.tighten_final) and
    #             then polished.  (Without the tightening the polish failed in
    #             all 16 rows of the first fast run.)
    # Both: step rejection (realized KL > 4 kappa or state change > 4 delta)
    #       and a secant predictor (reject_factor, secant).
    profile="run3",                    # the method reported in the paper
    # Profile per number of topologies k; overrides `profile` unless --profile
    # is given.  k = 1, 2, 4 are Configurations I, II, III: they use the loop of
    # the original miGoBuG.py (solve_ems_n_sources).
    profile_by_k={1: "original", 2: "original", 4: "original", 8: "original", 16: "original"},
    # Restart rule (profile "original" only; one rule for every case): the
    # continuation starts with start_steps (start_steps_by_k overrides it per
    # number of topologies, e.g. k=1 = Configuration I) and, if it ends above
    # eps_pi_target, is restarted from scratch with step_increment more steps,
    # as long as the step count stays <= max_steps.  The first attempt that ends
    # at or below eps_pi_target is kept (else the one ending at the smallest
    # eps_pi); the time of all other attempts is reported in t_restart_attempts
    # and included in t_total.  The decision uses only eps (never Gurobi).  The
    # step count used per case is logged in step_preference.csv; --step-pref
    # starts each case from that log.  --start-steps / --step-increment override.
    # resume_halving=True: before restarting from scratch, a stalled attempt is
    # first continued from its last converged solution over its remaining
    # targets with step halving (per-target, up to resume_max_halvings times).
    # Cases whose first attempt ends at or below eps_pi_target never use it.
    restart_rule=dict(eps_pi_target=2e-5, start_steps=1000, start_steps_by_k={1: 500},
                      step_increment=50, max_steps=1400, resume_halving=True, resume_max_halvings=6,
                      resume_max_failed_targets=3,
                      # restarts start from the first run's snapshot 2 decades of eps
                      # before its stall; a full restart only if that fails
                      restart_backoff_decades=2.0, restart_full_fallback=True),
    profiles=dict(
        # loop of the original miGoBuG.py: np.logspace(5, -10, 1000), a failed
        # step is skipped (no halving) and the next eps starts from the last
        # converged solution; tol 10 / 5 / 15, bc_tol = tol, max_nodes 50000;
        # the sequence is taken from the last converged solution (no polish)
        # final rule (revision): failed solves stop at 500 mesh nodes (no converged
        # solve needs more than 164) and an attempt stops after 70 failed steps in a row
        original=dict(adaptive=False, step_control="per_target", steps=1000,
                      tol_schedule=[[5e3, 10.0], [1e-2, 5.0], [-1e300, 15.0]], bc_tol=None,
                      tol_fallback=None, mesh_reset=False, max_nodes=500, mesh_density=1,
                      final_tighten=None, polish=False, max_consecutive_fail=70),
        # method reported in the paper (Algorithm 1):
        # geometric schedule, 1000 steps for every k, per-target halving up to
        # 6 times, solve_bvp tol 10 / 5 / 15, bc_tol = tol, no extra retries
        run3=dict(step_control="per_target", steps=1000, max_halvings=6,
                  tol_schedule=[[5e3, 10.0], [1e-2, 5.0], [-1e300, 15.0]], bc_tol=None,
                  tol_fallback=None, mesh_reset=False, max_nodes=50000, mesh_density=1,
                  final_tighten=None, polish_max_nodes=None),
        accurate={},
        # tol 1.0 throughout; polish attempted with a 5000-node cap (cheap failure)
        # max_nodes 10000 for every k: in run4_mig no converged solve used more
        # than 7331 nodes, while failed solves (up to 40000 nodes) took 959 of
        # the 1105 s total -> cap makes failures cheap, converged solves unchanged
        fast=dict(tol_schedule=[[-1e300, 1.0]], mesh_density=1, final_tighten=None,
                  polish_max_nodes=5000, max_nodes=10000),
        # same + re-solve at the final eps with tol 0.3 -> 0.1 before polishing
        fast_tight=dict(tol_schedule=[[-1e300, 1.0]], mesh_density=1, final_tighten=[0.3, 0.1]),
    ),
    cont_by_k={
        1:  dict(step_control="fisher", kappa=0.01, delta_state=0.05,
                 steps=1000, tol_schedule=[[5e3, 1.0], [-1e300, 0.1]], bc_tol=1e-2,
                 tol_fallback=5.0, max_nodes=5000, mesh_density=1,
                 max_halvings=12, mesh_reset=True),
        2:  dict(steps=1000, max_nodes=10000, mesh_density=2),
        4:  dict(steps=2000, max_nodes=20000),
        8:  dict(steps=3000, max_nodes=30000),
        16: dict(steps=4000, max_nodes=40000),
        32: dict(steps=5000, max_nodes=50000),
    },
    bvp_tol_schedule=None,             # None = original (10/5/15); e.g. [[5e3,1e-1],[1e-2,1e-2],[-1e300,1e-2]]
    polish_tol=1e-3,
    condC_every=50,                    # continuation steps between (C) snapshots
    condC_grid=41, condC_node_stride=2,

    # Gurobi
    miqp_time_limit=3600.0, pwl_time_limit=3600.0, mip_gap=1e-4, mip_focus=2,
    # Memory: the container has 14 GB and Gurobi's node tree outgrew it on V/1
    # (OOM kill after ~40 min).  Node files move the tree to disk beyond 2 GB;
    # this changes only where nodes are stored, not the search.
    miqp_extra_params=dict(NodefileStart=2.0, NodefileDir="/tmp/gurobi_nodes"),
    numeric_focus_retry=2,             # re-solve with NumericFocus=2 if status is not optimal/time_limit
    # MIQP formulation.  Tested on 240 small random instances
    # against the best feasible solution of all variants:
    #   topology binaries, NumericFocus 0: bound above the optimum in
    #        6/240 (up to 0.25 %) -> Gurobi claimed "optimal" for a worse solution
    #   switch binaries + c_g upper bound:       1/240 (0.046 %), 40 % faster
    #   switch + c_g ub + NumericFocus 3:        0/240, time = topology binaries
    # Binaries per interval: g instead of k = 2^g (exact: all combinations
    # admissible, additive connection costs; same continuous relaxation).
    # The reported results use topology binaries with the perspective grid cost;
    # every reported bound is compared with the best feasible solution of all
    # methods (bound_consistent in main.csv).  miqp_binaries selects the variant.
    miqp_binaries="topology", miqp_cg_ub=False, miqp_numeric_focus=0,
    threads=None, pwl_Ls=[8, 16, 32], pwl_configs=["III"],
    # --redo-pwl re-solves only these L on rows whose Gurobi results are reused
    # (needed for the sequence + QP rows of Table R1); L=8 own-dispatch values
    # are copied from the reused run.  New rows solve all pwl_Ls.
    pwl_redo_Ls=[16, 32],

    # scaling studies
    horizon_N=[48, 96, 144, 288, 576],      # at 30 min, consecutive days
    resolution_factors=[-2, 1, 2, 4],       # at 72 h: N = 72, 144, 288, 576
    horizon_datasets=[1, 2, 3],
    topology_g=[5],                         # g = 1..4 are Configs II..V (main.csv)
    topology_datasets=list(range(1, 11)),
    # nested nodes: G1..G4 of Config V + G5 "second strong tie".  G5 is the
    # cheapest single node for y/Pbar in [0.76, 0.95]; every node keeps its
    # own single-node range on the cost envelope (P.grid_envelope).
    topology_nodes=dict(Delta=[0.65, 0.95, 0.45, 0.38, 0.40],
                        eta_g=[0.95, 0.93, 0.97, 0.98, 0.98],
                        breakeven=[0.60, 0.45, 0.80, 0.95, 0.90]),
    # sensitivity (Configuration III)
    sens_L=[500, 1000, 2000, 4000], sens_eps0=[3.0, 7.0],  # log10 start values; 2000 = main for III
    sens_kappa=[0.0025, 0.01, 0.04], sens_delta=[0.02, 0.1],   # used when step_control="fisher"
    sens_datasets=list(range(1, 11)),
)


# ======================================================================
# 2. DATA  (taken from the old main block, files sorted for reproducibility)
# ======================================================================
def load_days(cfg):
    """Net-load profile of the test system (data/load.csv), split into days."""
    df = pd.read_csv(cfg["data_file"], float_precision="round_trip")
    days = [dict(day=int(d), p_net=g.sort_values("step")["p_net"].to_numpy(float))
            for d, g in df.groupby("day", sort=True)]
    lens = {len(d["p_net"]) for d in days}
    if len(lens) != 1:
        print(f"[warn] days have different lengths: {sorted(lens)}")
    return cfg["data_file"], days


def net_load(cfg, days):
    return np.concatenate([d["p_net"] for d in days])


def make_case(cfg, p_net, config, grid_override=None):
    sc = cfg["scale_factor"]
    cap = np.array(cfg["cap"])
    gcfg = cfg["configs"][config] if grid_override is None else grid_override
    g = gcfg["g"]
    if grid_override is not None and "X_g" in gcfg:
        X_g, eta_g, Delta, costs = gcfg["X_g"], gcfg["eta_g"], gcfg["Delta"], gcfg["costs"]
    else:
        X_g = (np.array(np.meshgrid(*[[0.0, 1.0]] * g, indexing="ij")).reshape(g, -1).T
               if g > 0 else np.zeros((1, 0)))
        Delta = np.array(gcfg["Delta"], float)
        eta_g = np.array(gcfg["eta_g"], float)
        costs = 0.5 * Delta * (np.array(gcfg["breakeven"]) * np.mean(p_net)) ** 2
    return P.Case(
        p_net=np.asarray(p_net, float), dt=cfg["dt"],
        E_init=cap * cfg["soc_init"], E_target=cap * cfg["soc_init"],
        E_min=cap * cfg["soc_min"], E_max=cap * cfg["soc_max"],
        P_dg_init=np.array([cfg["P_dg_init_kw"]]),
        S=np.array(cfg["S"]) / sc, Z=np.array(cfg["Z"]) / sc, B=np.array(cfg["B"]) / sc,
        Delta=np.asarray(Delta, float) / sc,
        eta_s=np.array(cfg["eta_s"]), eta_dg=np.array(cfg["eta_dg"]),
        eta_g=np.asarray(eta_g, float), X_g=np.asarray(X_g, float),
        base_grid_costs=np.asarray(costs, float) / sc,
        P_s_ub=cfg["P_s_ub"], P_g_ub=cfg["P_g_ub"], P_dg_lb=cfg["P_dg_lb"],
        P_dg_ub=cfg["P_dg_ub"], U_ub=cfg["U_ub"], scale=sc)


def eps_array(start, stop, steps):
    return np.logspace(start, stop, int(steps))


def cont_settings(cfg, k):
    """Continuation settings for a problem with k topologies (cfg['cont_by_k'],
    inherited upwards: k=8 gets the k=1/k=2 tolerances unless it overrides them)."""
    s = dict(steps=cfg["eps_schedule"]["steps"], kappa=0.01, delta_state=0.05,
             # secant OFF: A/B test on Configuration II, dataset 1 -- 8 failed secant guesses cost
             # 42 s (84.7 s vs 4.1 s); with all additions 17 failures, 101 s.
             # kl_measure "max": run B of the A/B test (fastest, 2.5 s on II/1)
             reject_factor=4.0, secant=False, final_tighten=None, kl_measure="max",
             tol_schedule=cfg.get("bvp_tol_schedule") or [list(x) for x in P.TOL_SCHEDULE],
             bc_tol=None, tol_fallback=cfg.get("tol_fallback"),
             max_nodes=cfg.get("cont_max_nodes", 50000), mesh_density=1,
             max_halvings=cfg["max_halvings"], mesh_reset=cfg.get("mesh_reset", False),
             step_control=cfg.get("step_control", "persistent"),
             adaptive=cfg["adaptive_continuation"], polish=True)
    for kk in sorted(cfg.get("cont_by_k", {})):
        if kk <= k:
            s.update(cfg["cont_by_k"][kk])
    if cfg.get("schedule_override"):
        s["step_control"] = cfg["schedule_override"]
    prof = cfg.get("profile_cli") or cfg.get("profile_by_k", {}).get(k) or cfg.get("profile", "accurate")
    s.update(cfg.get("profiles", {}).get(prof, {}))
    s["profile"] = prof
    if cfg.get("eps_steps_override"):          # --eps-steps wins over the profile's steps
        s["steps"] = cfg["eps_steps_override"]
    s.update(cfg.get("cli_overrides", {}))
    return s


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
# 4. BUILDING BLOCKS
# ======================================================================
def run_migobug(case, cfg, eps, out, tag, snapshots_every=None, env=None, cont_override=None):
    """Continuation + polish + sequence + fixed-topology QP + diagnostics.
    eps=None -> schedule with cont_settings(cfg, k)['steps'] steps (main,
    horizon, topology); an explicit eps array is used as given (sensitivity).
    All other continuation settings always come from cont_settings."""
    cs = cont_settings(cfg, case.k)
    cs.update(cont_override or {})
    rule = cfg.get("restart_rule")
    if eps is not None or not rule or cs["adaptive"]:
        if eps is None:
            es = cfg["eps_schedule"]
            eps = eps_array(es["start"], es["stop"], cs["steps"])
        cont = _continuation(case, cfg, cs, eps, snapshots_every)
        eps_min = float(eps[-1])
        restart = dict(steps_used=len(eps), n_attempts=1, t_restart_attempts=0.0,
                       restart_attempts="[]", step_pref_source="none")
    else:
        cont, restart = _continuation_with_restarts(case, cfg, cs, rule, snapshots_every, out, tag)
        eps_min = 10.0 ** cfg["eps_schedule"]["stop"]
    return _finish_migobug(case, cfg, cs, cont, restart, out, tag, env, eps_min)


def _continuation_with_restarts(case, cfg, cs, rule, snapshots_every, out, tag):
    """Restart rule of CFG['restart_rule'] (see there).  Starts from the step
    count logged for this case in --step-pref, if any; the time of the attempts
    that led to that step count is then taken from the log."""
    es = cfg["eps_schedule"]
    pref = STEP_PREF.get(tag)
    start = rule.get("start_steps_by_k", {}).get(case.k, rule.get("start_steps", cs["steps"]))
    L0 = int(pref["steps"]) if pref is not None else int(start)
    t_prior = float(pref["t_restart_attempts"]) if pref is not None else 0.0
    attempts, best, base = [], None, None

    def resume(c, eps_L, L):
        """Continue a stalled run from its last converged solution with step halving."""
        if not (rule.get("resume_halving") and c["sol"] is not None
                and c["eps_last"] * 1e4 > rule["eps_pi_target"]):
            return c
        rest = eps_L[eps_L < c["eps_last"] * (1 - 1e-12)]
        if not len(rest):
            return c
        c2 = P.run_continuation(case, rest, fix_ramp_units=cfg["fix_ramp_units"], adaptive=True,
                                step_control="per_target",
                                max_halvings=int(rule.get("resume_max_halvings", 6)),
                                max_nodes=cs["max_nodes"], tol_schedule=cs["tol_schedule"],
                                bc_tol=cs["bc_tol"], mesh_density=cs["mesh_density"],
                                max_consecutive_fail=int(rule.get("resume_max_failed_targets", 3)),
                                resume=(c["sol"], c["eps_last"]))
        say(f"   steps={L}: stalled at eps_pi={c['eps_last'] * 1e4:.2e}; resumed with halving -> "
            f"{c2['eps_last'] * 1e4:.2e}  {c2['time']:.1f}s")
        return dict(c2, time=c["time"] + c2["time"], log=c["log"] + c2["log"],
                    snapshots=c["snapshots"], failed_steps=c["failed_steps"] + c2["failed_steps"],
                    n_solves=c["n_solves"] + c2["n_solves"], n_refined=c2["n_refined"])

    n_max = max(0, (int(rule["max_steps"]) - L0) // int(rule["step_increment"]))
    for i in range(n_max + 1):
        L = L0 + i * int(rule["step_increment"])
        eps_L = eps_array(es["start"], es["stop"], L)
        back = rule.get("restart_backoff_decades")
        snap = None
        if i > 0 and back and base is not None and base["sol"] is not None:
            # partial restart: start the longer schedule from the first run's
            # snapshot `back` decades of eps before its stall point
            cand = [v for v in base["snapshots"].values() if v[0] >= base["eps_last"] * 10.0 ** back]
            snap = min(cand, key=lambda v: v[0]) if cand else None
        if snap is not None:
            rest = eps_L[eps_L < snap[0] * (1 - 1e-12)]
            c = P.run_continuation(case, rest, fix_ramp_units=cfg["fix_ramp_units"], adaptive=False,
                                   max_nodes=cs["max_nodes"], tol_schedule=cs["tol_schedule"],
                                   bc_tol=cs["bc_tol"], mesh_density=cs["mesh_density"],
                                   max_consecutive_fail=cs.get("max_consecutive_fail"), resume=(snap[1], snap[0]))
            c = dict(c, snapshots=base["snapshots"])
            say(f"   steps={L}: partial restart from eps_pi={snap[0] * 1e4:.2e} -> {c['eps_last'] * 1e4:.2e}  {c['time']:.1f}s")
            c = resume(c, eps_L, L)
            if rule.get("restart_full_fallback") and c["eps_last"] * 1e4 > rule["eps_pi_target"]:
                t_part = c["time"]
                cf = resume(_continuation(case, cfg, cs, eps_L, snapshots_every), eps_L, L)
                c = dict(cf, time=cf["time"] + t_part)
                say(f"   steps={L}: full restart -> {c['eps_last'] * 1e4:.2e}  (partial {t_part:.1f}s + full {cf['time']:.1f}s)")
        else:
            c = _continuation(case, cfg, cs, eps_L, snapshots_every)
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
                   restart_attempts=json.dumps(attempts, default=_js),
                   step_pref_source="logged" if pref is not None else "measured")
    log_path = out / "step_preference.csv"
    new = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, ["tag", "k", "steps", "eps_pi_last", "n_attempts", "t_restart_attempts",
                               "step_pref_source", "attempts"])
        if new:
            w.writeheader()
        w.writerow(dict(tag=tag, k=case.k, steps=chosen["steps"], eps_pi_last=chosen["eps_pi_last"],
                        n_attempts=len(attempts), t_restart_attempts=t_other,
                        step_pref_source=restart["step_pref_source"], attempts=restart["restart_attempts"]))
    return cont, restart


def _continuation(case, cfg, cs, eps, snapshots_every):
    snaps = ()
    if snapshots_every:
        # about 20 (C) snapshots per run whatever the number of steps
        every = snapshots_every if cfg.get("condC_dense") else max(snapshots_every, len(eps) // 20)
        snaps = tuple(range(0, len(eps), every)) + (len(eps) - 1,)
    return P.run_continuation(case, eps, fix_ramp_units=cfg["fix_ramp_units"], snapshots=snaps,
                              adaptive=cs["adaptive"],
                              max_halvings=cs["max_halvings"],
                              step_control=cs["step_control"],
                              mesh_reset=cs["mesh_reset"],
                              tol_fallback=cs["tol_fallback"],
                              max_nodes=cs["max_nodes"],
                              tol_schedule=cs["tol_schedule"], bc_tol=cs["bc_tol"],
                              mesh_density=cs["mesh_density"],
                              kappa=cs["kappa"], delta_state=cs["delta_state"],
                              reject_factor=cs["reject_factor"], secant=cs["secant"],
                              kl_measure=cs["kl_measure"],
                              max_consecutive_fail=cs.get("max_consecutive_fail"))


def _finish_migobug(case, cfg, cs, cont, restart, out, tag, env, eps_min):
    """Polish + sequence + fixed-topology QP + diagnostics of the kept continuation."""
    pd.DataFrame(cont["log"]).to_csv(out / "conv" / f"{tag}.csv", index=False)
    if cont.get("schedule"):
        pd.DataFrame(cont["schedule"]).to_csv(out / "conv" / f"{tag}_schedule.csv", index=False)
    if cont["sol"] is None:
        return dict(miGoBuG_status="no converged step"), cont, None, None
    t_tighten, tol_reached = 0.0, None
    cont_pol = cont
    if cs.get("final_tighten"):
        tg = P.tighten_final(case, cont, tols=tuple(cs["final_tighten"]), bc_tol=cs["bc_tol"] or 1e-2,
                             max_nodes=max(cs["max_nodes"], 50000))
        t_tighten, tol_reached = tg["time"], tg["tol_reached"]
        cont_pol = dict(cont, sol=tg["sol"])
    if cs["polish"]:
        pol = P.polish(case, cont_pol, tol=cfg["polish_tol"], bc_tol=cfg["polish_tol"],
                       max_nodes=cs.get("polish_max_nodes") or max(cs["max_nodes"], 50000))
    else:
        pol = dict(sol=None, success=False, time=0.0, max_rms=np.nan, max_bc_res=np.nan,
                   message="not attempted (profile without polish)")
    sol = pol["sol"] if pol["success"] else cont_pol["sol"]
    seq, traj = P.extract_sequence(case, sol, cont["weights"])
    if DEFER_FIXEDQP:
        # --defer-fixedqp: no Gurobi license here; the sequence is saved in
        # traj/*.npz and the fixed-topology QP (and its time) is added later
        ev = dict(runtime=np.nan, feasible=None, values=None)
    else:
        ev = P.evaluate_sequence(case, seq, env=env)
    ok = [r for r in cont["log"] if r["success"]]
    row = dict(
        miGoBuG_status="ok",
        t_continuation=cont["time"], t_tighten=t_tighten, tighten_tol_reached=tol_reached,
        t_polish=pol["time"], t_fixedQP=ev["runtime"],
        t_total=(cont["time"] + restart["t_restart_attempts"] + t_tighten + pol["time"]
                 + (0.0 if DEFER_FIXEDQP else ev["runtime"])),
        fixedQP_deferred=DEFER_FIXEDQP, **restart,
        n_rejected=cont.get("n_rejected"), n_secant_fail=cont.get("n_secant_fail"),
        t_failed_solves=float(sum(r["time"] for r in cont["log"] if not r["success"])),
        t_rejected_solves=float(cont.get("t_rejected", 0.0)),
        steps=cont["n_steps"], failed_steps=cont["failed_steps"],
        stalled_at_eps_pi=cont.get("stalled_at_eps_pi"),
        reached_eps_min=bool(cont["eps_last"] <= eps_min * (1 + 1e-9) or cont.get("binarized", False)),
        binarized_early=bool(cont.get("binarized", False)),
        max_kl_per_step=(max(a["kl_real"] for a in cont["schedule"]) if cont.get("schedule") else np.nan),
        max_kl_pointwise=(max(a.get("kl_pointmax", np.nan) for a in cont["schedule"])
                          if cont.get("schedule") else np.nan),
        max_dstate_per_step=(max(a["d_state"] for a in cont["schedule"]) if cont.get("schedule") else np.nan),
        final_tol=next((r["tol"] for r in reversed(cont["log"]) if r["success"]), np.nan),
        cont_settings=json.dumps(cs, default=_js),
        n_mesh_reset=cont.get("n_mesh_reset", 0), n_tol_fallback=cont.get("n_tol_fallback", 0),
        bvp_solves=cont["n_solves"], refined_solves=cont["n_refined"],
        eps_last=cont["eps_last"], eps_pi_last=cont["eps_last"] * 1e4,
        final_nodes=len(sol.x), max_nodes_seen=max(r["nodes"] for r in ok),
        newton_avg=float(np.nanmean([r["niter"] for r in ok])),
        newton_max=float(np.nanmax([r["niter"] for r in ok])),
        polish_success=pol["success"], polish_max_rms=pol["max_rms"],
        polish_max_bc=pol["max_bc_res"],
        seq_feasible=ev["feasible"], J_miGoBuG_ZOH=ev.get("J_true", np.nan),
        fixedQP_simult_chdis_kW=ev.get("simult_chdis_kW", np.nan),
        n_switches=int(np.sum(np.diff(seq) != 0)),
    )
    row.update(P.pmp_diagnostics(case, sol, cont["eps_last"], cont["weights"]))
    # final_*: the solution the sequence is extracted from (polished if the
    # polish succeeded); cont_*: the last continuation solution
    row.update({f"final_{k}": v for k, v in
                P.residual_report(case, sol, cont["eps_last"], cont["weights"]).items()})
    row.update({f"cont_{k}": v for k, v in
                P.residual_report(case, cont["sol"], cont["eps_last"], cont["weights"]).items()})
    row["final_is_polished"] = bool(pol["success"])
    row["polish_message"] = pol.get("message")
    np.savez_compressed(out / "traj" / f"{tag}_miGoBuG.npz",
                        t=sol.x, y=sol.y, seq=seq,
                        p_s=traj["p_s"], p_g=traj["p_g"], p_dg=traj["p_dg"], e=traj["e"],
                        **({f"fixedQP_{k}": v for k, v in ev["values"].items()}
                           if ev.get("values") else {}))
    return row, cont, sol, seq


def condition_C_profile(case, cfg, cont, out, tag):
    """Check (C) at every snapshot; eps_bar = smallest eps_pi reached from the
    top of the schedule without a failure."""
    rows = []
    for idx in sorted(cont["snapshots"]):
        eps, sol = cont["snapshots"][idx]
        if cfg.get("condC_certify"):                       # analytic bound only (Appendix A)
            r = dict(holds=None)
            r.update(P.certify_condition_C(case, sol, eps, cont["weights"]))
        else:
            r = P.check_condition_C(case, sol, eps, cont["weights"],
                                    n_grid=cfg["condC_grid"], node_stride=cfg["condC_node_stride"])
        rows.append(dict(tag=tag, step=idx, eps=eps, **r))
    if cfg.get("condC_certify"):
        eps_cert = None
        for r in rows:                   # smallest eps_pi down to which every snapshot is certified
            if r["cert_holds"]:
                eps_cert = r["eps_pi"]
            else:
                break
        first_fail = next((r["eps_pi"] for r in rows if not r["cert_holds"]), None)
        return rows, eps_cert, first_fail
    eps_bar = None
    for r in rows:                       # snapshots are in decreasing eps
        if r["holds"]:
            eps_bar = r["eps_pi"]
        else:
            break
    first_fail = next((r["eps_pi"] for r in rows if not r["holds"]), None)
    return rows, eps_bar, first_fail


def run_miqp(case, cfg, out, tag, env, pwl_L=None, limit=None):
    log = out / "gurobi_logs" / f"{tag}_{'MIQP' if pwl_L is None else f'PWL{pwl_L}'}.log"
    form = dict(binaries=cfg.get("miqp_binaries", "topology"),
                cg_ub=cfg.get("miqp_cg_ub", False))
    nf0 = cfg.get("miqp_numeric_focus", 0) if pwl_L is None else 0
    r = P.solve_benchmark(case, time_limit=limit or cfg["miqp_time_limit"],
                          mip_gap=cfg["mip_gap"], threads=cfg["threads"],
                          mip_focus=cfg["mip_focus"], pwl_L=pwl_L, env=env, log_file=log,
                          numeric_focus=nf0, extra_params=cfg.get("miqp_extra_params"), **form)
    nf = cfg.get("numeric_focus_retry", 0)
    # mem_limit (SoftMemLimit) is not numerical trouble: a re-solve would hit the same limit
    if nf and nf > nf0 and r["status_str"] not in P.VALID_MIQP_STATUS and r["status_str"] != "mem_limit":
        # e.g. status 13 (SUBOPTIMAL) in an earlier run: the bound exceeded two
        # feasible solutions, i.e. it was not valid.  Re-solve once, record it.
        say(f"   status {r['status_str']} -> re-solving with NumericFocus={nf}")
        first = r["status_str"]
        r = P.solve_benchmark(case, time_limit=limit or cfg["miqp_time_limit"],
                              mip_gap=cfg["mip_gap"], threads=cfg["threads"],
                              mip_focus=cfg["mip_focus"], pwl_L=pwl_L, env=env,
                              log_file=log.with_name(log.stem + "_NF.log"), numeric_focus=nf,
                              extra_params=cfg.get("miqp_extra_params"), **form)
        r["first_status"] = first
    with open(out / "progress" / f"{log.stem}.json", "w") as f:
        json.dump(r.get("progress", []), f, default=_js)
    if pwl_L is None and r.get("values"):
        np.savez_compressed(out / "traj" / f"{tag}_MIQP.npz",
                            **{k: v for k, v in r["values"].items()})
    return r


# --- heuristic baselines (baselines.py), costed in the benchmark model ---
def run_baselines(case, env, methods=("MINTOC", "SUR", "Adjoint")):
    args = (case.p_net, case.E_init, case.E_target, case.E_min, case.E_max, case.P_dg_init,
            case.S, case.Z, case.B, case.Delta, case.eta_s, case.eta_dg, case.eta_g,
            case.X_g, case.base_grid_costs, case.dt)
    bounds = dict(P_s_ub=case.P_s_ub, P_g_ub=case.P_g_ub, P_dg_lb=case.P_dg_lb,   # same bounds as the
                  P_dg_ub=case.P_dg_ub, U_ub=case.U_ub)                             # benchmark model
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


# ======================================================================
# 5. EXPERIMENTS
# ======================================================================
def _reuse_match(prev, case, cfg):
    """Can the Gurobi row `prev` (from --reuse-miqp) be used for `case`?
    Same grid costs, same node fingerprint, same MIQP formulation."""
    grid_costs = (case.base_grid_costs * case.scale).tolist()
    fp_now = json.dumps(dict(Delta=(case.Delta * case.scale).round(6).tolist(),
                             eta_g=case.eta_g.round(6).tolist()))
    try:
        fp = prev.get("grid_fingerprint")
        # rows written before the fingerprint existed
        # have NaN here -> rely on the exact match of the grid costs
        same_fp = (not isinstance(fp, str)) or fp == fp_now
        pc = json.loads(prev["grid_costs"])
        if not (len(pc) == len(grid_costs) and np.allclose(pc, grid_costs, rtol=1e-9) and same_fp):
            return False, "grid parameters differ from the reused run"
        if case.g > 0 and not cfg.get("reuse_any_formulation"):
            st = json.loads(prev["MIQP_settings"]) if isinstance(prev.get("MIQP_settings"), str) else {}
            prev_form = (st.get("binaries", "topology"), bool(st.get("cg_ub", False)),
                         int(st.get("numeric_focus", 0) or 0))
            cur_form = (cfg.get("miqp_binaries", "topology"), bool(cfg.get("miqp_cg_ub", False)),
                        int(cfg.get("miqp_numeric_focus", 0)))
            if prev_form != cur_form:
                return False, f"reused row used MIQP formulation {prev_form}, now {cur_form}"
    except Exception as ex:
        return False, f"cannot compare ({ex})"
    return True, "same problem"


def plan_main(cfg, cases, configs):
    """--plan: list which Gurobi solves the main experiment would run."""
    n_miqp = n_pwl = 0
    lines = []
    for config in configs:
        for ci, p_net in cases:
            case = make_case(cfg, p_net, config)
            prev = REUSE.get((config, ci))
            ok, why = (_reuse_match(prev, case, cfg) if prev is not None
                       else (False, "no valid row in the reused run"))
            if ok:
                st = prev.get("MIQP_status")
                pw = [L for L in cfg["pwl_redo_Ls"]] if (REDO_PWL and config in cfg["pwl_configs"]) else []
                n_pwl += len(pw)
                lines.append(f"  {config:>3}/{ci:<2}  reuse Gurobi ({st}, {float(prev.get('MIQP_time', 0)):.0f} s)"
                             + (f"; re-solve PWL L={pw}" if pw else ""))
            else:
                pw = cfg["pwl_Ls"] if config in cfg["pwl_configs"] else []
                n_miqp += 1
                n_pwl += len(pw)
                lines.append(f"  {config:>3}/{ci:<2}  SOLVE MIQP (<= {cfg['miqp_time_limit']:.0f} s): {why}"
                             + (f"; PWL L={pw}" if pw else "") + "; baselines")
    print("\n".join(lines))
    print(f"\nGurobi MIQP solves: {n_miqp} (each up to {cfg['miqp_time_limit']:.0f} s); "
          f"MILP-PWL solves: {n_pwl} (each up to {cfg['pwl_time_limit']:.0f} s).")
    print(f"Worst case Gurobi time: {(n_miqp * cfg['miqp_time_limit'] + n_pwl * cfg['pwl_time_limit']) / 3600:.1f} h "
          f"(plus miGoBuG and baselines). Nothing has been run.")


def exp_main(cfg, cases, configs, out, env, with_baselines=True):
    T = Table(out / "main.csv", ["config", "case"])
    TC = Table(out / "condC.csv", ["tag", "step"])
    eps = None                         # per-k schedule (cont_settings)
    for config in configs:
        for ci, p_net in cases:
            if T.has((config, ci)):
                continue
            tag = f"main_{config}_case{ci}"
            try:
                _main_row(T, TC, cfg, eps, config, ci, p_net, out, env, tag, with_baselines)
            except Exception:
                _log_error(out, tag)


REUSE_SOURCE = None
def _bound_check(row):
    """Gurobi's best bound must not exceed the cost of ANY feasible solution
    found by any method (miGoBuG, MINTOC, SUR, Adjoint, PWL + QP, Gurobi's own
    incumbent).  A violation means the bound is not valid (numerical trouble).
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
    """MILP-PWL row entries.  J_true: PWL dispatch re-costed with the quadratic
    objective.  seqQP: PWL switching sequence + fixed-topology QP, i.e. the SAME
    post-processing that miGoBuG-ZOH gets; without it
    the comparison would be unfair to the MILP."""
    c = {f"PWL{L}_status": r["status_str"], f"PWL{L}_time": r["runtime"],
         f"PWL{L}_J_true": r.get("J_true", np.nan),
         f"PWL{L}_gap_certified_pct": gap(r.get("J_true", np.nan), J_LB)}
    if r.get("seq") is not None:
        ev = P.evaluate_sequence(case, r["seq"], env=env)
        Jq = ev.get("J_true", np.nan) if ev["feasible"] else np.nan
        c.update({f"PWL{L}_J_seqQP": Jq, f"PWL{L}_time_seqQP": r["runtime"] + ev["runtime"],
                  f"PWL{L}_gap_seqQP_certified_pct": gap(Jq, J_LB)})
    return c


REUSE = {}          # (config, case) -> row of a previous main.csv  (--reuse-miqp)
REDO_PWL = False    # --redo-pwl: re-solve PWL even when Gurobi rows are reused
MIGOBUG_ONLY = False  # --migobug-only: never start MIQP / PWL / baselines
DEFER_FIXEDQP = False  # --defer-fixedqp: skip the fixed-topology QP (t_total then excludes it)
STEP_PREF = {}        # --step-pref: tag -> row of an earlier step_preference.csv
REUSE_PREFIX = ("MIQP_", "PWL", "MINTOC_", "SUR_", "Adjoint_")


def _main_row(T, TC, cfg, eps, config, ci, p_net, out, env, tag, with_baselines):
    say(f"[main] {tag}")
    case = make_case(cfg, p_net, config)
    row = dict(config=config, case=ci, N=case.N, g=case.g, k=case.k,
               grid_costs=(case.base_grid_costs * case.scale).tolist(),
               grid_fingerprint=json.dumps(dict(Delta=(case.Delta * case.scale).round(6).tolist(),
                                                eta_g=case.eta_g.round(6).tolist())))
    mg, cont, sol, seq = run_migobug(case, cfg, eps, out, tag,
                                     snapshots_every=cfg["condC_every"], env=env)
    row.update(mg)
    if seq is not None and case.g > 0:
        Xs = case.X_g[np.asarray(seq, int)]
        row["usage_miGoBuG"] = Xs.mean(axis=0).round(4).tolist()      # share of time each node is on
        row["multi_node_share_miGoBuG"] = float(np.mean(Xs.sum(axis=1) >= 2))
    if cont is not None and cont["snapshots"] and not cfg.get("skip_condC"):
        t_c = time.time()
        rows, eps_bar, first_fail = condition_C_profile(case, cfg, cont, out, tag)
        row["t_condC_check"] = time.time() - t_c          # verification only, not in t_total
        for r in rows:
            if not TC.has((r["tag"], r["step"])):
                TC.add(r)
        row.update(condC_eps_bar=eps_bar, condC_first_fail=first_fail)
    prev = REUSE.get((config, ci))
    if prev is not None:
        same, why = _reuse_match(prev, case, cfg)
        if not same:
            say(f"   {config}/{ci}: {why} -> re-solving Gurobi")
            prev = None
    if prev is not None:
        # same model, same Gurobi settings: copy instead of re-solving
        for k, v in prev.items():
            if k.startswith(REUSE_PREFIX):
                row[k] = v
        src = prev.get("MIQP_reused_from")
        row["MIQP_reused_from"] = src if isinstance(src, str) and src else REUSE_SOURCE
        Jm = row.get("J_miGoBuG_ZOH", np.nan)
        row.update(gap_vs_inc_pct=gap(Jm, float(prev["MIQP_J_inc"])),
                   gap_certified_pct=gap(Jm, float(prev["MIQP_J_LB"])))
        if REDO_PWL and not MIGOBUG_ONLY and config in cfg["pwl_configs"]:
            for L in cfg["pwl_redo_Ls"]:
                say(f"   PWL L={L} (re-solved, --redo-pwl)")
                r = run_miqp(case, cfg, out, tag, env, pwl_L=L, limit=cfg["pwl_time_limit"])
                row.update(pwl_columns(case, r, L, float(prev["MIQP_J_LB"]), env))
        _bound_check(row)
        T.add(row)
        say(f"   done (Gurobi/baselines reused): gap_cert={row['gap_certified_pct']:.4g}%  "
            f"miGoBuG={row.get('t_total', np.nan):.1f}s (cont {row.get('t_continuation', np.nan):.1f} "
            f"tight {row.get('t_tighten', 0):.1f} polish {row.get('t_polish', np.nan):.1f})  steps={row.get('steps')}  "
            f"solves={row.get('bvp_solves')} failed={row.get('failed_steps')}({row.get('t_failed_solves', 0):.0f}s) "
            f"rejected={row.get('n_rejected')}({row.get('t_rejected_solves', 0):.0f}s) secant_fail={row.get('n_secant_fail')}  "
            f"reached_eps_min={row.get('reached_eps_min')}  "
            f"res_max={row.get('final_res_max', np.nan):.3g}{'(polished)' if row.get('final_is_polished') else ''}  "
            f"condC_check={row.get('t_condC_check', 0):.0f}s")
        return
    if MIGOBUG_ONLY:
        # no reusable Gurobi row (different nodes, or invalid status in the
        # source): keep the miGoBuG result, gaps stay empty
        row["MIQP_reused_from"] = "none (--migobug-only)"
        T.add(row)
        say(f"   done (miGoBuG only, no bound available): J={row.get('J_miGoBuG_ZOH', np.nan):.6g}  "
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
        f"miGoBuG={row.get('t_total', np.nan):.1f}s  MIQP={mi['runtime']:.1f}s "
        f"({mi['status_str']})")


def exp_horizon(cfg, days, out, env):
    T = Table(out / "horizon.csv", ["series", "dataset", "N"])
    eps = None                         # per-k schedule (cont_settings)
    per = len(days[0]["p_net"])
    for d in cfg["horizon_datasets"]:
        start = (d - 1) * cfg["days_per_case"]
        # (a) horizon length at 30 min
        for N in cfg["horizon_N"]:
            ndays = int(np.ceil(N / per))
            if start + ndays > len(days):
                say(f"[horizon] dataset {d}, N={N}: not enough days, skipped")
                continue
            if T.has(("horizon", d, N)):
                continue
            p = net_load(cfg, days[start:start + ndays])[:N]
            _scaling_row(T, cfg, make_case(cfg, p, "III"), eps, out, env,
                         dict(series="horizon", dataset=d, N=N), f"hor_d{d}_N{N}")
        # (b) resolution at 72 h
        base = make_case(cfg, net_load(cfg, days[start:start + cfg["days_per_case"]]), "III")
        for fct in cfg["resolution_factors"]:
            case = base if fct == 1 else P.resample(base, fct)
            case = make_case(replace_dt(cfg, case.dt), case.p_net, "III")
            if T.has(("resolution", d, case.N)):
                continue
            _scaling_row(T, cfg, case, eps, out, env,
                         dict(series="resolution", dataset=d, N=case.N, dt=case.dt),
                         f"res_d{d}_N{case.N}")


def replace_dt(cfg, dt):
    c = dict(cfg)
    c["dt"] = dt
    return c


def _scaling_row(T, cfg, case, eps, out, env, keys, tag):
    try:
        _scaling_row_inner(T, cfg, case, eps, out, env, keys, tag)
    except Exception:
        _log_error(out, tag)


def _log_error(out, tag):
    tb = traceback.format_exc()
    say(f"   [error] {tag}: {tb.strip().splitlines()[-1]}  (see errors.log)")
    with open(out / "errors.log", "a") as f:
        f.write(f"\n[{time.ctime()}] {tag}\n{tb}")


def _scaling_row_inner(T, cfg, case, eps, out, env, keys, tag):
    say(f"[scaling] {tag}  (N={case.N}, k={case.k})")
    row = dict(keys, k=case.k, g=case.g)
    mg, *_ = run_migobug(case, cfg, eps, out, tag, env=env)
    row.update(mg)
    row["time_per_k"] = row.get("t_total", np.nan) / case.k
    if MIGOBUG_ONLY:                   # horizon/topology without branch-and-bound
        row["MIQP_status"] = "not run (--migobug-only)"
        T.add(row)
        say(f"   done (miGoBuG only): {row.get('t_total', np.nan):.1f}s  mesh={row.get('final_nodes')}  "
            f"newton={row.get('newton_avg', np.nan):.2f}  reached_eps_min={row.get('reached_eps_min')}")
        return
    mi = run_miqp(case, cfg, out, tag, env)
    row.update(MIQP_status=mi["status_str"], MIQP_time=mi["runtime"],
               MIQP_J_inc=mi.get("J_inc", np.nan), MIQP_J_LB=mi["J_LB"],
               MIQP_gap_pct=100 * mi["mip_gap"], MIQP_nodes=mi.get("node_count"),
               MIQP_n_bin=mi["n_bin"])
    row["gap_certified_pct"] = gap(row.get("J_miGoBuG_ZOH", np.nan), mi["J_LB"])
    row["time_per_k"] = row.get("t_total", np.nan) / case.k
    T.add(row)


def exp_topology(cfg, cases, out, env):
    """Topology scaling (Table R4 / VII), NESTED design.

    g = 1..4 are exactly Configurations II..V of the main benchmark, so they are
    taken from main.csv and not re-run.  This experiment only adds the g in
    cfg["topology_g"] (default [5]): the first g nodes of cfg["topology_nodes"]
    (G1..G4 of Config V plus G5).  An earlier version drew fresh random nodes for every g,
    which (i) made the rows for different g incomparable and (ii) could produce
    dominated nodes (see P.grid_envelope)."""
    T = Table(out / "topology.csv", ["dataset", "g"])
    eps = None                         # per-k schedule (cont_settings)
    nodes = cfg["topology_nodes"]
    for g in cfg["topology_g"]:
        segs, share = P.grid_envelope(nodes["Delta"][:g], nodes["eta_g"][:g], nodes["breakeven"][:g])
        if min(share) <= 0:
            say(f"[WARN] topology study g={g}: dominated node(s) {np.flatnonzero(np.array(share) <= 0) + 1}")
    with open(out / "topology_nodes.json", "w") as f:
        json.dump(nodes, f, indent=1)
    for ci, p_net in cases:
        if ci not in cfg["topology_datasets"]:
            continue
        for g in cfg["topology_g"]:
            if T.has((ci, g)):
                continue
            gc = dict(g=g, Delta=nodes["Delta"][:g], eta_g=nodes["eta_g"][:g],
                      breakeven=nodes["breakeven"][:g])
            case = make_case(cfg, p_net, None, grid_override=gc)
            _scaling_row(T, cfg, case, eps, out, env, dict(dataset=ci, g=g), f"top_d{ci}_g{g}")


def exp_sensitivity(cfg, cases, out, env):
    T = Table(out / "sensitivity.csv", ["dataset", "L", "eps0"])
    main = out / "main.csv"
    lb = {}
    if main.exists():
        df = pd.read_csv(main)
        for _, r in df[df["config"] == "III"].iterrows():
            lb[int(r["case"])] = r["MIQP_J_LB"]
    stop = cfg["eps_schedule"]["stop"]
    cs3 = cont_settings(cfg, 4)                       # Configuration III
    e_start = cfg["eps_schedule"]["start"]
    if cs3["step_control"] == "fisher":
        # the schedule is set by kappa (and delta_state), not by a step count
        runs = [(f"kappa={kp:g}", e_start, dict(kappa=kp)) for kp in cfg["sens_kappa"]] + \
               [(f"delta={ds:g}", e_start, dict(delta_state=ds)) for ds in cfg["sens_delta"]] + \
               [(f"kappa={cs3['kappa']:g}", e0, {}) for e0 in cfg["sens_eps0"]] + \
               [("kl=" + ("mean" if cs3["kl_measure"] == "max" else "max"), e_start,
                 dict(kl_measure="mean" if cs3["kl_measure"] == "max" else "max"))]
    else:
        runs = [(L, e_start, {}) for L in cfg["sens_L"]] + \
               [(cs3["steps"], e0, {}) for e0 in cfg["sens_eps0"]]
    for ci, p_net in cases:
        if ci not in cfg["sens_datasets"]:
            continue
        case = make_case(cfg, p_net, "III")
        if ci not in lb:
            say(f"[sens] no MIQP bound for case {ci} in main.csv -> solving MIQP")
            lb[ci] = run_miqp(case, cfg, out, f"sens_case{ci}", env)["J_LB"]
        for L, e0, ov in runs:
            if T.has((ci, L, e0)):
                continue
            tag = f"sens_d{ci}_L{L}_e{e0:g}".replace("=", "")
            say(f"[sens] {tag}")
            n_eps = L if isinstance(L, int) else 2          # fisher: only start/stop used
            try:
                mg, *_ = run_migobug(case, cfg, eps_array(e0, stop, n_eps), out, tag, env=env,
                                     cont_override=ov)
            except Exception:
                _log_error(out, tag)
                continue
            gamma = 10 ** ((stop - e0) / (L - 1)) if isinstance(L, int) else np.nan
            T.add(dict(dataset=ci, L=L, eps0=e0, gamma=gamma, J_LB=lb[ci],
                       gap_certified_pct=gap(mg.get("J_miGoBuG_ZOH", np.nan), lb[ci]),
                       converged=mg.get("miGoBuG_status") == "ok"
                       and mg.get("seq_feasible", False), **mg))


# ======================================================================
# 6. ENVIRONMENT + DATA DOCUMENTATION
# ======================================================================
def document(cfg, folder, days, cases, out):
    import numba, scipy
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=1, default=_js)
    env = dict(python=sys.version, platform=platform.platform(),
               processor=platform.processor(), cpu_count=os.cpu_count(),
               numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__,
               numba_threads=numba.get_num_threads(),
               gurobi=".".join(map(str, P.gp.gurobi.version())),
               data_file=folder, days=len(days),
               started=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(out / "environment.json", "w") as f:
        json.dump(env, f, indent=1)
    prof = []
    for i, d in enumerate(days):
        for j in range(len(d["p_net"])):
            prof.append(dict(day=i + 1, step=j, t_h=j * cfg["dt"], P_net=d["p_net"][j]))
    pd.DataFrame(prof).to_csv(out / "data_profiles.csv", index=False)
    stats = []
    for ci, p in cases:
        stats.append(dict(case=ci, N=len(p), mean_kW=p.mean(), std_kW=p.std(),
                          min_kW=p.min(), max_kW=p.max(),
                          share_negative=float(np.mean(p < 0))))
    pd.DataFrame(stats).to_csv(out / "data_stats.csv", index=False)


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


def warm_up(cfg, cases):
    """JIT-compile the numba kernels so compile time is not counted."""
    ci, p = cases[0]
    for config in ("I", "III"):
        c = make_case(cfg, p[:12], config)
        P.run_continuation(c, np.array([1e5, 1e3]), fix_ramp_units=cfg["fix_ramp_units"])


# ======================================================================
# 7. MAIN
# ======================================================================
def parse_range(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="+", default=["main", "horizon", "topology", "sensitivity"],
                    choices=["main", "horizon", "topology", "sensitivity"])
    ap.add_argument("--configs", nargs="+", default=["I", "II", "III", "IV", "V"])
    ap.add_argument("--cases", default="1-10", help="e.g. 1-10 or 1,3,5")
    ap.add_argument("--out", default=None, help="output folder (reuse it to resume)")
    ap.add_argument("--miqp-limit", type=float)
    ap.add_argument("--pwl-limit", type=float)
    ap.add_argument("--pwl-configs", nargs="*")
    ap.add_argument("--pwl-redo-Ls", nargs="*", type=int, default=None,
                    help="with --redo-pwl: which L to re-solve on reused rows (default 16 32)")
    ap.add_argument("--plan", action="store_true",
                    help="print which Gurobi solves the main experiment would run, then exit")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--reuse-miqp", default=None,
                    help="main.csv of an earlier run: copy its Gurobi, PWL and baseline "
                         "columns instead of re-solving (only miGoBuG is re-run)")
    ap.add_argument("--profile", choices=["original", "run3", "accurate", "fast", "fast_tight"],
                    default=None,
                    help="tolerance/mesh profile for every k (default: CFG profile_by_k, else profile)")
    ap.add_argument("--no-secant", action="store_true", help="plain warm start (default now)")
    ap.add_argument("--secant", action="store_true", help="secant predictor (slow: failed guesses are expensive)")
    ap.add_argument("--no-reject", action="store_true", help="never reject a converged step")
    ap.add_argument("--kl-measure", choices=["mean", "max"], default=None,
                    help="KL measure of the Fisher rule (max = first fast run)")
    ap.add_argument("--cont-max-nodes", type=int, default=None,
                    help="mesh-node cap per continuation solve (cheaper failures)")
    ap.add_argument("--reuse-any-formulation", action="store_true",
                    help="with --reuse-miqp: reuse Gurobi rows solved with another MIQP formulation "
                         "(e.g., topology binaries). For miGoBuG tuning runs only, not for final tables")
    ap.add_argument("--max-halvings", type=int, default=None,
                    help="stop after the step has been halved this many times (default 12)")
    ap.add_argument("--no-mesh-reset", action="store_true", help="no retry on the uniform mesh after a failure")
    ap.add_argument("--no-tol-fallback", action="store_true", help="no retry with the loose tolerance after a failure")
    ap.add_argument("--certify-condC", action="store_true",
                    help="(C) snapshots every --condC-every steps with the analytic bound of Appendix A")
    ap.add_argument("--condC-every", type=int, default=None)
    ap.add_argument("--skip-condC", action="store_true",
                    help="skip the condition (C) check (smoke tests; it is not part of the runtime)")
    ap.add_argument("--schedule", choices=["fisher", "persistent"], default=None,
                    help="override the epsilon schedule for every k (default: CFG cont_by_k)")
    ap.add_argument("--migobug-only", action="store_true",
                    help="run only miGoBuG (+ its fixed-topology QP); Gurobi bounds are taken "
                         "from --reuse-miqp where the grid nodes match, otherwise gaps stay empty")
    ap.add_argument("--defer-fixedqp", action="store_true",
                    help="with --migobug-only: do not solve the fixed-topology QP (no Gurobi needed); "
                         "J, seq_feasible and t_fixedQP stay empty and must be filled in later")
    ap.add_argument("--start-steps", type=int, default=None,
                    help="restart rule: first step count for every k except those in start_steps_by_k")
    ap.add_argument("--step-increment", type=int, default=None, help="restart rule: steps added per restart")
    ap.add_argument("--max-steps", type=int, default=None, help="restart rule: largest step count tried")
    ap.add_argument("--max-consecutive-fail", type=int, default=None,
                    help="profile original: stop an attempt after this many failed steps in a row")
    ap.add_argument("--resume-halving", action="store_true",
                    help="restart rule: continue a stalled attempt from its last converged solution with step halving first")
    ap.add_argument("--resume-max-failed-targets", type=int, default=None, help="resume: give up after this many failed targets in a row")
    ap.add_argument("--resume-max-halvings", type=int, default=None, help="resume: step halvings per target")
    ap.add_argument("--restart-backoff", type=float, default=None,
                    help="restart rule: restart longer schedules from the first run's snapshot this many decades of eps before its stall")
    ap.add_argument("--restart-full-fallback", action="store_true",
                    help="restart rule: if a partial restart fails, run the same step count from scratch")
    ap.add_argument("--step-pref", default=None,
                    help="step_preference.csv of an earlier run: start each case from its logged "
                         "step count (the logged time of the earlier attempts is kept in t_total)")
    ap.add_argument("--keep-run3-nodes", action="store_true",
                    help="use the earlier Configuration V nodes (dominated G4) so that earlier "
                         "Gurobi bounds can be reused; for tuning the continuation only")
    ap.add_argument("--redo-pwl", action="store_true",
                    help="re-solve MILP-PWL even when --reuse-miqp copies the Gurobi rows "
                         "(needed once to get the PWL sequence + fixed-QP columns)")
    ap.add_argument("--eps-steps", type=int, default=None,
                    help="override the number of continuation steps (debug)")
    ap.add_argument("--debug-truncate", type=int, default=None,
                    help="truncate every case to N steps (smoke test only)")
    a = ap.parse_args()

    cfg = dict(CFG)
    if a.miqp_limit: cfg["miqp_time_limit"] = a.miqp_limit
    if a.pwl_limit: cfg["pwl_time_limit"] = a.pwl_limit
    if a.pwl_configs is not None: cfg["pwl_configs"] = a.pwl_configs
    if a.pwl_redo_Ls is not None: cfg["pwl_redo_Ls"] = a.pwl_redo_Ls
    if a.threads: cfg["threads"] = a.threads
    if a.schedule:
        cfg["schedule_override"] = a.schedule
    if a.profile:
        cfg["profile_cli"] = a.profile
    cfg["skip_condC"] = a.skip_condC
    if a.certify_condC:
        cfg["condC_certify"] = True
        cfg["condC_dense"] = True
    if a.condC_every:
        cfg["condC_every"] = a.condC_every
    cfg["reuse_any_formulation"] = a.reuse_any_formulation
    ov = {}
    if a.no_secant: ov["secant"] = False
    if a.secant: ov["secant"] = True
    if a.no_reject: ov["reject_factor"] = float("inf")
    if a.kl_measure: ov["kl_measure"] = a.kl_measure
    if a.cont_max_nodes: ov["max_nodes"] = a.cont_max_nodes
    if a.max_halvings is not None: ov["max_halvings"] = a.max_halvings
    if a.no_mesh_reset: ov["mesh_reset"] = False
    if a.no_tol_fallback: ov["tol_fallback"] = None
    if a.max_consecutive_fail: ov["max_consecutive_fail"] = a.max_consecutive_fail
    cfg["cli_overrides"] = ov
    if a.keep_run3_nodes:
        cfg["configs"] = dict(cfg["configs"], V=dict(
            g=4, Delta=[0.65, 0.95, 0.45, 0.80], eta_g=[0.95, 0.93, 0.97, 0.96],
            breakeven=[0.60, 0.45, 0.80, 0.55]))
    if a.eps_steps:
        cfg["eps_steps_override"] = a.eps_steps
        cfg["eps_schedule"] = dict(cfg["eps_schedule"], steps=a.eps_steps)
        cfg["sens_L"] = [max(10, a.eps_steps // 2), a.eps_steps]
        cfg["condC_every"] = max(5, a.eps_steps // 4)
    if cfg["bvp_tol_schedule"] is not None:
        P.TOL_SCHEDULE = tuple(tuple(x) for x in cfg["bvp_tol_schedule"])

    out = Path(a.out or f"results/run_{time.strftime('%Y%m%d_%H%M%S')}")
    for sub in ("", "conv", "traj", "gurobi_logs", "progress"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    say(f"output -> {out}")

    folder, days = load_days(cfg)
    k = cfg["days_per_case"]
    want = parse_range(a.cases)
    cases = []
    for ci in range(1, len(days) // k + 1):
        if ci in want:
            p = net_load(cfg, days[(ci - 1) * k: ci * k])
            if a.debug_truncate:
                p = p[:a.debug_truncate]
            cases.append((ci, p))
    say(f"{len(days)} days -> {len(days) // k} cases; running cases {[c for c, _ in cases]}")
    if a.debug_truncate:
        cfg["horizon_N"] = [n for n in cfg["horizon_N"] if n <= a.debug_truncate] or [a.debug_truncate]
        cfg["resolution_factors"] = [1]

    global REDO_PWL, MIGOBUG_ONLY, DEFER_FIXEDQP
    REDO_PWL = a.redo_pwl
    MIGOBUG_ONLY = a.migobug_only
    DEFER_FIXEDQP = a.defer_fixedqp
    rr = dict(cfg["restart_rule"])
    if a.start_steps:
        rr["start_steps"] = a.start_steps
        rr["start_steps_by_k"] = {}                  # --start-steps applies to every k
    if a.step_increment: rr["step_increment"] = a.step_increment
    if a.max_steps: rr["max_steps"] = a.max_steps
    if a.resume_halving: rr["resume_halving"] = True
    if a.restart_backoff: rr["restart_backoff_decades"] = a.restart_backoff
    if a.restart_full_fallback: rr["restart_full_fallback"] = True
    if a.resume_max_failed_targets: rr["resume_max_failed_targets"] = a.resume_max_failed_targets
    if a.resume_max_halvings: rr["resume_max_halvings"] = a.resume_max_halvings
    cfg["restart_rule"] = rr
    if a.step_pref:
        for r in csv.DictReader(open(a.step_pref)):
            STEP_PREF[r["tag"]] = r
    if DEFER_FIXEDQP and not MIGOBUG_ONLY:
        ap.error("--defer-fixedqp requires --migobug-only")
    if MIGOBUG_ONLY and "sensitivity" in a.exp:
        ap.error("--migobug-only is implemented for main, horizon and topology")
    check_grid_nodes(cfg, out)
    if a.reuse_miqp:
        global REUSE_SOURCE
        REUSE_SOURCE = a.reuse_miqp
        prev = pd.read_csv(a.reuse_miqp)
        for _, r in prev.iterrows():
            if str(r.get("MIQP_status")) not in P.VALID_MIQP_STATUS and r["config"] != "I":
                say(f"   {r['config']}/{int(r['case'])}: MIQP status {r.get('MIQP_status')} "
                    f"-> not reused (bound not reliable), re-solving")
                continue                       # interrupted (11), suboptimal (13), ...
            if r["config"] != "I" and '"perspective": true' not in str(r.get("MIQP_settings", "")):
                say(f"[warn] {r['config']}/{r['case']} in {a.reuse_miqp} is not perspective-based; not reused")
                continue
            try:
                prev_limit = json.loads(r["MIQP_settings"]).get("time_limit", 0)
            except Exception:
                prev_limit = 0
            if str(r.get("MIQP_status")) == "time_limit" and cfg["miqp_time_limit"] > prev_limit:
                say(f"   {r['config']}/{int(r['case'])} timed out at {prev_limit:g} s -> re-solving "
                    f"with {cfg['miqp_time_limit']:g} s")
                continue                       # re-solve with the longer limit
            REUSE[(r["config"], int(r["case"]))] = r.to_dict()
        say(f"reusing Gurobi/baseline results for {len(REUSE)} rows from {a.reuse_miqp}")
    if a.plan:
        plan_main(cfg, cases, [c for c in cfg["configs"] if c in (a.configs or cfg["configs"])])
        return
    document(cfg, folder, days, cases, out)
    env = P.gurobi_env(0)
    say("warming up numba ...")
    warm_up(cfg, cases)

    t0 = time.time()
    for e in a.exp:
        try:
            if e == "main":
                exp_main(cfg, cases, a.configs, out, env, with_baselines=not a.no_baselines)
            elif e == "horizon":
                exp_horizon(cfg, days, out, env)
            elif e == "topology":
                exp_topology(cfg, cases, out, env)
            elif e == "sensitivity":
                exp_sensitivity(cfg, cases, out, env)
        except Exception:
            traceback.print_exc()
            with open(out / "errors.log", "a") as f:
                f.write(f"\n[{time.ctime()}] experiment {e}\n{traceback.format_exc()}")
    say(f"finished in {(time.time() - t0) / 3600:.2f} h -> {out}")


if __name__ == "__main__":
    main()