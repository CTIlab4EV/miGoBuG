"""Gurobi MINLP (convex MIQP) for selected cases,
outside the main driver, e.g. in a container with its own memory limit.

The output folder gets a main.csv in the same format as the driver's, so it can
be passed to   run_revision_benchmarks.py --reuse-miqp <out>/main.csv
(the driver checks grid parameters, MIQP formulation and status before reusing).
Model, formulation and Gurobi settings are taken from CFG in
run_revision_benchmarks.py, so the results are the same problem as in the
main benchmark.  Cases already in <out>/main.csv are skipped (resumable).

Usage (from the miGoBuG folder; Gurobi credentials in the environment):
    export GRB_WLSACCESSID=...  GRB_WLSSECRET=...  GRB_LICENSEID=...
    python run_gurobi_cases.py --config V --cases 1
    python run_gurobi_cases.py --config V --cases 1-10 --out results/gurobi_V

Memory: NodefileStart (GB) moves Gurobi's node tree to disk beyond that size
(default from CFG, 2 GB).  --soft-mem-limit X stops Gurobi cleanly at X GB
(status MEM_LIMIT) instead of the process being killed; such a row is not a
valid benchmark result and will not be reused.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import run_revision_benchmarks as R

P = R.P


def parse_cases(s):
    out = set()
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="V", choices=["I", "II", "III", "IV", "V"])
    ap.add_argument("--cases", default="1", help="e.g. 1 or 1-10 or 1,3,5")
    ap.add_argument("--out", default=None, help="output folder (default results/gurobi_<config>)")
    ap.add_argument("--threads", type=int, default=None, help="Gurobi threads (default: CFG, 0 = all cores)")
    ap.add_argument("--nodefile-start", type=float, default=None, help="GB before nodes go to disk (default: CFG)")
    ap.add_argument("--nodefile-dir", default=None, help="folder for node files (default: CFG)")
    ap.add_argument("--soft-mem-limit", type=float, default=None, help="GB; Gurobi stops cleanly above it")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="smoke tests only (default: CFG, 3600 s); a shorter limit makes the row non-reusable")
    a = ap.parse_args()

    cfg = dict(R.CFG)
    extra = dict(cfg.get("miqp_extra_params") or {})
    if a.nodefile_start is not None:
        extra["NodefileStart"] = a.nodefile_start
    if a.nodefile_dir is not None:
        extra["NodefileDir"] = a.nodefile_dir
    if a.soft_mem_limit is not None:
        extra["SoftMemLimit"] = a.soft_mem_limit
    if extra.get("NodefileDir"):
        Path(extra["NodefileDir"]).mkdir(parents=True, exist_ok=True)
    cfg["miqp_extra_params"] = extra
    if a.threads is not None:
        cfg["threads"] = a.threads
    if a.time_limit is not None:
        cfg["miqp_time_limit"] = a.time_limit

    out = Path(a.out or f"results/gurobi_{a.config}")
    for sub in ("", "gurobi_logs", "progress", "traj"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    csv_path = out / "main.csv"
    done = set()
    if csv_path.exists():
        done = {(r["config"], int(r["case"])) for r in csv.DictReader(open(csv_path))}

    folder, days = R.load_days(cfg)
    kd = cfg["days_per_case"]
    env = P.gurobi_env(0)
    R.say(f"Gurobi settings: time limit {cfg['miqp_time_limit']:g} s, MIPGap {cfg['mip_gap']:g}, "
          f"threads {cfg['threads'] or 'all'}, extra {extra}")

    for ci in parse_cases(a.cases):
        if (a.config, ci) in done:
            R.say(f"{a.config}/{ci} already in {csv_path}, skipped")
            continue
        tag = f"main_{a.config}_case{ci}"
        case = R.make_case(cfg, R.net_load(cfg, days[(ci - 1) * kd: ci * kd]), a.config)
        R.say(f"[gurobi] {tag}  (k={case.k})")
        row = dict(config=a.config, case=ci, N=case.N, g=case.g, k=case.k,
                   grid_costs=(case.base_grid_costs * case.scale).tolist(),
                   grid_fingerprint=json.dumps(dict(Delta=(case.Delta * case.scale).round(6).tolist(),
                                                    eta_g=case.eta_g.round(6).tolist())))
        t0 = time.time()
        mi = R.run_miqp(case, cfg, out, tag, env)
        row.update(MIQP_status=mi["status_str"], MIQP_time=mi["runtime"],
                   MIQP_J_inc=mi.get("J_inc", np.nan), MIQP_J_LB=mi["J_LB"],
                   MIQP_gap_pct=100 * mi["mip_gap"], MIQP_nodes=mi.get("node_count"),
                   MIQP_n_bin=mi["n_bin"], MIQP_n_var=mi["n_var"],
                   MIQP_simult_chdis_kW=mi.get("simult_chdis_kW", np.nan),
                   MIQP_t_first_inc=(mi["progress"][0][0] if mi.get("progress") else np.nan),
                   MIQP_settings=mi["settings"], MIQP_first_status=mi.get("first_status"),
                   MIQP_extra_params=json.dumps(extra))
        if mi.get("seq") is not None and case.g > 0:
            Xm = case.X_g[np.asarray(mi["seq"], int)]
            row["MIQP_usage"] = Xm.mean(axis=0).round(4).tolist()
            row["MIQP_multi_node_share"] = float(np.mean(Xm.sum(axis=1) >= 2))
        R._bound_check(row)
        row = {k: (json.dumps(v, default=R._js) if isinstance(v, (dict, list, tuple)) else v)
               for k, v in row.items()}
        new = not csv_path.exists()
        if not new:                                      # keep one header for all rows
            cols = next(csv.reader(open(csv_path)))
            if set(row) - set(cols):
                raise SystemExit(f"column mismatch with {csv_path}: {sorted(set(row) - set(cols))}")
        else:
            cols = list(row)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, cols)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in cols})
        R.say(f"   done: MIQP {mi['status_str']} in {mi['runtime']:.1f} s, final gap "
              f"{100 * mi['mip_gap']:.3f}%  (case total {time.time() - t0:.0f} s) -> {csv_path}")


if __name__ == "__main__":
    main()
