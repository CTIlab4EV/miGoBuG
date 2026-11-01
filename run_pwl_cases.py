"""MILP-PWL benchmark (L = 8, 16, 32) for chosen configurations, outside the main driver.

Solves only the piecewise-linear MILPs (same model, settings and post-processing as the
main benchmark: re-costing with the quadratic objective and the fixed-sequence QP), and
takes the MINLP best bound J_LB from an existing results folder.  miGoBuG and the MINLP are
not re-run.  Resumable: cases already in <out>/pwl.csv are skipped.

    export GRB_WLSACCESSID=...  GRB_WLSSECRET=...  GRB_LICENSEID=...
    python run_pwl_cases.py --configs I II IV V --cases 1-10
    python run_pwl_cases.py --merge          # adds the PWL columns to results/final_pc/main.csv

Time: each MILP is limited to --pwl-limit seconds (default CFG, 3600 s).
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

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
    ap.add_argument("--configs", nargs="+", default=["I", "II", "IV", "V"])
    ap.add_argument("--cases", default="1-10")
    ap.add_argument("--Ls", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--results", default="results/final_pc", help="folder with main.csv (J_LB is read from it)")
    ap.add_argument("--out", default="results/pwl_all")
    ap.add_argument("--pwl-limit", type=float, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--soft-mem-limit", type=float, default=10.0,
                    help="GB; Gurobi stops cleanly above it instead of the container killing the run (default 10)")
    ap.add_argument("--nodefile-start", type=float, default=None, help="GB before Gurobi writes nodes to disk (default: CFG, 2)")
    ap.add_argument("--merge", action="store_true", help="merge <out>/pwl.csv into <results>/main.csv and exit")
    a = ap.parse_args()

    res_csv = Path(a.results) / "main.csv"
    out = Path(a.out)
    pwl_csv = out / "pwl.csv"
    if a.merge:
        merge(res_csv, pwl_csv)
        return

    cfg = dict(R.CFG)
    if a.pwl_limit is not None:
        cfg["pwl_time_limit"] = a.pwl_limit
    if a.threads is not None:
        cfg["threads"] = a.threads
    extra = dict(cfg.get("miqp_extra_params") or {})
    if a.soft_mem_limit:
        extra["SoftMemLimit"] = a.soft_mem_limit
    if a.nodefile_start is not None:
        extra["NodefileStart"] = a.nodefile_start
    cfg["miqp_extra_params"] = extra
    if extra.get("NodefileDir"):
        nd = Path(extra["NodefileDir"])
        nd.mkdir(parents=True, exist_ok=True)
        # node files left behind by runs that were killed (Gurobi removes its own on a normal exit);
        # skipped if any other Gurobi run is active, whose files may still be in use
        others = [l for l in subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()
                  if ("run_pwl_cases" in l or "run_gurobi_cases" in l or "run_revision_benchmarks" in l)
                  and str(os.getpid()) != l.split()[0]]
        if not others:
            for d in nd.glob("grbnodes*"):
                shutil.rmtree(d, ignore_errors=True)
    for sub in ("", "gurobi_logs", "progress", "traj"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rows = {(r["config"], int(r["case"])): r for r in csv.DictReader(open(res_csv))}
    done = set()
    if pwl_csv.exists():
        done = {(r["config"], int(r["case"])) for r in csv.DictReader(open(pwl_csv))}
    folder, days = R.load_days(cfg)
    kd = cfg["days_per_case"]
    env = P.gurobi_env(0)
    R.say(f"MILP-PWL L={a.Ls}, limit {cfg['pwl_time_limit']:g} s, configs {a.configs}, cases {a.cases}, Gurobi extra {extra}")

    for config in a.configs:
        for ci in parse_cases(a.cases):
            if (config, ci) in done:
                continue
            tag = f"main_{config}_case{ci}"
            case = R.make_case(cfg, R.net_load(cfg, days[(ci - 1) * kd: ci * kd]), config)
            J_LB = float(rows[(config, ci)]["MIQP_J_LB"])
            row = dict(config=config, case=ci)
            t0 = time.time()
            for L in a.Ls:
                r = R.run_miqp(case, cfg, out, tag, env, pwl_L=L, limit=cfg["pwl_time_limit"])
                row.update(R.pwl_columns(case, r, L, J_LB, env))
            row = {k: (json.dumps(v, default=R._js) if isinstance(v, (dict, list, tuple)) else v) for k, v in row.items()}
            new = not pwl_csv.exists()
            cols = list(row) if new else next(csv.reader(open(pwl_csv)))
            with open(pwl_csv, "a", newline="") as f:
                w = csv.DictWriter(f, cols)
                if new:
                    w.writeheader()
                w.writerow({k: row.get(k, "") for k in cols})
            R.say(f"[pwl] {config}/{ci} done in {time.time() - t0:.0f} s -> {pwl_csv}")


def merge(res_csv, pwl_csv):
    if not pwl_csv.exists():
        raise SystemExit(f"{pwl_csv} does not exist yet: run the MILP solves first "
                         "(python run_pwl_cases.py --configs I II IV V --cases 1-10), then --merge.")
    rows = list(csv.DictReader(open(res_csv)))
    cols = list(rows[0])
    new = {(r["config"], r["case"]): r for r in csv.DictReader(open(pwl_csv))}
    backup = res_csv.with_name("main.before_pwl_merge.csv")
    if not backup.exists():
        shutil.copy(res_csv, backup)
    n = 0
    for r in rows:
        p = new.get((r["config"], r["case"]))
        if p:
            for k, v in p.items():
                if k.startswith("PWL"):
                    if k not in cols:
                        cols.append(k)
                    r[k] = v
            n += 1
    with open(res_csv, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in cols} for r in rows])
    print(f"merged PWL columns of {n} cases into {res_csv} (backup {backup})")


if __name__ == "__main__":
    main()
