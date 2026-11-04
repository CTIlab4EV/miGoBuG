"""MILP-PWL benchmark (L = 8, 16, 32) for chosen configurations, outside the main run.

Solves only the piecewise-linear MILPs, with the same model, settings and post-processing as
run_benchmarks.py, and takes the MINLP best bound from an existing results folder.  Resumable:
cases already in <out>/pwl.csv are skipped.

    python run_pwl_cases.py --configs I II IV V --results results/main --out results/pwl_all
    python run_pwl_cases.py --results results/main --out results/pwl_all --merge
"""
import argparse
import csv
import json
import time
from pathlib import Path

import run_benchmarks as R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=["I", "II", "IV", "V"])
    ap.add_argument("--cases", default="1-10")
    ap.add_argument("--Ls", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--results", default="results/main", help="folder with main.csv (the MINLP bound is read from it)")
    ap.add_argument("--out", default="results/pwl_all")
    ap.add_argument("--pwl-limit", type=float, default=None, help="time limit of each MILP [s] (default 3600)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--soft-mem-limit", type=float, default=10.0,
                    help="GB; Gurobi stops cleanly above it (default 10)")
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
    extra = dict(cfg["miqp_extra_params"], SoftMemLimit=a.soft_mem_limit)
    cfg["miqp_extra_params"] = extra
    Path(extra["NodefileDir"]).mkdir(parents=True, exist_ok=True)
    for sub in ("", "gurobi_logs", "progress", "traj"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rows = {(r["config"], int(r["case"])): r for r in csv.DictReader(open(res_csv))}
    done = set()
    if pwl_csv.exists():
        done = {(r["config"], int(r["case"])) for r in csv.DictReader(open(pwl_csv))}
    days = R.load_days(cfg)
    env = R.P.gurobi_env(0)
    R.say(f"MILP-PWL L={a.Ls}, limit {cfg['pwl_time_limit']:g} s, configs {a.configs}, cases {a.cases}, Gurobi extra {extra}")

    for config in a.configs:
        for ci in R.parse_range(a.cases):
            if (config, ci) in done:
                continue
            tag = f"main_{config}_case{ci}"
            case = R.make_case(cfg, R.case_load(cfg, days, ci), config)
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
        raise SystemExit(f"{pwl_csv} does not exist yet: run the MILP solves first.")
    rows = list(csv.DictReader(open(res_csv)))
    cols = list(rows[0])
    new = {(r["config"], r["case"]): r for r in csv.DictReader(open(pwl_csv))}
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
    print(f"merged PWL columns of {n} cases into {res_csv}")


if __name__ == "__main__":
    main()
