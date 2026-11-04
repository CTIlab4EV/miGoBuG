"""Summary of the regularization threshold of Appendix A from a threshold.csv
(written by run_benchmarks.py --threshold).

    python threshold_summary.py reference_results/threshold.csv

For every case of Configurations II-V: eps_bar = max over the continuation of
cert_eps_bar, and the smallest eps_pi down to which the convexity condition
holds at every snapshot from the start of the continuation.
"""
import csv
import math
import statistics as st
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "reference_results/threshold.csv"
by = defaultdict(list)
for r in csv.DictReader(open(path)):
    by[r["tag"]].append(r)

ebar, certified = [], []
for tag, rows in sorted(by.items()):
    if "_I_" in tag:                     # Configuration I: one topology, D(t) = 0
        continue
    vals = [float(r["cert_eps_bar"]) for r in rows if r["cert_eps_bar"] not in ("", "nan")]
    down_to = None
    for r in sorted(rows, key=lambda r: -float(r["eps_pi"])):
        if r["cert_holds"] != "True":
            break
        down_to = float(r["eps_pi"])
    ebar.append(max(vals))
    certified.append(down_to)
    print(f"{tag:16s} eps_bar = {max(vals):.4f}   certified from 1e9 down to eps_pi = {down_to:.3g}")

start = max(float(r["eps_pi"]) for rows in by.values() for r in rows)
worst = max(certified)
print(f"\n{len(ebar)} cases: eps_bar max {max(ebar):.3f}, median {st.median(ebar):.3f}; "
      f"certified down to eps_pi <= {worst:.2f} in every case "
      f"({math.log10(start / worst):.2f} decades from eps_pi = {start:.0e})")
