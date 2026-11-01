"""Per-case schedule selection: for every case, among all tuning runs
(results/<prefix>_L<steps>_<config>/main.csv) keep the fastest run that ends at
eps_pi <= TARGET.  Writes a step-preference file for --step-pref (one attempt,
no earlier runs counted: the reported time is that of the selected schedule,
as stated in Section IV-E) and a list of cases without such a run."""
import csv, glob, json, re, sys
from collections import defaultdict
prefix, target = sys.argv[1], float(sys.argv[2])
out_pref, out_missing, out_summary = sys.argv[3], sys.argv[4], sys.argv[5]
runs = defaultdict(list)
for f in glob.glob(f"results/{prefix}_L*_*/main.csv"):
    L = int(re.search(r"_L(\d+)_", f).group(1))
    for r in csv.DictReader(open(f)):
        try:
            runs[(r["config"], int(r["case"]))].append(dict(L=L, eps=float(r["eps_pi_last"]), t=float(r["t_total"]), k=r["k"]))
        except (KeyError, ValueError):
            pass
pref, missing, summary = [], defaultdict(list), []
for (c, i), rs in sorted(runs.items(), key=lambda x: (["I", "II", "III", "IV", "V"].index(x[0][0]), x[0][1])):
    ok = [r for r in rs if r["eps"] <= target]
    pick = min(ok, key=lambda r: r["t"]) if ok else min(rs, key=lambda r: r["eps"])
    if not ok:
        missing[c].append(str(i))
    pref.append(dict(tag=f"main_{c}_case{i}", k=pick["k"], steps=pick["L"], eps_pi_last=pick["eps"], n_attempts=1,
                     t_restart_attempts=0.0, step_pref_source="per-case selection",
                     attempts=json.dumps(sorted([[r["L"], r["eps"], round(r["t"], 2)] for r in rs]))))
    summary.append(f"{c}-{i}: steps {pick['L']}, eps_pi {pick['eps']:.2e}, {pick['t']:.1f} s"
                   f"{'' if ok else '  (no run reached the target)'}  [{len(ok)} of {len(rs)} runs reached it]")
with open(out_pref, "w", newline="") as f:
    w = csv.DictWriter(f, list(pref[0])); w.writeheader(); w.writerows(pref)
with open(out_missing, "w") as f:
    for c, cases in missing.items():
        f.write(f"{c} {','.join(cases)}\n")
open(out_summary, "w").write("\n".join(summary) + "\n")
print("\n".join(summary))
print(f"cases without a run at eps_pi <= {target:g}: {sum(len(v) for v in missing.values())}")
