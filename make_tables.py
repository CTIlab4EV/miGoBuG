"""LaTeX rows of Tables III, IV and V of the paper from ONE results folder.

    python make_tables.py reference_results

Prints the rows and writes them to <folder>/tables.tex.  Gaps are 100 (J - J_MINLP) / J.
"""
import csv
import statistics as st
import sys
from pathlib import Path

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "reference_results")
rows = list(csv.DictReader(open(folder / "main.csv")))
by = {(r["config"], int(r["case"])): r for r in rows}
CONFIGS = ["I", "II", "III", "IV", "V"]


def num(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def ft(x):
    """Time in seconds with a precision suited to its size."""
    return f"{x:.3f}" if x < 0.1 else f"{x:.2f}" if x < 1 else f"{x:.1f}" if x < 100 else f"{x:.0f}"


def fx(x, d):
    """Fixed-point format without a negative zero."""
    t = f"{x:.{d}f}"
    if t.startswith("-") and float(t) == 0:
        t = t[1:]
    return t.replace("-", "$-$")


# ---------------------------------------------------------------- Table III: MINLP and miGoBuG ZOH per dataset
table3 = []
for i in range(1, 11):
    cells = [str(i)]
    for c in CONFIGS:
        r = by[(c, i)]
        minlp = ft(num(r, "MIQP_time")) if r["MIQP_status"] == "optimal" else f"{num(r, 'MIQP_gap_pct'):.2f}$^\\dagger$"
        cells += [minlp, f"{num(r, 't_total'):.1f}", fx(num(r, "gap_vs_inc_pct"), 3)]
    table3.append(" & ".join(cells) + "\\\\")

# ---------------------------------------------------------------- Table IV: all methods, mean (max) gap and median time
METHODS = [("miGoBuG ZOH", "J_miGoBuG_ZOH", "t_total", None),
           ("MILP-PWL, $L=8$", "PWL8_J_true", "PWL8_time", "PWL8_status"),
           ("MILP-PWL, $L=16$", "PWL16_J_true", "PWL16_time", "PWL16_status"),
           ("MILP-PWL, $L=32$", "PWL32_J_true", "PWL32_time", "PWL32_status"),
           ("MILP-PWL, $L=16$ + QP", "PWL16_J_seqQP", "PWL16_time_seqQP", "PWL16_status"),
           ("MILP-PWL, $L=32$ + QP", "PWL32_J_seqQP", "PWL32_time_seqQP", "PWL32_status"),
           ("MS MINTOC", "MINTOC_J", "MINTOC_time", None),
           ("SUR NLP", "SUR_J", "SUR_time", None),
           ("Adjoint NLP", "Adjoint_J", "Adjoint_time", None)]


def fgap(x):
    return f"{x:.3f}" if abs(x) < 10 else f"{x:.2f}"


def ftime(x):
    return f"{x:.2f}" if x < 1 else f"{x:.1f}" if x < 100 else f"{x:.0f}"


stats = {}
for name, J_key, t_key, status_key in METHODS:
    for c in CONFIGS:
        R = [r for r in rows if r["config"] == c]
        d = [100.0 * (num(r, J_key) - num(r, "MIQP_J_inc")) / num(r, J_key) for r in R]
        limit = status_key is not None and any(r.get(status_key) not in ("optimal", "", None) for r in R)
        stats[name, c] = (st.mean(d), max(d), st.median(num(r, t_key) for r in R), limit)
lowest = {c: min(stats[m[0], c][0] for m in METHODS) for c in CONFIGS}
table4 = []
for idx, (name, *_) in enumerate(METHODS):
    if idx in (1, 6):
        table4.append("\\midrule")
    cells = []
    for c in CONFIGS:
        mean, worst, t, limit = stats[name, c]
        m = fgap(mean)
        if c != "I" and abs(mean - lowest[c]) < 5e-4:      # bold: lowest mean gap of the configuration
            m = "\\textbf{" + m + "}"
        cells += [f"{m} ({fgap(worst)})", ftime(t) + ("$^\\dagger$" if limit else "")]
    table4.append(name + " & " + " & ".join(cells) + " \\\\")

# ---------------------------------------------------------------- Table V: results per configuration
table5 = []
for c in CONFIGS:
    R = [r for r in rows if r["config"] == c]
    opt = sum(r["MIQP_status"] == "optimal" for r in R)
    tm = [num(r, "MIQP_time") for r in R]
    tot = [num(r, "t_total") for r in R]
    d = [num(r, "gap_vs_inc_pct") for r in R]
    k = R[0]["k"]
    binaries = "--" if c == "I" else str(int(k) * 144)
    trange = "3600" if min(tm) >= 3600 else f"{ft(min(tm))}--{ft(max(tm))}"
    # bold: the method with the shorter median execution time
    minlp_best = st.median(tm) < st.median(tot)
    t_minlp = f"\\textbf{{{trange}}}" if minlp_best else trange
    t_mig = f"{st.median(tot):.1f} ({max(tot):.1f})"
    t_mig = t_mig if minlp_best else f"\\textbf{{{t_mig}}}"
    table5.append(f"{c} & {k} & {binaries} & {opt} / {10 - opt} & {t_minlp} & {max(num(r, 'MIQP_gap_pct') for r in R):.2f} & "
                  f"{t_mig} & {fx(st.median(d), 3)} ({fx(max(d), 3)})\\\\")

tex = ["% Table III", *table3, "", "% Table IV", *table4, "", "% Table V", *table5, ""]
(folder / "tables.tex").write_text("\n".join(tex))
print("\n".join(tex))
print(f"written to {folder / 'tables.tex'}")
