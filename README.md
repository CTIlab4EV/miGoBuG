# miGoBuG

Mixed-Integer Alternative Pontryagin's Minimum Principle (mi-Alt-PMP) benchmark for microgrid energy management: code, data, and results of the paper

> N.-S. Luong, L. Tran, T. Vo-Duy, M. C. Ta, J. P. F. Trovão, and B.-H. Nguyễn,
> "Mixed-Integer Alternative Pontryagin's Minimum Principle (mi-Alt-PMP) for Global Optimal
> Benchmark of Microgrid Energy Management (miGoBuG)," *IEEE Transactions on Smart Grid*
> (under review, manuscript TSG-01638-2026).

All methods are scored in one benchmark model, a convex mixed-integer quadratic program in zero-order-hold (ZOH) form:

| Method | Description |
|---|---|
| **miGoBuG** | mi-Alt-PMP solver with entropy-regularized homotopy |
| **MINLP** | the benchmark model solved by Gurobi |
| **MILP-PWL** | the benchmark model with $L = 8, 16, 32$ piecewise-linear cost segments, with and without re-optimizing the dispatch of its switching sequence ("+ QP") |
| **Heuristic baselines** | MS MINTOC, SUR NLP, and Adjoint NLP |

## Contents

| Path | Content |
|---|---|
| `miGoBuG.py` | mi-Alt-PMP solver core (Numba kernels); does not use Gurobi |
| `benchmark.py` | benchmark model, Gurobi MINLP and MILP, fixed-sequence QP, continuation, regularization threshold of Appendix A |
| `baselines.py` | heuristic baselines |
| `run_benchmarks.py` | runs the benchmark; all parameters are in `CFG` at the top |
| `run_pwl_cases.py` | MILP-PWL for selected configurations |
| `make_tables.py`, `make_figures.py`, `threshold_summary.py` | Tables III–V, Figs. 3–7, and the threshold of Appendix A from a results folder |
| `tuning/` | per-case selection of the continuation schedule |
| `config/` | steps of every case, parameters and software of the reported run |
| `data/load.csv` | net-load profile of the test system |
| `reference_results/` | results reported in the paper |

## Installation

Python 3.10 or newer:

```bash
pip install -r requirements.txt
```

`make_figures.py` also needs a LaTeX installation.

**Gurobi licence.** No licence is included. miGoBuG itself runs without one. The MINLP and MILP benchmarks, the heuristic baselines, and the fixed-sequence QP that scores every switching sequence need a licence, because the size-limited one shipped with `pip install gurobipy` is too small for these models. Free academic licences are available from Gurobi. Either install a `gurobi.lic`, or set a Web License Service licence:

```bash
export GRB_WLSACCESSID=...  GRB_WLSSECRET=...  GRB_LICENSEID=...
```

## Quick start

miGoBuG on one case, without a Gurobi licence (Configuration III, dataset 3):

```bash
python run_benchmarks.py --configs III --cases 3 --migobug-only --defer-fixedqp \
    --step-pref config/per_case_steps.csv --out results/quickstart
```

Tables and figures of the paper from the reference results:

```bash
python make_tables.py reference_results
python make_figures.py reference_results
python threshold_summary.py reference_results/threshold.csv
```

## Reproducing the paper

1. **Main benchmark** (miGoBuG, MINLP with a 3600 s limit, MILP-PWL for Configuration III, and the heuristic baselines). The 24 MINLP instances that reach the limit take about 24 h in total. A run can be stopped and restarted with the same `--out`. `--miqp-limit` shortens the limit for a quick check, and `--no-baselines` skips the heuristic baselines.

   ```bash
   python run_benchmarks.py --step-pref config/per_case_steps.csv --out results/main
   ```

2. **MILP-PWL** for the other configurations, merged into `results/main/main.csv`:

   ```bash
   python run_pwl_cases.py --configs I II IV V --results results/main --out results/pwl_all
   python run_pwl_cases.py --results results/main --out results/pwl_all --merge
   ```

3. **Regularization threshold** of Appendix A along the continuation of every case:

   ```bash
   python run_benchmarks.py --migobug-only --defer-fixedqp --threshold \
       --step-pref config/per_case_steps.csv --out results/threshold
   python threshold_summary.py results/threshold/threshold.csv
   ```

4. **Tables and figures**:

   ```bash
   python make_tables.py results/main
   python make_figures.py results/main
   ```

Gurobi's runtimes and, for instances stopped at the time limit, its incumbents and bounds depend on the machine and the Gurobi version. The reference results were obtained with Gurobi 13.0.2 (`config/environment_used.json`).

## Continuation schedule

The continuation runs from $\epsilon_\pi = 10^9$ downwards on a geometric schedule. The number of continuation steps is selected per case: `config/per_case_steps.csv` (column `steps`) lists the schedules used in the paper, `--step-pref` applies them, and the reported execution time refers to the selected schedule. `tuning/tune_steps.sh` regenerates the selection, without a Gurobi licence.

## Data

`data/load.csv` holds the net-load profile $P_{ld}$ of the test system in kW, i.e., the load that the storage units, the diesel generator, and the grid connections must cover. Columns: `day`, `step` (30-min interval within the day), and `p_net`.

Every case covers three consecutive days (144 intervals), which gives ten cases from days 1 to 30. All parameters are in `CFG` of `run_benchmarks.py`, and `config/config_used.json` lists the values of the reported run.

## Results

Every run folder contains:

| File | Content |
|---|---|
| `main.csv` | one row per case, see below |
| `config.json`, `environment.json` | parameters and software |
| `traj/` | trajectories of miGoBuG (`*_miGoBuG.npz`) and of the MINLP (`*_MIQP.npz`) |
| `conv/` | continuation log of every case |
| `gurobi_logs/`, `progress/` | Gurobi logs, incumbent and bound over time |
| `threshold.csv` | threshold of Appendix A, with `--threshold` |

Main columns of `main.csv`:

| Column | Content |
|---|---|
| `J_miGoBuG_ZOH`, `t_total` | objective of the miGoBuG sequence in the benchmark model, execution time |
| `gap_vs_inc_pct` | $100\,(J - J_{\mathrm{MINLP}})/J$ of miGoBuG ZOH |
| `gap_certified_pct` | the same gap with respect to the MINLP best bound |
| `MIQP_status`, `MIQP_time`, `MIQP_J_inc`, `MIQP_J_LB`, `MIQP_gap_pct` | MINLP status, runtime, incumbent, best bound, final MIP gap |
| `MIQP_reused_from` | earlier run with the same model and settings that the MINLP result was taken from |
| `PWL<L>_J_true`, `PWL<L>_J_seqQP`, `PWL<L>_time`, `PWL<L>_status` | MILP-PWL objective, without and with "+ QP", runtime, status |
| `MINTOC_J`, `SUR_J`, `Adjoint_J`, `*_time` | heuristic baselines |
| `bound_consistent` | the MINLP bound does not exceed the cost of any feasible solution found by any method; a necessary condition for a valid bound, not a proof |
| `eps_pi_last`, `steps_used`, `failed_steps` | continuation of miGoBuG |

## License

MIT License, see `LICENSE`.

## Citation

Please cite the paper above if you use this code or data.
