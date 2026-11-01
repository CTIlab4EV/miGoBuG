# miGoBuG: Mixed-Integer Alternative Pontryagin's Minimum Principle Benchmark for Microgrid Energy Management

Code, data, and configuration for the paper

> N.-S. Luong, L. Tran, T. Vo-Duy, M. C. Ta, J. P. F. Trovão, and B.-H. Nguyễn,
> "Mixed-Integer Alternative Pontryagin's Minimum Principle (mi-Alt-PMP) for Global Optimal
> Benchmark of Microgrid Energy Management (miGoBuG)," *IEEE Transactions on Smart Grid*
> (under review, manuscript TSG-01638-2026).

The package contains three methods:

- **miGoBuG**: the mi-Alt-PMP solver with entropy-regularized homotopy;
- **MINLP**: the benchmark model solved by Gurobi, a convex mixed-integer quadratic program in its
  zero-order-hold (ZOH) transcription;
- **MILP-PWL**: the same model with $L = 8, 16, 32$ piecewise-linear cost segments, with and
  without re-optimizing the dispatch of its switching sequence in the MINLP ("+ QP").

It also contains the net-load profile, the configuration of all five test systems, the
continuation schedule of every case, and the results reported in the paper.

## Contents

| Path | Content |
|---|---|
| `miGoBuG.py` | mi-Alt-PMP solver (Numba kernels, TPBVP collocation); does not use Gurobi |
| `miGoBuG_revision_patch.py` | benchmark model (ZOH transcription), Gurobi MINLP/MILP, fixed-sequence QP, continuation loop, diagnostics, regularization threshold of Appendix A |
| `run_revision_benchmarks.py` | driver for all experiments; all parameters are in `CFG` at the top |
| `run_gurobi_cases.py` | MINLP for selected cases (e.g., in a separate process) |
| `run_pwl_cases.py` | MILP-PWL benchmarks ($L = 8, 16, 32$) for selected configurations |
| `make_tables.py`, `make_figures.py` | tables and figures of the paper from one results folder |
| `threshold_summary.py` | summary of the regularization threshold $\bar\epsilon_\pi$ (Appendix A) |
| `tuning/` | per-case selection of the continuation schedule |
| `config/per_case_steps.csv` | number of continuation steps of every case, as used in the paper |
| `config/config_used.json`, `config/environment_used.json` | full parameter set and software environment of the reported run |
| `data/load.csv` | net-load profile of the test system, 31 days at 30-min resolution |
| `reference_results/` | results reported in the paper (`main.csv`), trajectories (`traj/`), threshold data |

## Installation

Python 3.10 or newer:

```bash
pip install -r requirements.txt
```

**Gurobi licence.** No licence is included. miGoBuG itself (the continuous-time solver) runs
without Gurobi. A Gurobi licence is needed for the MINLP and MILP benchmarks and for the
fixed-sequence QP that evaluates every switching sequence in the benchmark
model; the size-limited licence shipped with `pip install gurobipy` is too small for these
models. Free academic licences are available from Gurobi. Either install a `gurobi.lic`, or set a
Web License Service licence in the environment:

```bash
export GRB_WLSACCESSID=...  GRB_WLSSECRET=...  GRB_LICENSEID=...
```

`make_figures.py` renders text with LaTeX (`text.usetex`), so a LaTeX installation is
needed for the figures.

## Quick start

miGoBuG on one case, without Gurobi (Configuration III, dataset 3):

```bash
python run_revision_benchmarks.py --exp main --configs III --cases 3 \
    --migobug-only --defer-fixedqp --skip-condC \
    --step-pref config/per_case_steps.csv --out results/quickstart
```

The continuous-time solution is written to `results/quickstart/main.csv` and
`results/quickstart/traj/`. Without `--defer-fixedqp` (Gurobi needed), the switching sequence is
also evaluated in the benchmark model, which gives `J_miGoBuG_ZOH`.

Tables and figures from the reference results:

```bash
python make_tables.py reference_results
python make_figures.py reference_results
python threshold_summary.py reference_results/threshold_condC.csv
```

## Reproducing the paper

1. **Main benchmark** (MINLP with a 3600 s limit, MILP-PWL for Configuration III, miGoBuG; Gurobi needed). With the
   time limit, the 24 instances that reach it take about 24 h in total:

   ```bash
   python run_revision_benchmarks.py --exp main --step-pref config/per_case_steps.csv \
       --skip-condC --out results/main
   ```

   The run can be stopped and restarted with the same `--out`; finished rows are kept.
   `--miqp-limit` shortens the Gurobi time limit for a quick check.

2. **MILP-PWL** for the remaining configurations (Configuration III is solved in step 1), then
   merged into `results/main/main.csv`:

   ```bash
   python run_pwl_cases.py --configs I II IV V --results results/main --out results/pwl_all
   python run_pwl_cases.py --results results/main --out results/pwl_all --merge
   ```

3. **Regularization threshold** of Appendix A along the continuation of every case:

   ```bash
   python run_revision_benchmarks.py --exp main --migobug-only --defer-fixedqp --certify-condC \
       --step-pref config/per_case_steps.csv --out results/threshold
   python threshold_summary.py results/threshold/condC.csv
   ```

4. **Tables and figures**:

   ```bash
   python make_tables.py results/main
   python make_figures.py results/main
   ```

Gurobi's runtimes and, for instances stopped at the time limit, its incumbents and bounds depend
on the machine and the Gurobi version; the reference results were obtained with Gurobi 13.0.2
(`config/environment_used.json`).

## Continuation schedule

The continuation runs from $\epsilon_\pi = 10^9$ downwards on a geometric schedule. The number of
continuation steps is selected per case; the schedules used in the paper are listed in
`config/per_case_steps.csv` (column `steps`), and `--step-pref` makes the driver use them. The
reported execution time refers to the selected schedule. `tuning/tune_steps.sh` regenerates the
selection (miGoBuG only, no Gurobi needed).

## Test system and data

`data/load.csv` holds the net-load profile $P_{ld}$ of the test system in kW, i.e., the load that
the storage units, the diesel generator, and the grid connections must cover:

| Column | Content |
|---|---|
| `day` | day 1 to 31 |
| `step` | 30-min interval within the day |
| `p_net` | net load [kW] |

Every case covers three consecutive days (144 intervals); the 31 days give ten cases. The
parameters of the storage units, the diesel generator, and grid nodes G1 to G4 of
Configurations II to V are defined in `CFG` of `run_revision_benchmarks.py`;
`config/config_used.json` lists every parameter of the reported run.

## Reference results

`reference_results/main.csv` contains, for every configuration and dataset, the objective, status,
best bound, MIP gap, and runtime of the MINLP solved by Gurobi, the objective and runtime of
miGoBuG with its switching sequence evaluated in the same benchmark model (`J_miGoBuG_ZOH`), and
the MILP-PWL benchmarks with $L = 8, 16, 32$ segments. `reference_results/traj/` holds the
trajectories of miGoBuG and of the MINLP, and `threshold_condC.csv` the regularization threshold
of Appendix A along the continuation.

## Output

Every run folder contains `main.csv` (one row per case: solver status, objective, best bound, MIP
gap, runtimes, and diagnostics of every method), `config.json`, `environment.json`, the Gurobi
logs (`gurobi_logs/`), the incumbent/bound progress of every Gurobi run (`progress/`), the
continuation logs (`conv/`), and the trajectories (`traj/`).

## License

The code and data in this repository are released under the MIT License (see `LICENSE`).

## Citation

Please cite the paper above if you use this code or data.
