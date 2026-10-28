# miGoBuG: Mixed-Integer Alternative Pontryagin's Minimum Principle Benchmark for Microgrid Energy Management

Data, configuration, and results of the paper

> N.-S. Luong, L. Tran, T. Vo-Duy, M. C. Ta, J. P. F. Trovão, and B.-H. Nguyễn,
> "Mixed-Integer Alternative Pontryagin's Minimum Principle (mi-Alt-PMP) for Global Optimal
> Benchmark of Microgrid Energy Management (miGoBuG)," *IEEE Transactions on Smart Grid*
> (under review, manuscript TSG-01638-2026).

The source code of miGoBuG and of the MINLP and MILP benchmarks will be added to this repository.

## Contents

| Path | Content |
|---|---|
| `data/load.csv` | net-load profile of the test system, 31 days at 30-min resolution |
| `config/config_used.json` | parameters of all five test systems and of the solvers |
| `config/environment_used.json` | software environment of the reported run |
| `config/per_case_steps.csv` | number of continuation steps of every case (column `steps`) |
| `reference_results/main.csv` | results reported in the paper, one row per configuration and dataset |
| `reference_results/traj/` | trajectories of miGoBuG and of the MINLP (`.npz`) |
| `reference_results/threshold_condC.csv`, `threshold_summary.json` | regularization threshold of Appendix A along the continuation |
| `reference_results/data_profiles.csv`, `data_stats.csv` | net-load profile per interval and statistics per dataset |

## Data

`data/load.csv` holds the net-load profile $P_{ld}$ in kW, i.e., the load that the storage units,
the diesel generator, and the grid connections must cover:

| Column | Content |
|---|---|
| `day` | day 1 to 31 |
| `step` | 30-min interval within the day |
| `p_net` | net load [kW] |

Every dataset covers three consecutive days (144 intervals); the 31 days give ten datasets.
Configurations I to V use zero to four grid connection nodes (1 to 16 admissible topologies); their
parameters are listed in `config/config_used.json`.

## Results

`reference_results/main.csv` contains, for every configuration and dataset, the objective, status,
best bound, MIP gap, and runtime of the MINLP solved by Gurobi, the objective and runtime of
miGoBuG with its switching sequence evaluated in the same benchmark model (`J_miGoBuG_ZOH`), and
the MILP-PWL benchmarks with $L = 8, 16, 32$ segments.

## Citation

Please cite the paper above if you use these data.
