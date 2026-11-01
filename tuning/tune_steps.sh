#!/bin/bash
# Per-case selection of the number of continuation steps (miGoBuG only, no Gurobi licence needed).
# 1) every case is solved with schedules of 300-1400 steps (round 2: up to 2000 for cases without a run at the target)
# 2) select_steps.py keeps, per case, the fastest schedule that ends at eps_pi <= TARGET
# Output: config/per_case_steps.csv (the file shipped in config/ is the one used in the paper).
cd "$(dirname "$0")/.."
PY=${PY:-python}
TARGET=2e-5
tune() {  # $1 steps, $2 configuration, $3 cases
  out=results/pcase_L$1_$2
  $PY run_revision_benchmarks.py --exp main --configs $2 --cases $3 --migobug-only --defer-fixedqp --skip-condC \
      --start-steps $1 --max-steps $1 --out $out > $out.log 2>&1
}
mkdir -p results tuning/out
for L in 300 400 500 600 700 800 900 1000 1050 1100 1150 1200 1250 1300 1350 1400; do
  for C in I II III IV V; do tune $L $C 1-10; done
done
$PY tuning/select_steps.py pcase $TARGET tuning/out/per_case_steps.csv tuning/out/missing.txt tuning/out/selection_round1.txt
if [ -s tuning/out/missing.txt ]; then
  for L in 1450 1500 1550 1600 1650 1700 1750 1800 1850 1900 1950 2000; do
    while read C CASES; do tune $L $C $CASES; done < tuning/out/missing.txt
  done
  $PY tuning/select_steps.py pcase $TARGET tuning/out/per_case_steps.csv tuning/out/missing.txt tuning/out/selection_round2.txt
fi
echo "selected schedules: tuning/out/per_case_steps.csv"
