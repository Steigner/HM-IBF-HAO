#!/usr/bin/env bash
#
# Full-pipeline smoke test: preprocess -> train -> evaluate, end to end.
#
# Runs inside the container built from `Dockerfile`; use `./run.sh smoke` from the host.
# Training shells out to R/IRACE, so this needs the Nix-enabled image:
#
#   HM_IBF_NIX=1 ./run.sh smoke               # bash
#   $env:HM_IBF_NIX = "1"; .\run.bat smoke    # PowerShell
#
# It proves the three stages are wired together and that each one produces the artefacts
# the next one reads. It is NOT a real experiment: the budgets are tiny and the instances
# are generated from synthetic terrain, so the objective values it reports mean nothing.
#
# Everything is written below a scratch directory outside the repository, so nothing it
# produces can be committed. Pass a directory as the first argument to keep the output.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SMOKE="${1:-$(mktemp -d /tmp/hm-ibf-smoke.XXXXXX)}"
KEEP="${1:+yes}"

step() {
    echo
    echo "==> $*"
}

fail() {
    echo "SMOKE FAILED: $*" >&2
    exit 1
}

require_file() {
    [ -f "$1" ] || fail "expected $1 to exist"
    echo "    ok  $1"
}

mkdir -p "$SMOKE"
echo "==> Smoke directory: $SMOKE"
cd "$REPO_ROOT/hm-ibf-hao"

# --------------------------------------------------------------------------- #
# 1. Preprocess                                                               #
# --------------------------------------------------------------------------- #
# Synthetic terrain rather than a downloaded benchmark map: the smoke test must stay
# offline and finish in seconds. `--maps`/benchmark mode is exercised by the pytest suite.
step "1/4 Generating synthetic terrain"
python3 - "$SMOKE/terrain.npy" <<'PY'
"""Write a small synthetic heightmap with a ridge, for the smoke pipeline."""

import sys

import numpy as np

rows, columns = 60, 90
row_grid, column_grid = np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij")
ridge = 400.0 * np.exp(-(((row_grid - rows / 2) / 6.0) ** 2))
np.save(sys.argv[1], (ridge + 3.0 * column_grid).astype(np.float64))
PY
require_file "$SMOKE/terrain.npy"

step "2/4 Preprocess: generating a train/eval instance split"
python3 -m preprocessing.prepare_instances \
    --source "$SMOKE/terrain.npy" \
    --train-dir "$SMOKE/instances_train" \
    --eval-dir "$SMOKE/instances_eval" \
    --n-train 2 --n-eval 2 \
    --resolution 60 90 \
    --min-distance-fraction 0.3 \
    --oversample-factor 3 \
    --n-buckets 2 \
    --n-dimensions-allowed 2 \
    --backbone-step 2.0 \
    --seed 7
require_file "$SMOKE/instances_train/summary.json"
require_file "$SMOKE/instances_eval/summary.json"

# The generator's `dimensions_allowed` recommendation is what the training configuration
# has to carry, so it is read back from the summary rather than hardcoded here.
step "3/4 Writing a minimal tuning configuration"
python3 - "$SMOKE/instances_train/summary.json" "$SMOKE" <<'PY'
"""Render the smoke run's params_*.conf from the generated set's recommendation."""

import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
smoke = Path(sys.argv[2])
dimensions = ", ".join(str(value) for value in summary["dimensions_allowed"])

(smoke / "params_training.conf").write_text(
    f"""epsilon = 1e-8
max_evaluations = 200
num_repetitions = 1
num_tuning_repetitions = 1
num_tuning_experiments = 10
num_iterations = 1
max_island_iterations = 5
max_island_population = 8
dimensions_allowed = [{dimensions}]

[grahf]
max_initial_nodes = 3
initial_edge_p = 0.3
population_size = 5
tournament_size = 2
archive_size = 1
elitist_freq = 2
pc = 0.6
rm_node = 0.12
rm_edge = 0.15
rm_node_weight = 0.08
rm_edge_weight = 0.2
""",
    encoding="utf-8",
)
(smoke / "params_evaluation.conf").write_text(
    "max_iterations = 50\nmax_population_size = 50\nbest_value_tolerance = 1e-9\n",
    encoding="utf-8",
)
print(f"dimensions_allowed = [{dimensions}]")
PY
require_file "$SMOKE/params_training.conf"
require_file "$SMOKE/params_evaluation.conf"

# --------------------------------------------------------------------------- #
# 2. Train and evaluate                                                       #
# --------------------------------------------------------------------------- #
if ! command -v hm-ibf >/dev/null 2>&1; then
    fail "hm-ibf is not on PATH; run this inside the container (./run.sh smoke)"
fi
if [ -z "${IN_NIX_SHELL:-}" ] && ! command -v nix >/dev/null 2>&1; then
    fail "training needs R/IRACE from the Nix image; use HM_IBF_NIX=1 ./run.sh smoke"
fi

step "4/4 Pipeline: train on the split, then evaluate the elitist on the held-out set"
hm-ibf pipeline \
    --instances-dir "$SMOKE/instances_train" \
    --eval-instances-dir "$SMOKE/instances_eval" \
    --run-dir "$SMOKE/hao_run" \
    --training-params "$SMOKE/params_training.conf" \
    --evaluation-params "$SMOKE/params_evaluation.conf" \
    --seed 1 \
    --experiments-dir "$SMOKE/experiments_hao" \
    --results-dir "$SMOKE/results" \
    --summary-csv "$SMOKE/results/results_hao.csv" \
    --first-seed 1 --num-seeds 1

step "Checking the artefacts"
require_file "$SMOKE/hao_run/elitist_0.json"
require_file "$SMOKE/hao_run/elitist_0.params"
require_file "$SMOKE/hao_run/statistics.json"
require_file "$SMOKE/results/results_hao.csv"

runs=$(find "$SMOKE/results" -mindepth 1 -maxdepth 1 -type d | wc -l)
[ "$runs" -ge 1 ] || fail "no per-run result folder under $SMOKE/results"
for run in "$SMOKE"/results/*/; do
    require_file "${run}results.json"
    require_file "${run}best_value.csv"
    require_file "${run}avg_value.csv"
done

echo
echo "==> Smoke test passed."
if [ -n "$KEEP" ]; then
    echo "    Output kept in $SMOKE"
else
    rm -rf "$SMOKE"
    echo "    Removed $SMOKE (pass a directory to keep it)"
fi
