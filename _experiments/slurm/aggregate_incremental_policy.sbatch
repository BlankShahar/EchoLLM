#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --job-name=echollm-sparq-merge
#SBATCH --output=echollm-sparq-merge-%j.out
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail

: "${EXPERIMENT_ARRAY_JOB_ID:?Missing source array job ID}"
: "${RUN_PREFIX:?Missing result run prefix}"
: "${EXPECTED_TASKS:?Missing expected task count}"
: "${BASELINE_RESULTS_DIR:?Missing baseline results directory}"

EXPERIMENT_PATH="${EXPERIMENT_PATH:-$HOME/_experiments/echollm-sage}"
SOURCE_PATH="${SOURCE_PATH:-${SLURM_SUBMIT_DIR:-$PWD}}"
RESULT_GROUP="$RUN_PREFIX-$EXPERIMENT_ARRAY_JOB_ID"
RESULT_ROOT="$EXPERIMENT_PATH/results/$RESULT_GROUP"
TASKS_ROOT="$RESULT_ROOT/tasks"
INCREMENTAL_DIRECTORY="$RESULT_ROOT/sparq-only"
COMPARISON_DIRECTORY="$RESULT_ROOT/comparison"

export MPLBACKEND=Agg

echo "SPARQ incremental aggregation started at $(date --iso-8601=seconds)"
echo "Baseline: $BASELINE_RESULTS_DIR"
echo "Tasks: $TASKS_ROOT"
echo "Comparison output: $COMPARISON_DIRECTORY"

eval "$(/storage/modules/packages/anaconda/bin/conda shell.bash hook)"
conda activate py313
cd "$SOURCE_PATH"

TASK_CONFIG=$(
  find "$TASKS_ROOT" -mindepth 2 -maxdepth 2 -name config.json \
    -type f -print | sort | head -n 1
)
if [[ -z "$TASK_CONFIG" ]]; then
  echo "No completed SPARQ task config found under $TASKS_ROOT" >&2
  exit 3
fi

echo "Aggregating $EXPECTED_TASKS SPARQ tasks..."
python -m _experiments.aggregate \
  --tasks-root "$TASKS_ROOT" \
  --output-dir "$INCREMENTAL_DIRECTORY" \
  --expected-tasks "$EXPECTED_TASKS" \
  --config "$TASK_CONFIG"

echo "Validating and merging with the completed baseline..."
python -m _experiments.merge_results merge \
  --baseline-dir "$BASELINE_RESULTS_DIR" \
  --incremental-dir "$INCREMENTAL_DIRECTORY" \
  --output-dir "$COMPARISON_DIRECTORY" \
  --policy SPARQ

echo "SPARQ-only results: $INCREMENTAL_DIRECTORY"
echo "All-policy summary and plots: $COMPARISON_DIRECTORY"
echo "Incremental aggregation completed at $(date --iso-8601=seconds)"
