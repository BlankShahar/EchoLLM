#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --job-name=echollm-sub1k-merge
#SBATCH --output=echollm-sub1k-merge-%j.out
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail

: "${EXPERIMENT_ARRAY_JOB_ID:?Missing source array job ID}"
: "${RUN_PREFIX:?Missing result run prefix}"
: "${EXPECTED_TASKS:?Missing expected task count}"
: "${EXISTING_RESULTS_DIR:?Missing completed all-policy comparison}"

EXPERIMENT_PATH="${EXPERIMENT_PATH:-$HOME/_experiments/echollm-sage}"
SOURCE_PATH="${SOURCE_PATH:-${SLURM_SUBMIT_DIR:-$PWD}}"
RESULT_GROUP="$RUN_PREFIX-$EXPERIMENT_ARRAY_JOB_ID"
RESULT_ROOT="$EXPERIMENT_PATH/results/$RESULT_GROUP"
TASKS_ROOT="$RESULT_ROOT/tasks"
SUPPLEMENTAL_DIRECTORY="$RESULT_ROOT/sub1k-only"
COMPARISON_DIRECTORY="$RESULT_ROOT/comparison"

export MPLBACKEND=Agg

echo "Sub-1K aggregation started at $(date --iso-8601=seconds)"
echo "Existing comparison: $EXISTING_RESULTS_DIR"
echo "Tasks: $TASKS_ROOT"
echo "Merged output: $COMPARISON_DIRECTORY"

eval "$(/storage/modules/packages/anaconda/bin/conda shell.bash hook)"
conda activate py313
cd "$SOURCE_PATH"

TASK_CONFIG=$(
  find "$TASKS_ROOT" -mindepth 2 -maxdepth 2 -name config.json \
    -type f -print | sort | head -n 1
)
if [[ -z "$TASK_CONFIG" ]]; then
  echo "No completed supplemental task config found under $TASKS_ROOT" >&2
  exit 3
fi

echo "Aggregating $EXPECTED_TASKS all-policy supplemental tasks..."
python -m _experiments.aggregate \
  --tasks-root "$TASKS_ROOT" \
  --output-dir "$SUPPLEMENTAL_DIRECTORY" \
  --expected-tasks "$EXPECTED_TASKS" \
  --config "$TASK_CONFIG"

echo "Validating and merging the supplemental capacities..."
python -m _experiments.merge_results merge-capacities \
  --existing-dir "$EXISTING_RESULTS_DIR" \
  --supplemental-dir "$SUPPLEMENTAL_DIRECTORY" \
  --output-dir "$COMPARISON_DIRECTORY"

echo "Supplemental-only results: $SUPPLEMENTAL_DIRECTORY"
echo "Merged all-policy summary and plots: $COMPARISON_DIRECTORY"
echo "Sub-1K aggregation completed at $(date --iso-8601=seconds)"
