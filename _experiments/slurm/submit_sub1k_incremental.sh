#!/bin/bash

set -euo pipefail

: "${EXPERIMENT_CONFIG_RELATIVE:?Missing source experiment config}"
: "${RUN_PREFIX:?Missing result run prefix}"
: "${DATASET_LABEL:?Missing dataset label}"
: "${EXISTING_RESULTS_DIR:?Set EXISTING_RESULTS_DIR to the completed all-policy comparison}"
: "${TASK_LOG_PREFIX:?Missing task log prefix}"
: "${MERGE_LOG_PREFIX:?Missing merge log prefix}"

SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOURCE_PATH=$(cd "$SCRIPT_DIRECTORY/../.." && pwd)
EXPERIMENT_PATH="${EXPERIMENT_PATH:-$HOME/_experiments/echollm-sage}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
SUB1K_CACHE_SIZES="${SUB1K_CACHE_SIZES:-50:100:250:500:750}"

if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CONCURRENT must be a positive integer" >&2
  exit 2
fi
if ! [[ "$SUB1K_CACHE_SIZES" =~ ^[1-9][0-9]*(:[1-9][0-9]*)*$ ]]; then
  echo "SUB1K_CACHE_SIZES must be colon-separated positive integers." >&2
  exit 2
fi

IFS=':' read -r -a REQUESTED_CACHE_SIZES <<<"$SUB1K_CACHE_SIZES"
for size in "${REQUESTED_CACHE_SIZES[@]}"; do
  if ((size >= 1000)); then
    echo "Supplemental cache size must be below 1000; found $size." >&2
    exit 2
  fi
done

if [[ ! -d "$EXISTING_RESULTS_DIR" ]]; then
  echo "Completed comparison not found: $EXISTING_RESULTS_DIR" >&2
  exit 3
fi
EXISTING_RESULTS_DIR=$(cd "$EXISTING_RESULTS_DIR" && pwd)
for required in summary.csv dataset_stats.json; do
  if [[ ! -f "$EXISTING_RESULTS_DIR/$required" ]]; then
    echo "Completed comparison is missing $EXISTING_RESULTS_DIR/$required" >&2
    exit 3
  fi
done

PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/py313/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=$(command -v python3 || true)
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python was not found; set PYTHON_BIN to the py313 interpreter." >&2
  exit 3
fi

cd "$SOURCE_PATH"
echo "Validating the supplemental capacity grid..."
VALIDATION_OUTPUT=$(
  "$PYTHON_BIN" -m _experiments.merge_results validate-capacities \
    --existing-dir "$EXISTING_RESULTS_DIR" \
    --source-config "$SOURCE_PATH/$EXPERIMENT_CONFIG_RELATIVE" \
    --cache-sizes "${REQUESTED_CACHE_SIZES[@]}"
)
echo "$VALIDATION_OUTPUT"

EXPECTED_TASKS=$(
  printf '%s\n' "$VALIDATION_OUTPUT" |
    sed -n 's/^EXPECTED_TASKS=//p' |
    tail -n 1
)
CANONICAL_CACHE_SIZES=$(
  printf '%s\n' "$VALIDATION_OUTPUT" |
    sed -n 's/^CACHE_SIZES=//p' |
    tail -n 1
)
if ! [[ "$EXPECTED_TASKS" =~ ^[1-9][0-9]*$ ]] || \
   [[ -z "$CANONICAL_CACHE_SIZES" ]]; then
  echo "Preflight did not return a valid replay plan." >&2
  exit 3
fi

ARTIFACT_RESULTS_DIR="$EXISTING_RESULTS_DIR"
export ARTIFACT_RESULTS_DIR EXPERIMENT_PATH PYTHON_BIN
source "$SCRIPT_DIRECTORY/resolve_replay_artifacts.sh"

LAST_INDEX=$((EXPECTED_TASKS - 1))
CACHE_SIZE_EXPORT="${CANONICAL_CACHE_SIZES//,/:}"

echo "Submitting $EXPECTED_TASKS CPU replay tasks for $DATASET_LABEL..."
echo "Supplemental capacities: $CANONICAL_CACHE_SIZES"
ARRAY_JOB_ID=$(sbatch \
  --parsable \
  --array="0-${LAST_INDEX}%${MAX_CONCURRENT}" \
  --job-name="$TASK_LOG_PREFIX" \
  --output="$SOURCE_PATH/${TASK_LOG_PREFIX}-%A_%a.out" \
  --export="ALL,SOURCE_PATH=$SOURCE_PATH,EXPERIMENT_PATH=$EXPERIMENT_PATH,EXPERIMENT_CONFIG_RELATIVE=$EXPERIMENT_CONFIG_RELATIVE,RUN_PREFIX=$RUN_PREFIX,DATASET_LABEL=$DATASET_LABEL,RECORDED_LLM_PATH=$RECORDED_LLM_PATH,PREPARED_PAIRS_PATH=$PREPARED_PAIRS_PATH,EMBEDDING_CACHE_PATH=$EMBEDDING_CACHE_PATH,EMBEDDING_DEVICE=cpu,INSTALL_REQUIREMENTS=0,POSITIVE_CACHE_SIZES_ONLY=1,CACHE_SIZE_FILTER=$CACHE_SIZE_EXPORT" \
  "$SOURCE_PATH/_experiments/slurm/replay_experiment.sbatch")
ARRAY_JOB_ID="${ARRAY_JOB_ID%%;*}"

MERGE_JOB_ID=$(sbatch \
  --parsable \
  --dependency="afterok:$ARRAY_JOB_ID" \
  --output="$SOURCE_PATH/${MERGE_LOG_PREFIX}-%j.out" \
  --export="ALL,SOURCE_PATH=$SOURCE_PATH,EXPERIMENT_PATH=$EXPERIMENT_PATH,EXPERIMENT_ARRAY_JOB_ID=$ARRAY_JOB_ID,RUN_PREFIX=$RUN_PREFIX,EXPECTED_TASKS=$EXPECTED_TASKS,EXISTING_RESULTS_DIR=$EXISTING_RESULTS_DIR" \
  "$SOURCE_PATH/_experiments/slurm/aggregate_incremental_capacities.sbatch")
MERGE_JOB_ID="${MERGE_JOB_ID%%;*}"

RESULT_ROOT="$EXPERIMENT_PATH/results/$RUN_PREFIX-$ARRAY_JOB_ID"
echo "Replay array job: $ARRAY_JOB_ID"
echo "Merge/plot job: $MERGE_JOB_ID (starts only after every replay succeeds)"
echo "Monitor: squeue -j $ARRAY_JOB_ID,$MERGE_JOB_ID"
echo "Accounting: sacct -j $ARRAY_JOB_ID --format=JobID,State,Elapsed,Start,End"
echo "First task log: tail -f $SOURCE_PATH/${TASK_LOG_PREFIX}-${ARRAY_JOB_ID}_0.out"
echo "Merge log: tail -f $SOURCE_PATH/${MERGE_LOG_PREFIX}-${MERGE_JOB_ID}.out"
echo "Supplemental-only results: $RESULT_ROOT/sub1k-only"
echo "Merged summary and plots: $RESULT_ROOT/comparison"
echo "Cancel: scancel $ARRAY_JOB_ID $MERGE_JOB_ID"
