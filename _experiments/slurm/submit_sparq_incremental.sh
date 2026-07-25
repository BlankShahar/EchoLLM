#!/bin/bash

set -euo pipefail

: "${EXPERIMENT_CONFIG_RELATIVE:?Missing source experiment config}"
: "${RUN_PREFIX:?Missing result run prefix}"
: "${DATASET_LABEL:?Missing dataset label}"
: "${BASELINE_RESULTS_DIR:?Set BASELINE_RESULTS_DIR to the completed baseline run}"
: "${TASK_LOG_PREFIX:?Missing task log prefix}"
: "${MERGE_LOG_PREFIX:?Missing merge log prefix}"

SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOURCE_PATH=$(cd "$SCRIPT_DIRECTORY/../.." && pwd)
EXPERIMENT_PATH="${EXPERIMENT_PATH:-$HOME/_experiments/echollm-sage}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"

if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CONCURRENT must be a positive integer" >&2
  exit 2
fi

if [[ ! -d "$BASELINE_RESULTS_DIR" ]]; then
  echo "Baseline result directory not found: $BASELINE_RESULTS_DIR" >&2
  exit 3
fi
BASELINE_RESULTS_DIR=$(cd "$BASELINE_RESULTS_DIR" && pwd)

for required in summary.csv dataset_stats.json experiment_config.yaml; do
  if [[ ! -f "$BASELINE_RESULTS_DIR/$required" ]]; then
    echo "Baseline is incomplete; missing $BASELINE_RESULTS_DIR/$required" >&2
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

TASK_CONFIG="${BASELINE_TASK_CONFIG:-}"
if [[ -z "$TASK_CONFIG" ]]; then
  TASK_CONFIG=$(
    find "$BASELINE_RESULTS_DIR/tasks" -mindepth 2 -maxdepth 2 \
      -name config.json -type f -print 2>/dev/null | sort | head -n 1 || true
  )
fi

json_config_value() {
  "$PYTHON_BIN" -c \
    'import json,sys; value=json.load(open(sys.argv[1]))[sys.argv[2]].get(sys.argv[3]); print(value or "")' \
    "$1" "$2" "$3"
}

RECORDED_LLM_PATH="${RECORDED_LLM_PATH:-}"
EMBEDDING_CACHE_PATH="${EMBEDDING_CACHE_PATH:-}"
if [[ -n "$TASK_CONFIG" ]]; then
  RECORDED_LLM_PATH="${RECORDED_LLM_PATH:-$(json_config_value "$TASK_CONFIG" llm recorded_path)}"
  EMBEDDING_CACHE_PATH="${EMBEDDING_CACHE_PATH:-$(json_config_value "$TASK_CONFIG" embedding cache_path)}"
fi

PREPARED_PAIRS_PATH="${PREPARED_PAIRS_PATH:-}"
if [[ -z "$PREPARED_PAIRS_PATH" ]]; then
  BASELINE_GROUP=$(basename "$BASELINE_RESULTS_DIR")
  ARTIFACT_PREFIX="${ARTIFACT_PREFIX:-${BASELINE_GROUP%-*}}"
  PREPARED_PAIRS_PATH=$(
    find "$EXPERIMENT_PATH/prepared" -maxdepth 1 -type f \
      -name "${ARTIFACT_PREFIX}-*.jsonl.gz" -printf '%T@ %p\n' 2>/dev/null |
      sort -nr | cut -d' ' -f2- | head -n 1 || true
  )
fi

if [[ ! -f "$RECORDED_LLM_PATH" ]]; then
  echo "Recorded LLM database not found: ${RECORDED_LLM_PATH:-<unset>}" >&2
  echo "Set RECORDED_LLM_PATH to the recording used by the baseline." >&2
  exit 3
fi
if [[ ! -f "$PREPARED_PAIRS_PATH" ]]; then
  echo "Prepared trace not found: ${PREPARED_PAIRS_PATH:-<unset>}" >&2
  echo "Set PREPARED_PAIRS_PATH to the artifact used by the baseline." >&2
  exit 3
fi
if [[ ! -f "$EMBEDDING_CACHE_PATH" ]]; then
  echo "Embedding cache not found: ${EMBEDDING_CACHE_PATH:-<unset>}" >&2
  echo "Set EMBEDDING_CACHE_PATH to the cache used by the baseline." >&2
  exit 3
fi

cd "$SOURCE_PATH"
echo "Validating the baseline before submitting jobs..."
VALIDATION_OUTPUT=$("$PYTHON_BIN" -m _experiments.merge_results validate \
  --baseline-dir "$BASELINE_RESULTS_DIR" \
  --source-config "$SOURCE_PATH/$EXPERIMENT_CONFIG_RELATIVE" \
  --policy SPARQ)
echo "$VALIDATION_OUTPUT"

CACHE_SIZE_FILTER=$(
  printf '%s\n' "$VALIDATION_OUTPUT" |
    sed -n 's/^CACHE_SIZES=//p' |
    tail -n 1
)
if [[ -z "$CACHE_SIZE_FILTER" ]]; then
  echo "Preflight did not return a bounded cache-size plan." >&2
  exit 3
fi
IFS=',' read -r -a CACHE_SIZES <<<"$CACHE_SIZE_FILTER"
EXPECTED_TASKS=${#CACHE_SIZES[@]}
LAST_INDEX=$((EXPECTED_TASKS - 1))
CACHE_SIZE_EXPORT="${CACHE_SIZE_FILTER//,/:}"

echo "Bounded capacities: ${CACHE_SIZES[*]}"
echo "Submitting $EXPECTED_TASKS CPU-only SPARQ replay tasks for $DATASET_LABEL..."
ARRAY_JOB_ID=$(sbatch \
  --parsable \
  --array="0-${LAST_INDEX}%${MAX_CONCURRENT}" \
  --job-name="$TASK_LOG_PREFIX" \
  --output="$SOURCE_PATH/${TASK_LOG_PREFIX}-%A_%a.out" \
  --export="ALL,SOURCE_PATH=$SOURCE_PATH,EXPERIMENT_PATH=$EXPERIMENT_PATH,EXPERIMENT_CONFIG_RELATIVE=$EXPERIMENT_CONFIG_RELATIVE,RUN_PREFIX=$RUN_PREFIX,DATASET_LABEL=$DATASET_LABEL,RECORDED_LLM_PATH=$RECORDED_LLM_PATH,PREPARED_PAIRS_PATH=$PREPARED_PAIRS_PATH,EMBEDDING_CACHE_PATH=$EMBEDDING_CACHE_PATH,EMBEDDING_DEVICE=cpu,INSTALL_REQUIREMENTS=0,POLICY_FILTER=SPARQ,POSITIVE_CACHE_SIZES_ONLY=1,CACHE_SIZE_FILTER=$CACHE_SIZE_EXPORT" \
  "$SOURCE_PATH/_experiments/slurm/replay_experiment.sbatch")
ARRAY_JOB_ID="${ARRAY_JOB_ID%%;*}"

MERGE_JOB_ID=$(sbatch \
  --parsable \
  --dependency="afterok:$ARRAY_JOB_ID" \
  --output="$SOURCE_PATH/${MERGE_LOG_PREFIX}-%j.out" \
  --export="ALL,SOURCE_PATH=$SOURCE_PATH,EXPERIMENT_PATH=$EXPERIMENT_PATH,EXPERIMENT_ARRAY_JOB_ID=$ARRAY_JOB_ID,RUN_PREFIX=$RUN_PREFIX,EXPECTED_TASKS=$EXPECTED_TASKS,BASELINE_RESULTS_DIR=$BASELINE_RESULTS_DIR" \
  "$SOURCE_PATH/_experiments/slurm/aggregate_incremental_policy.sbatch")
MERGE_JOB_ID="${MERGE_JOB_ID%%;*}"

RESULT_ROOT="$EXPERIMENT_PATH/results/$RUN_PREFIX-$ARRAY_JOB_ID"
echo "SPARQ array job: $ARRAY_JOB_ID"
echo "Merge/plot job: $MERGE_JOB_ID (starts only after all SPARQ tasks succeed)"
echo "Monitor: squeue -j $ARRAY_JOB_ID,$MERGE_JOB_ID"
echo "Progress: tail -f $SOURCE_PATH/${TASK_LOG_PREFIX}-${ARRAY_JOB_ID}_0.out"
echo "Merge log: tail -f $SOURCE_PATH/${MERGE_LOG_PREFIX}-${MERGE_JOB_ID}.out"
echo "SPARQ-only results: $RESULT_ROOT/sparq-only"
echo "All-policy plots: $RESULT_ROOT/comparison/plots"
echo "Cancel: scancel $ARRAY_JOB_ID $MERGE_JOB_ID"
