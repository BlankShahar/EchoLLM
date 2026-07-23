#!/bin/bash

set -euo pipefail

: "${ARRAY_SCRIPT_RELATIVE:?Missing array script path}"
: "${EXPERIMENT_CONFIG_RELATIVE:?Missing experiment config path}"
: "${RUN_PREFIX:?Missing run prefix}"
: "${EXPECTED_TASKS:?Missing expected task count}"
: "${TASK_LOG_PREFIX:?Missing task log prefix}"
: "${DATASET_LABEL:?Missing dataset label}"

SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOURCE_PATH=$(cd "$SCRIPT_DIRECTORY/../.." && pwd)
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
EXPERIMENT_PATH="${EXPERIMENT_PATH:-$HOME/_experiments/echollm-sage}"
PRECOMPUTE_RECORDED_LLM="${PRECOMPUTE_RECORDED_LLM:-1}"

if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CONCURRENT must be a positive integer" >&2
  exit 2
fi

if [[ -n "${MAX_ALLOWED_CONCURRENT:-}" ]] && ((MAX_CONCURRENT > MAX_ALLOWED_CONCURRENT)); then
  echo "MAX_CONCURRENT=$MAX_CONCURRENT exceeds the limit of $MAX_ALLOWED_CONCURRENT" >&2
  exit 2
fi

WAVES=$(((EXPECTED_TASKS + MAX_CONCURRENT - 1) / MAX_CONCURRENT))

if [[ "$PRECOMPUTE_RECORDED_LLM" == "1" ]]; then
  echo "Preflight: one GPU preparation job, then $EXPECTED_TASKS CPU replay tasks" \
    "in $WAVES waves (max $MAX_CONCURRENT concurrent)."
else
  echo "Preflight: $EXPECTED_TASKS live-backend GPU tasks in $WAVES waves, " \
    "max $MAX_CONCURRENT concurrent."
fi

if [[ -n "${TRACE_REQUESTS:-}" && -n "${ESTIMATED_PROMPTS_PER_SECOND:-}" ]]; then
  GENERATION_WAVES="$WAVES"

  if [[ "$PRECOMPUTE_RECORDED_LLM" == "1" ]]; then
    GENERATION_WAVES=1
  fi

  PROJECTED_HOURS=$(awk \
  -v waves="$GENERATION_WAVES" \
  -v replay_waves="$WAVES" \
  -v requests="$TRACE_REQUESTS" \
  -v qps="$ESTIMATED_PROMPTS_PER_SECOND" \
  -v setup_minutes="${SETUP_MINUTES_PER_WAVE:-0}" \
  -v replay_hours="${ARRAY_REPLAY_HOURS_PER_WAVE:-0.75}" \
  'BEGIN { printf "%.2f", waves * (requests / qps / 3600 + setup_minutes / 60) + replay_waves * replay_hours }')

  echo "Projected worst-case array runtime: ${PROJECTED_HOURS}h " \
    "(${TRACE_REQUESTS} unique backend recordings using ${MODEL:-the configured model} " \
    "at ${ESTIMATED_PROMPTS_PER_SECOND} prompt/s, then parallel replay)."

  if [[ -n "${TARGET_HOURS:-}" ]] && \
    awk -v projected="$PROJECTED_HOURS" -v target="$TARGET_HOURS" \
      'BEGIN { exit !(projected > target) }'; then
    echo "Projection exceeds the ${TARGET_HOURS}h target." >&2
    echo "Set ALLOW_OVERRUN=1 only after measuring a sufficient generation rate." >&2

    if [[ "${ALLOW_OVERRUN:-0}" != "1" ]]; then
      exit 4
    fi
  fi
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: preflight passed; no jobs submitted."
  exit 0
fi

LAST_INDEX=$((EXPECTED_TASKS - 1))
cd "$SOURCE_PATH"

RECORD_JOB_ID=""
RECORDED_LLM_PATH=""
PREPARED_PAIRS_PATH=""
SHARED_EMBEDDING_CACHE=""
ARRAY_DEPENDENCY_ARGS=()
ARRAY_INSTALL_REQUIREMENTS=1
ARRAY_SCRIPT="$SOURCE_PATH/$ARRAY_SCRIPT_RELATIVE"
ARRAY_LOG_PREFIX="$TASK_LOG_PREFIX"

ARRAY_EXPORTS="ALL,SOURCE_PATH=$SOURCE_PATH,EXPERIMENT_PATH=$EXPERIMENT_PATH"

if [[ "$PRECOMPUTE_RECORDED_LLM" == "1" ]]; then
  SAFE_MODEL=$(printf '%s' "${MODEL:-qwen3:4b-instruct}" | tr '/:' '__')

  CONFIG_DIGEST=$(
    sha256sum "$SOURCE_PATH/$EXPERIMENT_CONFIG_RELATIVE" |
      awk '{print substr($1, 1, 12)}'
  )

  # Backend recordings depend on dataset/run prefix and LLM settings, not on
  # cache-policy tuning. Store metadata rejects an incompatible model/options
  # reuse, while threshold/window sweeps reuse the expensive generations.
  RECORDED_LLM_PATH="$EXPERIMENT_PATH/recordings/${RUN_PREFIX}-${SAFE_MODEL}.sqlite3"
  PREPARED_PAIRS_PATH="$EXPERIMENT_PATH/prepared/${RUN_PREFIX}-${CONFIG_DIGEST}.jsonl.gz"
  SHARED_EMBEDDING_CACHE="$EXPERIMENT_PATH/embeddings/${RUN_PREFIX}-${CONFIG_DIGEST}.sqlite3"

  mkdir -p \
    "$(dirname "$RECORDED_LLM_PATH")" \
    "$(dirname "$PREPARED_PAIRS_PATH")" \
    "$(dirname "$SHARED_EMBEDDING_CACHE")"

  RECORD_JOB_ID=$(sbatch \
    --parsable \
    --export="ALL,SOURCE_PATH=$SOURCE_PATH,EXPERIMENT_PATH=$EXPERIMENT_PATH,EXPERIMENT_CONFIG_RELATIVE=$EXPERIMENT_CONFIG_RELATIVE,RUN_PREFIX=${RUN_PREFIX}-record,DATASET_LABEL=$DATASET_LABEL,MODEL=${MODEL:-qwen3:4b-instruct},RECORD_OUTPUT_PATH=$RECORDED_LLM_PATH,PREPARED_PAIRS_PATH=$PREPARED_PAIRS_PATH,EMBEDDING_CACHE_PATH=$SHARED_EMBEDDING_CACHE,INSTALL_REQUIREMENTS=1" \
    "$SOURCE_PATH/_experiments/slurm/record_llm.sbatch")

  RECORD_JOB_ID="${RECORD_JOB_ID%%;*}"

  ARRAY_DEPENDENCY_ARGS=(--dependency="afterok:$RECORD_JOB_ID")
  ARRAY_INSTALL_REQUIREMENTS=0
  ARRAY_SCRIPT="$SOURCE_PATH/_experiments/slurm/replay_experiment.sbatch"
  ARRAY_LOG_PREFIX="echollm-cache-replay"
fi

ARRAY_EXPORTS="$ARRAY_EXPORTS,RECORDED_LLM_PATH=$RECORDED_LLM_PATH,PREPARED_PAIRS_PATH=$PREPARED_PAIRS_PATH,EMBEDDING_CACHE_PATH=$SHARED_EMBEDDING_CACHE,INSTALL_REQUIREMENTS=$ARRAY_INSTALL_REQUIREMENTS,EXPERIMENT_CONFIG_RELATIVE=$EXPERIMENT_CONFIG_RELATIVE,RUN_PREFIX=$RUN_PREFIX,DATASET_LABEL=$DATASET_LABEL,MODEL=${MODEL:-qwen3:4b-instruct}"

ARRAY_JOB_ID=$(sbatch \
  --parsable \
  "${ARRAY_DEPENDENCY_ARGS[@]}" \
  --array="0-$LAST_INDEX%$MAX_CONCURRENT" \
  --export="$ARRAY_EXPORTS" \
  "$ARRAY_SCRIPT")

ARRAY_JOB_ID="${ARRAY_JOB_ID%%;*}"

AGGREGATE_JOB_ID=$(sbatch \
  --parsable \
  --dependency="afterok:$ARRAY_JOB_ID" \
  --export="ALL,SOURCE_PATH=$SOURCE_PATH,EXPERIMENT_ARRAY_JOB_ID=$ARRAY_JOB_ID,RUN_PREFIX=$RUN_PREFIX,EXPECTED_TASKS=$EXPECTED_TASKS,EXPERIMENT_CONFIG_RELATIVE=$EXPERIMENT_CONFIG_RELATIVE" \
  "$SOURCE_PATH/_experiments/slurm/aggregate_experiment.sbatch")

AGGREGATE_JOB_ID="${AGGREGATE_JOB_ID%%;*}"

echo "Array job:       $ARRAY_JOB_ID ($EXPECTED_TASKS CPU replay tasks, max $MAX_CONCURRENT concurrent)"

if [[ -n "$RECORD_JOB_ID" ]]; then
  echo "Recording job:   $RECORD_JOB_ID (one real generation per unique prompt)"
  echo "Preparation log: tail -f $SOURCE_PATH/echollm-record-backend-$RECORD_JOB_ID.out"
  echo "Recorded data:   $RECORDED_LLM_PATH"
  echo "Prepared trace:  $PREPARED_PAIRS_PATH"
  echo "Embedding cache: $SHARED_EMBEDDING_CACHE"
fi

echo "Aggregation job: $AGGREGATE_JOB_ID (starts after every array task succeeds)"

MONITOR_IDS="$ARRAY_JOB_ID,$AGGREGATE_JOB_ID"

if [[ -n "$RECORD_JOB_ID" ]]; then
  MONITOR_IDS="$RECORD_JOB_ID,$MONITOR_IDS"
fi

echo "Monitor:         squeue -j $MONITOR_IDS"
echo "Accounting:      sacct -j $ARRAY_JOB_ID --format=JobID,State,Elapsed,Start,End"
echo "First task log:  tail -f $SOURCE_PATH/$ARRAY_LOG_PREFIX-${ARRAY_JOB_ID}_0.out"
echo "Aggregate log:   tail -f $SOURCE_PATH/echollm-wsage-aggregate-$AGGREGATE_JOB_ID.out"
echo "Cancel all:      scancel $MONITOR_IDS"
echo "Results:         ${EXPERIMENT_PATH:-$HOME/_experiments/echollm-sage}/results/$RUN_PREFIX-$ARRAY_JOB_ID"
