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

# Modes:
#   1 + no supplied recording -> generate/complete recording, then CPU replay.
#   0 + RECORDED_LLM_PATH      -> reuse existing recording, CPU replay only.
#   0 + no RECORDED_LLM_PATH   -> live backend (GPU array).
PRECOMPUTE_RECORDED_LLM="${PRECOMPUTE_RECORDED_LLM:-1}"

# Preserve externally supplied reusable artifacts.
RECORDED_LLM_PATH="${RECORDED_LLM_PATH:-}"
PREPARED_PAIRS_PATH="${PREPARED_PAIRS_PATH:-}"
EMBEDDING_CACHE_PATH="${EMBEDDING_CACHE_PATH:-}"

if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_CONCURRENT must be a positive integer" >&2
    exit 2
fi

if [[ -n "${MAX_ALLOWED_CONCURRENT:-}" ]] && \
   ((MAX_CONCURRENT > MAX_ALLOWED_CONCURRENT)); then
    echo "MAX_CONCURRENT=$MAX_CONCURRENT exceeds the limit of $MAX_ALLOWED_CONCURRENT" >&2
    exit 2
fi

WAVES=$(((EXPECTED_TASKS + MAX_CONCURRENT - 1) / MAX_CONCURRENT))

REUSE_EXISTING_RECORDING=0
if [[ "$PRECOMPUTE_RECORDED_LLM" != "1" && -n "$RECORDED_LLM_PATH" ]]; then
    REUSE_EXISTING_RECORDING=1
fi

if [[ "$PRECOMPUTE_RECORDED_LLM" == "1" ]]; then
    echo "Preflight: one GPU preparation job, then $EXPECTED_TASKS CPU replay tasks" \
         "in $WAVES waves (max $MAX_CONCURRENT concurrent)."

elif [[ "$REUSE_EXISTING_RECORDING" == "1" ]]; then
    echo "Preflight: reusing existing LLM recordings."
    echo "$EXPECTED_TASKS CPU replay tasks in $WAVES waves" \
         "(max $MAX_CONCURRENT concurrent)."
    echo "No LLM generation and no GPU jobs will be submitted."

else
    echo "Preflight: $EXPECTED_TASKS live-backend GPU tasks in $WAVES waves, " \
         "max $MAX_CONCURRENT concurrent."
fi

# Generation-time projection is only relevant when an LLM backend will
# actually be called. Skip it when replaying an existing recording.
if [[ "$REUSE_EXISTING_RECORDING" != "1" &&
      -n "${TRACE_REQUESTS:-}" &&
      -n "${ESTIMATED_PROMPTS_PER_SECOND:-}" ]]; then

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
        'BEGIN {
            printf "%.2f",
                waves * (requests / qps / 3600 + setup_minutes / 60)
                + replay_waves * replay_hours
        }')

    echo "Projected worst-case array runtime: ${PROJECTED_HOURS}h " \
         "(${TRACE_REQUESTS} unique backend recordings using " \
         "${MODEL:-the configured model} at " \
         "${ESTIMATED_PROMPTS_PER_SECOND} prompt/s, then parallel replay)."

    if [[ -n "${TARGET_HOURS:-}" ]] && \
       awk -v projected="$PROJECTED_HOURS" \
           -v target="$TARGET_HOURS" \
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
ARRAY_DEPENDENCY_ARGS=()

ARRAY_INSTALL_REQUIREMENTS=1
ARRAY_SCRIPT="$SOURCE_PATH/$ARRAY_SCRIPT_RELATIVE"
ARRAY_LOG_PREFIX="$TASK_LOG_PREFIX"

ARRAY_EXPORTS="ALL,SOURCE_PATH=$SOURCE_PATH,EXPERIMENT_PATH=$EXPERIMENT_PATH"

#
# Mode 1: Generate/complete the real LLM recording, then replay on CPUs.
#
if [[ "$PRECOMPUTE_RECORDED_LLM" == "1" ]]; then

    SAFE_MODEL=$(printf '%s' "${MODEL:-qwen3:4b-instruct}" | tr '/:' '__')

    CONFIG_DIGEST=$(
        sha256sum "$SOURCE_PATH/$EXPERIMENT_CONFIG_RELATIVE" |
        awk '{print substr($1, 1, 12)}'
    )

    RECORDED_LLM_PATH="$EXPERIMENT_PATH/recordings/${RUN_PREFIX}-${SAFE_MODEL}.sqlite3"
    PREPARED_PAIRS_PATH="$EXPERIMENT_PATH/prepared/${RUN_PREFIX}-${CONFIG_DIGEST}.jsonl.gz"
    EMBEDDING_CACHE_PATH="$EXPERIMENT_PATH/embeddings/${RUN_PREFIX}-${CONFIG_DIGEST}.sqlite3"

    mkdir -p \
        "$(dirname "$RECORDED_LLM_PATH")" \
        "$(dirname "$PREPARED_PAIRS_PATH")" \
        "$(dirname "$EMBEDDING_CACHE_PATH")"

    RECORD_JOB_ID=$(sbatch \
        --parsable \
        --export="ALL,SOURCE_PATH=$SOURCE_PATH,EXPERIMENT_PATH=$EXPERIMENT_PATH,EXPERIMENT_CONFIG_RELATIVE=$EXPERIMENT_CONFIG_RELATIVE,RUN_PREFIX=${RUN_PREFIX}-record,DATASET_LABEL=$DATASET_LABEL,MODEL=${MODEL:-qwen3:4b-instruct},RECORD_OUTPUT_PATH=$RECORDED_LLM_PATH,PREPARED_PAIRS_PATH=$PREPARED_PAIRS_PATH,EMBEDDING_CACHE_PATH=$EMBEDDING_CACHE_PATH,INSTALL_REQUIREMENTS=1" \
        "$SOURCE_PATH/_experiments/slurm/record_llm.sbatch")

    RECORD_JOB_ID="${RECORD_JOB_ID%%;*}"

    ARRAY_DEPENDENCY_ARGS=(--dependency="afterok:$RECORD_JOB_ID")
    ARRAY_INSTALL_REQUIREMENTS=0
    ARRAY_SCRIPT="$SOURCE_PATH/_experiments/slurm/replay_experiment.sbatch"
    ARRAY_LOG_PREFIX="echollm-cache-replay"

#
# Mode 2: Reuse an already-completed recording. No GPU / no LLM calls.
#
elif [[ "$REUSE_EXISTING_RECORDING" == "1" ]]; then

    if [[ ! -f "$RECORDED_LLM_PATH" ]]; then
        echo "Recorded LLM database not found: $RECORDED_LLM_PATH" >&2
        exit 3
    fi

    # cache_sizes changed, so the config digest changed, but the prepared
    # dataset and embeddings themselves did not. Reuse the latest artifacts
    # from the previous successful run for this same dataset.
    if [[ -z "$PREPARED_PAIRS_PATH" ]]; then
        PREPARED_PAIRS_PATH=$(
            ls -1t "$EXPERIMENT_PATH"/prepared/"$RUN_PREFIX"-*.jsonl.gz \
                2>/dev/null | head -n 1 || true
        )
    fi

    if [[ -z "$EMBEDDING_CACHE_PATH" ]]; then
        EMBEDDING_CACHE_PATH=$(
            ls -1t "$EXPERIMENT_PATH"/embeddings/"$RUN_PREFIX"-*.sqlite3 \
                2>/dev/null | head -n 1 || true
        )
    fi

    if [[ -z "$PREPARED_PAIRS_PATH" || ! -f "$PREPARED_PAIRS_PATH" ]]; then
        echo "Could not find the previous prepared trace for $RUN_PREFIX." >&2
        echo "Expected under: $EXPERIMENT_PATH/prepared/" >&2
        exit 3
    fi

    if [[ -z "$EMBEDDING_CACHE_PATH" || ! -f "$EMBEDDING_CACHE_PATH" ]]; then
        echo "Could not find the previous embedding cache for $RUN_PREFIX." >&2
        echo "Expected under: $EXPERIMENT_PATH/embeddings/" >&2
        exit 3
    fi

    echo "Reusing recorded LLM database: $RECORDED_LLM_PATH"
    echo "Reusing prepared trace:       $PREPARED_PAIRS_PATH"
    echo "Reusing embedding cache:      $EMBEDDING_CACHE_PATH"

    # Previous experiment already installed the environment.
    ARRAY_INSTALL_REQUIREMENTS=0
    ARRAY_SCRIPT="$SOURCE_PATH/_experiments/slurm/replay_experiment.sbatch"
    ARRAY_LOG_PREFIX="echollm-cache-replay"
fi

ARRAY_EXPORTS="$ARRAY_EXPORTS,RECORDED_LLM_PATH=$RECORDED_LLM_PATH,PREPARED_PAIRS_PATH=$PREPARED_PAIRS_PATH,EMBEDDING_CACHE_PATH=$EMBEDDING_CACHE_PATH,INSTALL_REQUIREMENTS=$ARRAY_INSTALL_REQUIREMENTS,EXPERIMENT_CONFIG_RELATIVE=$EXPERIMENT_CONFIG_RELATIVE,RUN_PREFIX=$RUN_PREFIX,DATASET_LABEL=$DATASET_LABEL,MODEL=${MODEL:-qwen3:4b-instruct}"

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

echo "Array job: $ARRAY_JOB_ID ($EXPECTED_TASKS CPU replay tasks, max $MAX_CONCURRENT concurrent)"

if [[ -n "$RECORD_JOB_ID" ]]; then
    echo "Recording job: $RECORD_JOB_ID (one real generation per unique prompt)"
    echo "Preparation log: tail -f $SOURCE_PATH/echollm-record-backend-$RECORD_JOB_ID.out"
    echo "Recorded data: $RECORDED_LLM_PATH"
    echo "Prepared trace: $PREPARED_PAIRS_PATH"
    echo "Embedding cache: $EMBEDDING_CACHE_PATH"
elif [[ "$REUSE_EXISTING_RECORDING" == "1" ]]; then
    echo "Recording job: skipped — using existing recordings"
    echo "Recorded data: $RECORDED_LLM_PATH"
fi

echo "Aggregation job: $AGGREGATE_JOB_ID (starts after every array task succeeds)"

MONITOR_IDS="$ARRAY_JOB_ID,$AGGREGATE_JOB_ID"

if [[ -n "$RECORD_JOB_ID" ]]; then
    MONITOR_IDS="$RECORD_JOB_ID,$MONITOR_IDS"
fi

echo "Monitor: squeue -j $MONITOR_IDS"
echo "Accounting: sacct -j $ARRAY_JOB_ID --format=JobID,State,Elapsed,Start,End"
echo "First task log: tail -f $SOURCE_PATH/$ARRAY_LOG_PREFIX-${ARRAY_JOB_ID}_0.out"
echo "Aggregate log: tail -f $SOURCE_PATH/echollm-cache-aggregate-$AGGREGATE_JOB_ID.out"
echo "Cancel all: scancel $MONITOR_IDS"
echo "Results: ${EXPERIMENT_PATH:-$HOME/_experiments/echollm-sage}/results/$RUN_PREFIX-$ARRAY_JOB_ID"
