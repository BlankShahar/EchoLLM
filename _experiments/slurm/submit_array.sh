#!/bin/bash

set -euo pipefail

: "${ARRAY_SCRIPT_RELATIVE:?Missing array script path}"
: "${EXPERIMENT_CONFIG_RELATIVE:?Missing experiment config path}"
: "${RUN_PREFIX:?Missing run prefix}"
: "${EXPECTED_TASKS:?Missing expected task count}"

SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOURCE_PATH=$(cd "$SCRIPT_DIRECTORY/../.." && pwd)
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"

if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CONCURRENT must be a positive integer" >&2
  exit 2
fi

LAST_INDEX=$((EXPECTED_TASKS - 1))
ARRAY_JOB_ID=$(sbatch \
  --parsable \
  --array="0-$LAST_INDEX%$MAX_CONCURRENT" \
  --export="ALL,SOURCE_PATH=$SOURCE_PATH" \
  "$SOURCE_PATH/$ARRAY_SCRIPT_RELATIVE")
ARRAY_JOB_ID="${ARRAY_JOB_ID%%;*}"

AGGREGATE_JOB_ID=$(sbatch \
  --parsable \
  --dependency="afterok:$ARRAY_JOB_ID" \
  --export="ALL,SOURCE_PATH=$SOURCE_PATH,EXPERIMENT_ARRAY_JOB_ID=$ARRAY_JOB_ID,RUN_PREFIX=$RUN_PREFIX,EXPECTED_TASKS=$EXPECTED_TASKS,EXPERIMENT_CONFIG_RELATIVE=$EXPERIMENT_CONFIG_RELATIVE" \
  "$SOURCE_PATH/_experiments/slurm/aggregate_experiment.sbatch")
AGGREGATE_JOB_ID="${AGGREGATE_JOB_ID%%;*}"

echo "Array job:       $ARRAY_JOB_ID ($EXPECTED_TASKS tasks, max $MAX_CONCURRENT concurrent)"
echo "Aggregation job: $AGGREGATE_JOB_ID (starts after every array task succeeds)"
echo "Results:         ${EXPERIMENT_PATH:-$HOME/_experiments/echollm-sage}/results/$RUN_PREFIX-$ARRAY_JOB_ID"
