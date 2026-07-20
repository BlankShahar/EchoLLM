#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=01:00:00
#SBATCH --job-name=echollm-wsage-aggregate
#SBATCH --output=echollm-wsage-aggregate-%j.out
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail

: "${EXPERIMENT_ARRAY_JOB_ID:?Missing source array job ID}"
: "${RUN_PREFIX:?Missing result run prefix}"
: "${EXPECTED_TASKS:?Missing expected task count}"
: "${EXPERIMENT_CONFIG_RELATIVE:?Missing experiment config path}"

EXPERIMENT_PATH="${EXPERIMENT_PATH:-$HOME/_experiments/echollm-sage}"
SOURCE_PATH="${SOURCE_PATH:-${SLURM_SUBMIT_DIR:-$PWD}}"
RESULT_GROUP="$RUN_PREFIX-$EXPERIMENT_ARRAY_JOB_ID"
OUTPUT_DIRECTORY="$EXPERIMENT_PATH/results/$RESULT_GROUP"
TASKS_ROOT="$OUTPUT_DIRECTORY/tasks"
CONFIG_PATH="$SOURCE_PATH/$EXPERIMENT_CONFIG_RELATIVE"

export MPLBACKEND=Agg

echo "Aggregating $EXPECTED_TASKS tasks from array $EXPERIMENT_ARRAY_JOB_ID..."
eval "$(/storage/modules/packages/anaconda/bin/conda shell.bash hook)"
conda activate py313
cd "$SOURCE_PATH"

python -m _experiments.aggregate \
  --tasks-root "$TASKS_ROOT" \
  --output-dir "$OUTPUT_DIRECTORY" \
  --expected-tasks "$EXPECTED_TASKS" \
  --config "$CONFIG_PATH"

echo "Combined results: $OUTPUT_DIRECTORY"
