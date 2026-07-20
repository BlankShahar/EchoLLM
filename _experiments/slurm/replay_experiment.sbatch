#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=0-04:00:00
#SBATCH --job-name=echollm-cache-replay
#SBATCH --output=echollm-cache-replay-%A_%a.out
#SBATCH --mem=58G
#SBATCH --cpus-per-task=8

set -euo pipefail

: "${SOURCE_PATH:?Missing repository source path}"
: "${EXPERIMENT_CONFIG_RELATIVE:?Missing experiment config path}"
: "${RUN_PREFIX:?Missing result run prefix}"
: "${DATASET_LABEL:?Missing dataset label}"
: "${RECORDED_LLM_PATH:?Missing recorded LLM database}"
: "${PREPARED_PAIRS_PATH:?Missing prepared trace}"
: "${EMBEDDING_CACHE_PATH:?Missing shared embedding cache}"

COMMON_SCRIPT="$SOURCE_PATH/_experiments/slurm/run_experiment.sh"
if [[ ! -f "$COMMON_SCRIPT" ]]; then
  echo "Submit from the EchoLLM repository root." >&2
  exit 3
fi

export EMBEDDING_DEVICE=cpu
source "$COMMON_SCRIPT"
