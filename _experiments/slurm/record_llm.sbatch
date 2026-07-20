#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-08:00:00
#SBATCH --job-name=echollm-record-backend
#SBATCH --output=echollm-record-backend-%j.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_6000:1

set -euo pipefail

: "${EXPERIMENT_CONFIG_RELATIVE:?Missing experiment config path}"
: "${RUN_PREFIX:?Missing run prefix}"
: "${DATASET_LABEL:?Missing dataset label}"
: "${RECORD_OUTPUT_PATH:?Missing recorded backend output path}"
: "${PREPARED_PAIRS_PATH:?Missing prepared-pairs output path}"
: "${EMBEDDING_CACHE_PATH:?Missing shared embedding-cache path}"

COMMON_SCRIPT="${SOURCE_PATH:-${SLURM_SUBMIT_DIR:-$PWD}}/_experiments/slurm/run_experiment.sh"
if [[ ! -f "$COMMON_SCRIPT" ]]; then
  echo "Submit from the EchoLLM repository root." >&2
  exit 3
fi
source "$COMMON_SCRIPT"
