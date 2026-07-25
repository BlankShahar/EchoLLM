#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-04:00:00
#SBATCH --job-name=echollm-sparq-oasst1
#SBATCH --output=echollm-sparq-oasst1-%A_%a.out
#SBATCH --array=0-90%8
#SBATCH --mem=58G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_6000:1

EXPERIMENT_CONFIG_RELATIVE="_experiments/configs/oasst1_default.yaml"
RUN_PREFIX="oasst1-sparq"
DATASET_LABEL="OASST1 single-path trace"
DEFAULT_MODEL="qwen3:4b-instruct"

COMMON_SCRIPT="${SOURCE_PATH:-${SLURM_SUBMIT_DIR:-$PWD}}/_experiments/slurm/run_experiment.sh"
if [[ ! -f "$COMMON_SCRIPT" ]]; then
  echo "Run this command from the EchoLLM repository root:" >&2
  echo "  sbatch _experiments/slurm/run_oasst1.sbatch" >&2
  exit 3
fi
source "$COMMON_SCRIPT"
