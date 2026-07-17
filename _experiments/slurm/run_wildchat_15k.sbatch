#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=1-00:00:00
#SBATCH --job-name=echollm-wsage-wildchat15k
#SBATCH --output=echollm-wsage-wildchat15k-%A_%a.out
#SBATCH --array=0-49%8
#SBATCH --mem=58G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_6000:1

EXPERIMENT_CONFIG_RELATIVE="_experiments/configs/wildchat_15k.yaml"
RUN_PREFIX="wildchat15k-wsage"
DATASET_LABEL="WildChat first 15K chronological requests"

COMMON_SCRIPT="${SOURCE_PATH:-${SLURM_SUBMIT_DIR:-$PWD}}/_experiments/slurm/run_experiment.sh"
if [[ ! -f "$COMMON_SCRIPT" ]]; then
  echo "Run this command from the EchoLLM repository root:" >&2
  echo "  sbatch _experiments/slurm/run_wildchat_15k.sbatch" >&2
  exit 3
fi
source "$COMMON_SCRIPT"
