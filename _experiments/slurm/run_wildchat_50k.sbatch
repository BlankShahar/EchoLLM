#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=7-00:00:00
#SBATCH --job-name=echollm-wsage-wildchat50k
#SBATCH --output=echollm-wsage-wildchat50k-%j.out
#SBATCH --mem=58G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_6000:1

EXPERIMENT_CONFIG_RELATIVE="_experiments/configs/wildchat_50k.yaml"
RUN_PREFIX="wildchat50k-wsage"
DATASET_LABEL="WildChat first 50K chronological requests"

COMMON_SCRIPT="${SLURM_SUBMIT_DIR:-$PWD}/_experiments/slurm/run_experiment.sh"
if [[ ! -f "$COMMON_SCRIPT" ]]; then
  echo "Run this command from the EchoLLM repository root:" >&2
  echo "  sbatch _experiments/slurm/run_wildchat_50k.sbatch" >&2
  exit 3
fi
source "$COMMON_SCRIPT"
