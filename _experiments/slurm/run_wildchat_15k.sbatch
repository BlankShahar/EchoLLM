#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=7-00:00:00
#SBATCH --job-name=echollm-wsage-wildchat15k
#SBATCH --output=echollm-wsage-wildchat15k-%j.out
#SBATCH --mem=58G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_6000:1

EXPERIMENT_CONFIG_RELATIVE="_experiments/configs/wildchat_15k.yaml"
RUN_PREFIX="wildchat15k-wsage"
DATASET_LABEL="WildChat first 15K chronological requests"

COMMON_SCRIPT="${SLURM_SUBMIT_DIR:-$PWD}/_experiments/slurm/run_experiment.sh"
if [[ ! -f "$COMMON_SCRIPT" ]]; then
  echo "Run this command from the EchoLLM repository root:" >&2
  echo "  sbatch _experiments/slurm/run_wildchat_15k.sbatch" >&2
  exit 3
fi
source "$COMMON_SCRIPT"
