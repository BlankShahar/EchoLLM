#!/bin/bash

ARRAY_SCRIPT_RELATIVE="_experiments/slurm/run_wildchat_15k.sbatch"
EXPERIMENT_CONFIG_RELATIVE="_experiments/configs/wildchat_15k.yaml"
RUN_PREFIX="wildchat15k-wsage"
EXPECTED_TASKS=50

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_array.sh"
