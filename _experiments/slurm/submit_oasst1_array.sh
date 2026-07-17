#!/bin/bash

ARRAY_SCRIPT_RELATIVE="_experiments/slurm/run_oasst1.sbatch"
EXPERIMENT_CONFIG_RELATIVE="_experiments/configs/oasst1_default.yaml"
RUN_PREFIX="oasst1-wsage"
EXPECTED_TASKS=50

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_array.sh"
