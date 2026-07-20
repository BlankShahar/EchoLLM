#!/bin/bash

ARRAY_SCRIPT_RELATIVE="_experiments/slurm/run_oasst1.sbatch"
EXPERIMENT_CONFIG_RELATIVE="_experiments/configs/oasst1_default.yaml"
RUN_PREFIX="oasst1-wsage"
EXPECTED_TASKS=22
TRACE_REQUESTS=14142
TARGET_HOURS=6
ESTIMATED_PROMPTS_PER_SECOND="${ESTIMATED_PROMPTS_PER_SECOND:-1.8}"
SETUP_MINUTES_PER_WAVE="${SETUP_MINUTES_PER_WAVE:-8}"
MAX_ALLOWED_CONCURRENT=8
TASK_LOG_PREFIX="echollm-wsage-oasst1"
export MODEL="${MODEL:-qwen3:4b-instruct}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_array.sh"
