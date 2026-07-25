#!/bin/bash

ARRAY_SCRIPT_RELATIVE="_experiments/slurm/run_wildchat_15k.sbatch"
EXPERIMENT_CONFIG_RELATIVE="_experiments/configs/wildchat_15k.yaml"

RUN_PREFIX="wildchat15k-sparq"
DATASET_LABEL="WildChat first 15K chronological requests"

# 0 once + 15 positive capacities * 6 policies = 91 tasks.
EXPECTED_TASKS=91

TRACE_REQUESTS=13838
TARGET_HOURS=15

ESTIMATED_PROMPTS_PER_SECOND="${ESTIMATED_PROMPTS_PER_SECOND:-1.8}"
SETUP_MINUTES_PER_WAVE="${SETUP_MINUTES_PER_WAVE:-8}"
MAX_ALLOWED_CONCURRENT=8

TASK_LOG_PREFIX="echollm-sparq-wildchat15k"

export MODEL="${MODEL:-qwen3:4b-instruct}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_array.sh"
