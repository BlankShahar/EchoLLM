#!/bin/bash

EXPERIMENT_CONFIG_RELATIVE="_experiments/configs/wildchat_15k.yaml"
RUN_PREFIX="wildchat15k-sparq-incremental"
DATASET_LABEL="WildChat first 15K chronological requests"
TASK_LOG_PREFIX="echollm-sparq-only-wildchat15k"
MERGE_LOG_PREFIX="echollm-sparq-merge-wildchat15k"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_sparq_incremental.sh"
