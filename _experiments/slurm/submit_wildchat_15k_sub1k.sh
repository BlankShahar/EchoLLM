#!/bin/bash

EXPERIMENT_CONFIG_RELATIVE="_experiments/configs/wildchat_15k.yaml"
RUN_PREFIX="wildchat15k-sub1k"
DATASET_LABEL="WildChat first 15K chronological requests"
TASK_LOG_PREFIX="echollm-sub1k-wildchat15k"
MERGE_LOG_PREFIX="echollm-sub1k-merge-wildchat15k"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_sub1k_incremental.sh"
