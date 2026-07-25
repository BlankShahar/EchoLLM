#!/bin/bash

EXPERIMENT_CONFIG_RELATIVE="_experiments/configs/oasst1_default.yaml"
RUN_PREFIX="oasst1-sub1k"
DATASET_LABEL="OASST1 single-path trace"
TASK_LOG_PREFIX="echollm-sub1k-oasst1"
MERGE_LOG_PREFIX="echollm-sub1k-merge-oasst1"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_sub1k_incremental.sh"
