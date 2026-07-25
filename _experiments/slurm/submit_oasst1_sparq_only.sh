#!/bin/bash

EXPERIMENT_CONFIG_RELATIVE="_experiments/configs/oasst1_default.yaml"
RUN_PREFIX="oasst1-sparq-incremental"
DATASET_LABEL="OASST1 single-path trace"
TASK_LOG_PREFIX="echollm-sparq-only-oasst1"
MERGE_LOG_PREFIX="echollm-sparq-merge-oasst1"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_sparq_incremental.sh"
