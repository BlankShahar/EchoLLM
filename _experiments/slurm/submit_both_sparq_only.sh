#!/bin/bash

set -euo pipefail

: "${OASST_BASELINE_RESULTS:?Set OASST_BASELINE_RESULTS to the completed OASST1 result directory}"
: "${WILDCHAT_BASELINE_RESULTS:?Set WILDCHAT_BASELINE_RESULTS to the completed WildChat result directory}"

SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "Submitting incremental OASST1 SPARQ replay..."
BASELINE_RESULTS_DIR="$OASST_BASELINE_RESULTS" \
  bash "$SCRIPT_DIRECTORY/submit_oasst1_sparq_only.sh"

echo
echo "Submitting incremental WildChat SPARQ replay..."
BASELINE_RESULTS_DIR="$WILDCHAT_BASELINE_RESULTS" \
  bash "$SCRIPT_DIRECTORY/submit_wildchat_15k_sparq_only.sh"
