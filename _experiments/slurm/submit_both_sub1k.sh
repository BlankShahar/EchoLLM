#!/bin/bash

set -euo pipefail

: "${OASST_EXISTING_RESULTS:?Set OASST_EXISTING_RESULTS to the completed OASST1 all-policy comparison}"
: "${WILDCHAT_EXISTING_RESULTS:?Set WILDCHAT_EXISTING_RESULTS to the completed WildChat all-policy comparison}"

SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "Submitting supplemental sub-1K OASST1 replay..."
EXISTING_RESULTS_DIR="$OASST_EXISTING_RESULTS" \
  bash "$SCRIPT_DIRECTORY/submit_oasst1_sub1k.sh"

echo
echo "Submitting supplemental sub-1K WildChat replay..."
EXISTING_RESULTS_DIR="$WILDCHAT_EXISTING_RESULTS" \
  bash "$SCRIPT_DIRECTORY/submit_wildchat_15k_sub1k.sh"
