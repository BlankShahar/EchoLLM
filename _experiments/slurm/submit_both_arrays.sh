#!/bin/bash

set -euo pipefail

SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "Submitting OASST1 array..."
bash "$SCRIPT_DIRECTORY/submit_oasst1_array.sh"
echo
echo "Submitting WildChat array..."
bash "$SCRIPT_DIRECTORY/submit_wildchat_15k_array.sh"
