#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

FC="${FC:-ifort}"
OUTPUT_FILE="${OUTPUT_FILE:-../../4_potential_2x2_test/w_density}"
WAN_MIN="${WAN_MIN:-1}"
WAN_MAX="${WAN_MAX:-44}"
WEIGHT_FLOOR="${WEIGHT_FLOOR:-0.1}"
NSPIN="${NSPIN:-1}"
CONVERT_X="${CONVERT_X:-1}"
CONVERT_Y="${CONVERT_Y:-1}"
CONVERT_Z="${CONVERT_Z:-0}"

for band in $(seq "$WAN_MIN" "$WAN_MAX"); do
    if [ ! -s "wan_${band}" ]; then
        echo "ERROR: missing ${SCRIPT_DIR}/wan_${band}" >&2
        echo "Run the Wannier-function generation jobs before this step." >&2
        exit 1
    fi
done

"$FC" -O2 make_w_density.f90 -o make_w_density.x
./make_w_density.x "$OUTPUT_FILE" "$WAN_MIN" "$WAN_MAX" "$WEIGHT_FLOOR" \
    "$NSPIN" "$CONVERT_X" "$CONVERT_Y" "$CONVERT_Z" | tee make_w_density.log

