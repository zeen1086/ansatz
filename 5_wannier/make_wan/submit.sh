#!/usr/bin/env bash
# Run inside an existing allocation; no jobs are submitted.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PYTHON=${PYTHON:-python3}
cd "$ROOT"
"$PYTHON" build.py --fc "${MPIFC:-mpifort}" 5_wannier/make_wan/make_wan_fun_mpi.f90
WAN_MIN=$("$PYTHON" parameters.py wan_min)
WAN_MAX=$("$PYTHON" parameters.py wan_max)
cd "$SCRIPT_DIR"
MPI_RANKS=${MPI_RANKS:-2}
if ((MPI_RANKS < 2)); then
    echo 'The master/worker Wannier generator needs at least two MPI ranks.' >&2
    exit 1
fi
for ((band=WAN_MIN; band<=WAN_MAX; band++)); do
    mpirun -np "$MPI_RANKS" ./make_wan_fun_mpi.x "$band"
done
