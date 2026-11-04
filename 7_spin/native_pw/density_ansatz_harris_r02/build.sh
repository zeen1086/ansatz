#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
eval "$("${PYTHON:-python3}" "$ROOT/parameters.py" --shell)"
QE=$QE_ROOT
MPIF90=$MPIF90
cd "$SCRIPT_DIR"
binary_tmp="$SCRIPT_DIR/pp_variational.x"
test ! -e "$binary_tmp"
flags=(
    -O2 -assume byterecl -g -traceback -nomodule -fpp -allow nofpp_comments
    -D__DFTI -D__MPI
    -I"$QE/external/devxlib/src" -I"$QE/PW/src" -I"$QE/PP/src"
    -I"$QE/include" -I"$QE/FoX/finclude" -I"$MKL_INCLUDE"
    -I"$QE/upflib" -I"$QE/XClib" -I"$QE/Modules" -I"$QE/FFTXlib/src"
    -I"$QE/LAXlib" -I"$QE/UtilXlib" -I"$QE/MBD" -I"$QE/KS_Solvers" -I"$QE/dft-d3"
)
"$MPIF90" "${flags[@]}" -c export.f90 -o driver.o
"$MPIF90" -o "$binary_tmp" \
    driver.o "$QE/PP/src/libpp.a" "$NATIVE_LIBPW" \
    "$QE/KS_Solvers/libks_solvers.a" "$QE/dft-d3/libdftd3qe.a" \
    "$QE/Modules/libqemod.a" "$QE/upflib/libupf.a" "$QE/XClib/xc_lib.a" \
    "$QE/FFTXlib/src/libqefft.a" "$QE/LAXlib/libqela.a" \
    "$QE/UtilXlib/libutil.a" "$QE/MBD/libmbd.a" \
    -L"$QE/external/devxlib/src" -ldevXlib \
    -L"$QE/FoX/lib" -lFoX_dom -lFoX_sax -lFoX_wxml -lFoX_common -lFoX_utils -lFoX_fsys \
    -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
sha256sum export.f90 pp_variational.x "$NATIVE_GRADCORR" > BUILD.sha256
