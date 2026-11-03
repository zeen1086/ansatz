#!/bin/bash
set -euo pipefail
QE=/home/chen/software/qe-7.1
MPIF90=/opt/intel/oneapi/mpi/2021.2.0/bin/mpiifort
cd /home/25_li/noncollinear/CrI3/deployment/pure_s_derivative_20260911_r00/full_k_density_readout_r01
binary_tmp=/home/25_li/noncollinear/CrI3/deployment/pure_s_derivative_20260911_r00/full_k_density_readout_r01/pp_variational.x
test ! -e "$binary_tmp"
flags=(
    -O2 -assume byterecl -g -traceback -nomodule -fpp -allow nofpp_comments
    -D__DFTI -D__MPI
    -I"$QE/external/devxlib/src" -I"$QE/PW/src" -I"$QE/PP/src"
    -I"$QE/include" -I"$QE/FoX/finclude" -I/opt/intel/oneapi/mkl/2021.2.0/include
    -I"$QE/upflib" -I"$QE/XClib" -I"$QE/Modules" -I"$QE/FFTXlib/src"
    -I"$QE/LAXlib" -I"$QE/UtilXlib" -I"$QE/MBD" -I"$QE/KS_Solvers" -I"$QE/dft-d3"
)
"$MPIF90" "${flags[@]}" -c export.f90 -o driver.o
"$MPIF90" "${flags[@]}" -c sum_band_collected.f90 -o sum_band_collected.o
"$MPIF90" -o "$binary_tmp" \
    driver.o sum_band_collected.o "$QE/PP/src/libpp.a" /home/25_li/noncollinear/CrI3/deployment/pure_s_derivative_20260911_r00/qx_variational_plus_ethr_floor_r01/libpw_cr_only.a \
    "$QE/KS_Solvers/libks_solvers.a" "$QE/dft-d3/libdftd3qe.a" \
    "$QE/Modules/libqemod.a" "$QE/upflib/libupf.a" "$QE/XClib/xc_lib.a" \
    "$QE/FFTXlib/src/libqefft.a" "$QE/LAXlib/libqela.a" \
    "$QE/UtilXlib/libutil.a" "$QE/MBD/libmbd.a" \
    -L"$QE/external/devxlib/src" -ldevXlib \
    -L"$QE/FoX/lib" -lFoX_dom -lFoX_sax -lFoX_wxml -lFoX_common -lFoX_utils -lFoX_fsys \
    -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
sha256sum export.f90 pp_variational.x /home/25_li/noncollinear/CrI3/deployment/pure_s_derivative_20260911_r00/qx_variational_plus_ethr_floor_r01/gradcorr.f90 > BUILD.sha256
