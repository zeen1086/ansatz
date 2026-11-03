#!/bin/bash
#PBS -N 9_spin
#PBS -j oe
#PBS -o log
#PBS -l nodes=1:ppn=4
#PBS -q GroupA
#PBS -V


cd $PBS_O_WORKDIR

export THE_DIR_OVERRIDE="${THE_DIR_OVERRIDE:-$(cd ../.. && pwd)/}"

ifort cal_Hsk.f90 -mkl -xHost -o cal_Hsk 
ifort combine.f90 -mkl -xHost -o combine 
ifort cal_torque_hf.f90 -qopenmp -O2 -mkl -xHost -o cal_tau

mkdir -p "./tot_HR"
./cal_Hsk 
./combine
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
./cal_tau

OMP_NUM_THREADS=1
# mpiifort band.f90 -mkl
# mpirun -np 16 ./a.out
