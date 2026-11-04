# CrI3 ansatz

Ansatz potential fitting, Wannier projection, RVR construction, and spin Hamiltonian/torque calculations.

## Layout

```text
1_scf/             Magnetic SCF and potential-export templates
2_band/            Magnetic band-structure templates
3_band_no_mag/     Nonmagnetic SCF and band-structure templates
4_potential/       Potential splitting, symmetry, and real-space indexing
  fit/             Potential fitting and symmetry checks
5_wannier/         Wannier90, pw2wannier, and real-space Wannier functions
6_RVR/             RVR construction and combination
  speedup/         Blocked ZGEMM implementation
7_spin/            Spin Hamiltonian, bands, occupations, and torques
  native_pw/       Native plane-wave potential and torque routines
parameter          Global settings (TOML)
configure.py       Generate QE/Wannier inputs and the Fortran parameter module
build.py           Compile Fortran programs
```

## Configuration and build

Requires Python 3.11+ (or Python 3.10 with `tomli`), NumPy, and a Fortran compiler
with BLAS/LAPACK. MPI programs also require MPI. The native plane-wave routines
use the modified Quantum ESPRESSO libraries specified in `parameter`.

Edit `parameter` for paths, grids, model dimensions, fitting controls, and spin
settings. Atomic positions are in `1_scf/structure.in`; Wannier projections are
in `5_wannier/CrI3.win.template`. Relative paths are resolved from the parameter
file's directory. Use `ANSATZ_PARAMETER` to select another parameter file.

```bash
python3 configure.py
python3 build.py 4_potential/make_R_vec.f90 4_potential/fit/fit_com_ortho_cold.f90
python3 build.py 7_spin/cal_Hsk.f90 7_spin/combine.f90 7_spin/cal_torque_hf.f90
```

For Intel MPI/MKL:

```bash
python3 build.py --fc mpiifort --flags='-O2 -qopenmp -mkl' --libs='-mkl' \
  5_wannier/make_wan/make_wan_fun_mpi.f90 6_RVR/cal_RVR_split.f90 \
  6_RVR/speedup/cal_RVR_zgemm.f90 7_spin/band.f90
```

After changing `parameter`, regenerate inputs and rebuild affected Fortran
programs. Python entry points read the settings at runtime. Generated inputs,
binaries, and calculation data are excluded from Git.

## Running

Provide the pseudopotentials, SCF saved states, training fields, and Wannier data
for the chosen calculation. The Wannier/RVR chain requires matching `wan_data`,
`cal_TF`, `V_HYC_dp`, `R_vec`, and `V_func_tmp`. Spin assembly also needs the
Hamiltonian files in `super_h_dir` and a local `spin.in` texture.
The Wannier spin assembly supports `n_func=18/36`; the native-PW full14 routines
use a separate 14-kernel representation.

Run both RVR implementations from `6_RVR/`; the first argument selects the row:

```bash
cd 6_RVR
mkdir -p tmp
mpirun -np 2 ./cal_RVR_split.x 1
# Alternative implementation:
mpirun -np 2 ./speedup/cal_RVR_zgemm.x 1
```

The reference RVR and Wannier generators require at least two MPI ranks.
Combine RVR rows with `6_RVR/combine.x`. The native QE programs have individual
`build.sh` scripts under `7_spin/native_pw/`; configure their library paths first.
`7_spin/fit_native_torque.py` reads saved three-term torques and fits J1–J3.

Basic checks:

```bash
python3 -B test_configuration.py
python3 -B 7_spin/torque_s13.py check
python3 -B 7_spin/response_matrix.py check
```
