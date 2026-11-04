# CrI3 ansatz

Potential fitting, Wannier projection, RVR construction, and spin Hamiltonian/torque calculations.

| Directory | Input | Main step | Output |
|---|---|---|---|
| `0_QE` | Clean QE 7.6 source trees | Apply common and magnetic/nonmagnetic sources; build | QE executables |
| `1_scf` | Structure and pseudopotentials | Magnetic SCF; `pp_exam.x < print_V.in` | Saved state, `V_HYC_dp`, `V_QE` |
| `2_band` | Magnetic saved state | `pw.x < band.in`; `bands.x < bands.in` | Magnetic bands |
| `3_band_no_mag` | Fixed scalar `V_HYC_dp`, matching saved state | Nonmagnetic SCF/bands | Nonmagnetic bands |
| `4_potential` | Training fields, spin labels, Wannier weights | Split, fit, symmetry, `make_R_vec` | `V_func_tmp`, `R_vec` |
| `5_wannier` | Nonmagnetic state and `V_HYC_dp` | NSCF, Wannier90, real-space conversion | `wan_data`, `cal_TF`, primitive HR |
| `6_RVR` | Fitted potential and the matching Wannier data | ZGEMM RVR; combine; fold primitive HR | `B_general`, `super_H` |
| `7_spin` | RVR, supercell HR, `spin.in` | Assemble Hamiltonian and torque | `tot_HR`, `direct_torque_*` |

Training fields, pseudopotentials, saved QE states and calculation data are supplied separately.
Keep `wan_data`, `cal_TF`, primitive HR and Wannier weights from the same Wannier run.

## Parameters and build

Requires Python 3.11+ (or 3.10 with `tomli`), NumPy, a Fortran compiler, BLAS/LAPACK,
MPI for MPI programs, and Wannier90. See [0_QE/README.md](0_QE/README.md) for QE 7.6.

Edit the TOML file `parameter`. Atomic positions are in `1_scf/structure.in`; projections
are in `5_wannier/CrI3.win.template`. Paths inside `parameter` are relative to that file.
`nspin=1` is the scalar-Wannier convention with two spin slots; `nspin=2` uses spinor Wanniers.
The supplied spin model supports 18 or 36 functions. `lambda_high` applies only to 18 functions;
36 functions use the fitted modulation channels. `cr_magnetization` also sets the spin-vector length.

```bash
python3 configure.py
python3 build.py 4_potential/make_R_vec.f90 4_potential/fit/fit_com_ortho_cold.f90
python3 build.py 7_spin/cal_Hsk.f90 7_spin/combine.f90 7_spin/cal_torque_hf.f90
```

For Intel MPI/MKL:

```bash
export FC=mpiifort FFLAGS='-O2 -qopenmp -mkl' LIBS='-mkl'
python3 build.py 5_wannier/disentangled/make_dis_mpi.f90 \
  5_wannier/make_wan/make_wan_fun_mpi.f90 6_RVR/speedup/cal_RVR_zgemm.f90 \
  6_RVR/combine.f90 6_RVR/rearrange_HR.f90
```

Builds are reused only when parameters, sources, compiler and flags match. A failed compile
keeps the previous binary. Run through `run.py` to reject stale builds; direct `.x` execution
bypasses that check. Run from the calculation directory shown below. Re-run `configure.py`
after changing parameters; rebuild affected programs with `build.py`.

For independent cases, set `ANSATZ_PARAMETER` and `ANSATZ_BUILD_DIR` to separate paths and
use `configure.py --output-dir /path/to/case`. Relative values of these two environment
variables are resolved from the repository root. Paths inside each parameter file should
point to that case's data. `run.py` keeps the current working directory.

## Wannier and RVR conversion

Run the nonmagnetic NSCF, `wannier90.x -pp CrI3`, `pw2wannier90.x < pw2wannier.in`,
`wannier90.x CrI3`, and the modified `wfck2r.x < wfck2r.in` in `5_wannier`.
The SCF/NSCF steps use the matching fixed scalar potential and saved state. Replace `CrI3`
with `prefix` if changed. The supplied workflow uses no outer `dis_win_min/max` bounds,
`use_ws_distance=false`, and the contiguous `first_band..last_band` window.

Convert the standard `_u.mat`, `_u_dis.mat`, `_hr.dat` files into the records used here:

```bash
python3 5_wannier/prepare.py 5_wannier
cd 5_wannier/disentangled
python3 ../../run.py --mpi 2 5_wannier/disentangled/make_dis_mpi.f90
cd ../..
bash 5_wannier/make_wan/submit.sh
```

The modified `wfck2r.x` writes `WFN/wfn_k*_b*_vec_dp` with `write_hyc=true`.
The disentangling step writes `disentangled/WFN/wfn_k*_w*_vec_dp`;
the Wannier generator writes sequential `make_wan/wan_*` records, normalized by
`sqrt(Ngrid) * Nk`. Provide matching `V_HYC_dp` in `5_wannier`.
Generate Wannier weights before fitting with `bash 5_wannier/make_wan/run_w_density.sh`.

After fitting, place the validated fitted field at `func_dir/V_func_tmp_tmp`, then run
`make_R_vec` to write `R_vec` and `V_func_tmp` in `func_dir`; its grid comes from `scf_dir`.
Convert Wanniers to that grid, then use the accelerated RVR implementation:

```bash
python3 5_wannier/make_wan/prepare_rvr.py 5_wannier/make_wan
mkdir -p 6_RVR/tmp 6_RVR/super_H
cd 6_RVR
python3 ../run.py --mpi 16 6_RVR/speedup/cal_RVR_zgemm.f90 1
# Repeat for every row: 2*wan_min-1..2*wan_max for nspin=1; wan_min..wan_max for nspin=2.
python3 ../run.py 6_RVR/combine.f90
python3 ../run.py 6_RVR/rearrange_HR.f90
cd ..
```

Both converters refuse to replace existing output files. They validate dimensions and preserve the
Wannier amplitudes. The grid converter keeps all directions in `cal_TF`; an existing
`6_RVR/dir_active.in` can select the validated translation support. The scalar RVR program
`6_RVR/cal_RVR_split.f90` is available for comparison and needs at least two MPI ranks.
The standard HR text export limits precision to its printed digits; reuse same-run binary HR
records when their additional precision is needed. RVR is stored in Ry; primitive HR is in eV. Spin combination converts the magnetic contribution
with the existing factor 13.6 before adding it to HR.

## Spin texture and checks

In `7_spin/spin.in`, put all polar angles on the first line and all azimuthal angles on the
second, in degrees. Each line contains `2*period_x*period_y*period_z` values. Ordering is
cell x, then y, then z (z fastest), with Cr sites 1 and 2 in each cell. An optional
`response_sites.in` contains a count followed by zero-based physical site indices;
without it torque is calculated for all sites.

```bash
bash 7_spin/run_spin.sh
python3 -B check.py
```

Checks report passed, failed and skipped cases separately. They cover configuration,
Fortran records, conversion ordering, spin-channel derivatives and the RVR contraction.
Generated inputs, binaries and calculation outputs are excluded from Git.
