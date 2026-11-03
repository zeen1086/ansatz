# CrI3 ansatz workflow

Current-version code for the noncollinear CrI3 ansatz workflow: Wannier pipeline
→ RVR → H_sk / three-term torque, the native-PW torque chain, and the
frozen-magnon model that consumes it.

This repository is **code only**. Raw inputs and products (`V_func_tmp`,
`B_general`, `wan_data`, densities, wavefunctions, `runs/`) are not included, so
the code cannot be executed without the offline working tree. History-only
material (dated experiment directories, H_sk variants, benchmarks, old check
scripts) is deliberately excluded: one current version per stage.

Snapshot: 2026-11-04. `physical_release = false` — every number produced by
this code is conditional on its stated inputs and is not a released result.

## Layout

```
ansatz/                      entry points of the current model
  torque_s13.py              full-14 s31+s13+SD joint J (two fixed-potential endpoints)
  response_matrix.py         response-matrix driver on top of torque_s13
  native_pw/                 native three-term torque chain (dependency closure only)
    physical_validation_20260924_r01/   fit14.py, torque_sd.py, coordinates.py,
                                        run.py, projection_consistency.py
    coordinate_layer1_issp_20260923_r01/run.py     spin/geometry layer
    three_cell_torque_20260920_r01/analyze.py      torque analysis
    three_cell_torque_20260921_r02/run.py          native readout driver
    full_spatial_candidate_20260918_r02/model.py   ansatz model
    density_ansatz_harris_r02/          pp_variational.x source + build.sh
    frozen_pw_projection_scan_r01/      pw_frozen_wfc.x source + build.sh
    full_k_density_readout_r01/         pp_variational.x readout source + build.sh
    full_spatial_candidate_20260921_r03/model.py    model used by the evaluator
    full_potential_evaluator_20260921_r03/evaluate.py   potential evaluator
    general_torque_correct_grid_20260921_r01/fields.py  field grids / torque weights
    q0_constraint_torque.py, sd_fft_repair_20260921_r01/resample.py
  j2_energy_mapping_20260801/solve_J123.py         J1/J2/J3 mapping used by analyze.py
frozen_magnon/               frozen-magnon / full-HP model + native torque reader
wannier/                     Wannier ansatz pipeline (one current version per stage)
  4_potential/               V_split, R-vectors, supercell atoms, symmetrisation
  4_potential_2x2_test3/     C3 basis / block mask / Wannier weight / V_func_tmp fit
  5_nscf/make_wan/           Wannier function + density generation
  6_THE_5x5_torque/1_make_RVR/          cal_RVR_split.f90 + combine.f90 (RVR production)
  6_THE_5x5_torque/9_spin/              cal_Hsk / cal_Hsk_trace / cal_torque_hf / combine / band
  6_THE_5x5_torque/1_make_RVR_speedup/  packed + blocked-ZGEMM RVR candidate
                                        (see NOTE_RVR_SPEEDUP.md, 8.8x end-to-end)
```

## Pipeline

```
4_potential (V_split) → 4_potential_2x2_test3 (V_func_tmp fit)
  → 5_nscf/make_wan (wan_data, cal_TF)
  → 6_THE_5x5_torque/1_make_RVR (B_general)
  → 6_THE_5x5_torque/9_spin (H_sk, three-term torque)
  → frozen_magnon (model) → ansatz (full-14 fit, s13 endpoints)
```

The native-PW chain is the second route to the torque: density ansatz →
frozen-potential diagonalisation → full-k density readout → three-term torque
`-∫rho_out dV + ∫rho_in dV + ∫(V-V_native) drho_in`.

## Notes

* Scripts keep their original absolute cluster paths
  (`/home/25_li/noncollinear/CrI3/...`, `/home/chen/software/qe-7.1`); edit
  `THE_DIR_OVERRIDE`, `OLD`, `NATIVE` or the `build.sh` paths before reusing
  elsewhere. `frozen_magnon/fit_native_torque.py::native_root()` resolves
  `ansatz/native_pw` first, which this layout provides.
* `ansatz/README.md`, `frozen_magnon/README.md` are the original stage notes;
  they reference sibling directories (`../data`, `../records`, `../runs`) that
  exist only in the offline working tree.
* Build products (`*.x`, `*.o`, `*.mod`) and data are git-ignored.
