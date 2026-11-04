# QE 7.6 modifications

Source replacements for the official [Quantum ESPRESSO 7.6](https://gitlab.com/QEF/q-e/-/archive/qe-7.6/q-e-qe-7.6.tar.gz) tree, taken from
the magnetic and nonmagnetic builds on ISSP.

- `magnetic/`: spin-direction constraints and the `pp_exam.x` potential exporter.
- `nonmagnetic/`: the same base plus a cached scalar-potential reader for
  `pw_no_mag.x`. It reads the first channel of `V_HYC_dp` and removes the local
  magnetic field in each Hamiltonian update.

Paths inside each directory follow the QE source layout. Start with two complete
QE 7.6 trees, including QE's external dependencies, then copy the corresponding
replacement files into each tree:

```bash
cp -R magnetic/. /path/to/qe-7.6-magnetic/
cp -R nonmagnetic/. /path/to/qe-7.6-nonmagnetic/
```

After loading Intel MPI/MKL, build from each QE root:

```bash
bash HYC/build_mpi_mkl.sh
source HYC/mpi_env.sh
```

The magnetic build provides `pw.x` and `pp_exam.x`; the nonmagnetic build provides
`pw.x` and its `pw_no_mag.x` alias. Use the magnetic build for SCF/potential export
and the nonmagnetic build for the fixed-scalar band/Wannier stages.

Export with `pp_exam.x < print_V.in`. The workflow's generated `print_V.in`
enables `write_v_qe` for the potential-fitting stage. `V_HYC_dp` contains a grid
header and four float64 channels `(V0,Bx,By,Bz)` in Ry, stored with Fortran record
markers and x-fast indexing. The nonmagnetic executable requires this file in
its working directory and checks its grid against the QE state.

For directional constraints, set `constrained_magnetization='atomic vector direction'`
and select species with `constrain_species` in `&SYSTEM`. The mode fixes the
target direction while allowing the moment magnitude to vary.

The original QE copyright notices and [GPL license](License) are retained.
