#!/usr/bin/env python3
"""Build selected ansatz Fortran sources; no calculations or scheduler calls."""
from pathlib import Path
import argparse
import os
import shlex
import subprocess

from configure import configure
from parameters import ROOT

FIT_MODULES = ["wannier_weight_mod", "cri3_block_mask_mod", "cri3_c3_basis_mod"]


def build(sources: list[Path], fc: str, flags: list[str], libs: list[str], check: bool = False) -> None:
    configure()
    for source in sources:
        source = source.resolve()
        source.relative_to(ROOT)
        if "native_pw" in source.parts:
            raise ValueError("build QE-linked native sources with their own build.sh")
        work = ROOT / ".build" / source.relative_to(ROOT).with_suffix("")
        work.mkdir(parents=True, exist_ok=True)
        module = ROOT / ".build/parameter_mod.f90"
        dependencies = [module]
        if source.parent.name == "fit" and source.stem not in FIT_MODULES:
            dependencies += [source.parent / (name + ".f90") for name in FIT_MODULES]
        if source.stem == "cal_RVR_zgemm":
            dependencies.append(source.parent / "rvr_kernel_mod.f90")
        cmd = shlex.split(fc) + flags
        if check:
            cmd += ["-fsyntax-only"]
        cmd += [str(p) for p in dependencies] + [str(source)]
        if not check:
            cmd += ["-o", str(source.with_suffix(".x"))] + libs
        print(shlex.join(cmd), flush=True)
        subprocess.run(cmd, cwd=work, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--fc", default=os.environ.get("FC", "gfortran"))
    parser.add_argument("--flags", default=os.environ.get("FFLAGS", "-O2 -fopenmp -ffree-line-length-none -fallow-argument-mismatch"))
    parser.add_argument("--libs", default=os.environ.get("LIBS", "-llapack -lblas"))
    parser.add_argument("--check", action="store_true", help="GNU syntax check only; use an MPI compiler for MPI sources")
    args = parser.parse_args()
    build(args.sources, args.fc, shlex.split(args.flags), shlex.split(args.libs), args.check)
