#!/usr/bin/env python3
"""Solve the four-configuration energy mapping for E0, J1, J2, J3.

Model: E_cell(config) = E0 + S^2 * sum_s J_s * B_s(config),
where H = +sum_{pairs} J_s S_i.S_j (paper convention), |S| = 3/2, and
B_s(config) = sum over unique shell-s Cr pairs (with periodic image multiplicity)
of sigma_i * sigma_j on the 2x2x1 supercell.

Bond counting is done programmatically from the geometry (hex metric
d^2 = dx^2 - dx*dy + dy^2 in primitive-lattice units) — hand-derived counts are
never trusted.  A synthetic round-trip self-test must pass before the real solve
is printed.  Pure python (no numpy) so it runs anywhere.

Usage: solve_J123.py <campaign_dir>   (expects cfg_*/scf.out and PATTERNS.json)
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

CONFIGS = ["fm", "neel", "zigzag", "stripy"]
SHELL_D2 = (1.0 / 3.0, 1.0, 4.0 / 3.0)  # NN, 2NN, 3NN in units of a^2
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from parameters import load
S_MAG = load()['spin_s']
RY_TO_MEV = 13605.693122994
PX = PY = 2  # supercell periods in primitive cells

# Primitive-cell basis (crystal coords of the primitive cell): A=(1/3,2/3), B=(2/3,1/3)
BASIS = {"A": (1.0 / 3.0, 2.0 / 3.0), "B": (2.0 / 3.0, 1.0 / 3.0)}


def bond_sums(sites: list[dict], signs: list[int]) -> list[float]:
    """B_s = (1/2) sum_{i,j,images} sigma_i sigma_j [d2 == shell_s], primitive units."""
    n = len(sites)
    sums = [0.0, 0.0, 0.0]
    for i in range(n):
        bi = BASIS[sites[i]["sublattice"]]
        pi = (sites[i]["ix"] + bi[0], sites[i]["iy"] + bi[1])
        for j in range(n):
            bj = BASIS[sites[j]["sublattice"]]
            pj = (sites[j]["ix"] + bj[0], sites[j]["iy"] + bj[1])
            for mx in range(-2, 3):
                for my in range(-2, 3):
                    dx = pj[0] + mx * PX - pi[0]
                    dy = pj[1] + my * PY - pi[1]
                    d2 = dx * dx - dx * dy + dy * dy
                    for s, ref in enumerate(SHELL_D2):
                        if abs(d2 - ref) < 1.0e-9:
                            sums[s] += 0.5 * signs[i] * signs[j]
    return sums


def solve4(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    aug = [row[:] + [v] for row, v in zip(matrix, rhs)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-14:
            raise RuntimeError("singular design matrix")
        aug[col], aug[piv] = aug[piv], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for r in range(n):
            if r != col:
                f = aug[r][col]
                aug[r] = [v - f * p for v, p in zip(aug[r], aug[col])]
    return [aug[r][n] for r in range(n)]


def parse_scf_out(path: Path) -> tuple[float, bool, list[float]]:
    """Return (final total energy in Ry, converged?, last per-atom mz list for atoms 1..8)."""
    text = path.read_text(errors="replace")
    energies = re.findall(r"^!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", text, re.M)
    if not energies:
        raise RuntimeError(f"{path}: no '!    total energy' line")
    converged = "convergence has been achieved" in text
    # noncolinear per-atom blocks: "atom number    N ... magnetization :  mx my mz"
    mz_by_atom: dict[int, float] = {}
    atom = None
    for line in text.splitlines():
        m = re.match(r"\s*atom number\s+(\d+)", line)
        if m:
            atom = int(m.group(1))
            continue
        m = re.match(r"\s*magnetization\s*:\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", line)
        if m and atom is not None:
            mz_by_atom[atom] = float(m.group(3))  # last occurrence wins
    mz = [mz_by_atom.get(i, float("nan")) for i in range(1, 9)]
    return float(energies[-1]), converged, mz


def self_test(design: list[list[float]]) -> None:
    j_true = [-4.298812e-3 / RY_TO_MEV * RY_TO_MEV, -0.5, 0.23]  # meV, arbitrary probe
    j_true = [-4.3, -0.5, 0.23]
    e0_true = -12345.678901234
    rhs = [
        e0_true + sum(design[c][s + 1] * (j_true[s] / RY_TO_MEV) for s in range(3))
        for c in range(4)
    ]
    sol = solve4(design, rhs)
    j_back = [sol[s + 1] * RY_TO_MEV for s in range(3)]
    err = max(abs(a - b) for a, b in zip(j_back, j_true)) + abs(sol[0] - e0_true)
    if err > 1e-8:
        raise RuntimeError(f"SELF-TEST FAIL: round-trip error {err}")
    print(f"self-test PASS (round-trip error {err:.2e})")


def main() -> None:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    patterns = json.loads((base / "PATTERNS.json").read_text())

    # Design matrix rows [1, S^2 B1, S^2 B2, S^2 B3] per config
    design, bsums = [], {}
    for config in CONFIGS:
        b = bond_sums(patterns[config]["cr_sites"], patterns[config]["signs"])
        bsums[config] = b
        design.append([1.0] + [S_MAG**2 * v for v in b])
        # sanity: bond sums must be (half-)integers
        for v in b:
            if abs(v - round(v)) > 1e-9:
                raise RuntimeError(f"{config}: non-integer bond sum {b}")
    print("bond sums B1,B2,B3 per config (pairs, sign-weighted):")
    for config in CONFIGS:
        print(f"  {config:8s} {[round(v) for v in bsums[config]]}")

    det_check = solve4  # will raise on singular
    self_test([row[:] for row in design])

    energies, all_ok = [], True
    for config in CONFIGS:
        out = base / f"cfg_{config}" / "scf.out"
        e, conv, mz = parse_scf_out(out)
        signs = patterns[config]["signs"]
        mag_ok = all(
            not math.isnan(m) and 2.0 <= abs(m) <= 4.0 and (m > 0) == (sg > 0)
            for m, sg in zip(mz, signs)
        )
        status = "OK" if (conv and mag_ok) else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(
            f"  {config:8s} E={e:.8f} Ry  converged={conv}  "
            f"mz={['%+.2f' % m for m in mz]}  gate={status}"
        )
        energies.append(e)

    if not all_ok:
        print("MAGNETIZATION/CONVERGENCE GATE FAILED — solve aborted "
              "(rerun failed config with constrained_magnetization).")
        sys.exit(2)

    sol = solve4([row[:] for row in design], energies)
    e0, j_ry = sol[0], sol[1:]
    print()
    print(f"E0 = {e0:.8f} Ry")
    for s in range(3):
        j_paper = j_ry[s] * RY_TO_MEV
        j_yours = -j_paper * S_MAG**2
        print(
            f"J{s+1}_paper = {j_paper:+.4f} meV   "
            f"(J{s+1}_yours = {j_yours:+.4f} meV)"
        )
    print()
    print("convention: H = +sum_<ij> J S_i.S_j, |S|=3/2; negative = FM")


if __name__ == "__main__":
    main()
