#!/usr/bin/env python3
"""Run the Luo full-HP V1--V4 scattering path.

Consumes explicitly signed J1/J2/J3 in meV. The dispersion can be
kept at T=0 or renormalized self-consistently with Luo et al. PRB 108,
094432 (2023), Eqs. (6), (7), (12), and (14)--(16).
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "fortran" / "cal_luo_full_hp_scattering.f90"
if not SOURCE.exists():
    SOURCE = ROOT / "cal_luo_full_hp_scattering.f90"
TEMPERATURES = (6.7, 20.0, 30.0, 50.0)
MODEL_DEFINITIONS = {
    "calculation_identity_version": 1,
    "h2_definition": "HEISENBERG_SIA_NORMAL_ORDERED_V1",
    "vertex_definition": "HP_COMPLEX_BONDS_V1",
    "fgr_definition": "ORDERED_FINAL_HALF_IDENTICAL_V1",
    "fgr_final_state_factors": [0.5, 0.5, 1.0, 1.0],
    "fgr_transition_filter": "exclude_f_equals_i_and_degenerate_branch_conversion",
    "degeneracy_tolerance_meV": 1.0e-10,
    "hbar_meV_s": 6.582119569e-13,
}


def model_definitions(scattering: str = "legacy_four_channel") -> dict:
    if scattering == "legacy_four_channel":
        return dict(MODEL_DEFINITIONS)
    if scattering != "full_sunset":
        raise ValueError(f"unsupported scattering definition: {scattering}")
    return {
        **MODEL_DEFINITIONS,
        "vertex_definition": "HP_COMPLEX_BONDS_ALL_BRANCHES_V1",
        "fgr_definition": "ORDERED_FINAL_HALF_ALL_BRANCHES_V1",
        "fgr_final_state_factors": [0.5] * 16,
        "fgr_transition_filter": "exclude_same_fock_state_and_degenerate_spectator_conversion",
        "spectral_definition": "ON_SHELL_HWHM_HBAR_NET_RATE_OVER_2",
        "bose_factor": "n_bath*(1+n_final1+n_final2)-n_final1*n_final2",
        "reference": "Eto_PRX16_021053_2026_Eqs12_15_29",
        "quartic_order": "existing_leading_HP_normal_ordered_J123_A",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    return payload


def resolve_path(config_path: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def parse_ansatz_j(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        record = load_json(path)
        if record.get("exchange_convention") != "H=+sum_<ij> J_ij S_i.S_j":
            raise ValueError("exchange_convention must explicitly declare H=+sum_<ij> J_ij S_i.S_j")
        if record.get("rank") != 3 or not record.get("J_meV"):
            raise ValueError("J1/J2/J3 must be independently identified (rank 3)")
        values = {key: float(record["J_meV"][key]) for key in ("J1", "J2", "J3")}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("non-finite exchange")
        return values
    # Historical reports use both Hamiltonian signs. Never infer a sign from J_paper.
    signs = set(re.findall(r"H\s*=\s*([+-])\s*sum", text))
    if len(signs) != 1:
        raise ValueError(f"missing or ambiguous Hamiltonian sign in {path}")
    factor = 1.0 if signs == {"+"} else -1.0
    values: dict[str, float] = {}
    pattern = re.compile(r"^(J[123])_paper\s*=.*?\(([-+0-9.eE]+)\s+meV\)", re.MULTILINE)
    for match in pattern.finditer(text):
        key = match.group(1)
        if key in values:
            raise ValueError(f"duplicate {key} in {path}")
        values[key] = factor * float(match.group(2))
    if set(values) != {"J1", "J2", "J3"}:
        raise ValueError(f"did not find exactly J1/J2/J3 paper-convention values in {path}")
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"non-finite exchange in {path}")
    return values


def frozen_model(config_path: Path) -> tuple[dict[str, object], Path]:
    config = load_json(config_path)
    source = resolve_path(config_path, config["j_source_file"])
    j_values = parse_ansatz_j(source)
    expected = config.get("expected_j_meV")
    if isinstance(expected, dict):
        for key, actual in j_values.items():
            target = float(expected[key])
            if not math.isclose(actual, target, rel_tol=0.0, abs_tol=5e-12):
                raise ValueError(f"{key} changed in ansatz source: expected {target}, got {actual}")
    spin_s = float(config.get("spin_S", 1.5))
    if source.suffix == ".json" and not math.isclose(float(load_json(source)["spin_S"]), spin_s, rel_tol=0, abs_tol=1e-12):
        raise ValueError("spin_S differs from the torque-derived exchange normalization")
    a_mev = float(config["A_meV"])
    if not math.isfinite(spin_s) or spin_s <= 0 or not math.isfinite(a_mev) or a_mev < 0:
        raise ValueError("spin_S must be positive and A_meV must be finite and nonnegative")
    dispersion = str(config.get("dispersion", "fixed_zero_temperature"))
    if dispersion not in {"fixed_zero_temperature", "self_consistent_mean_field"}:
        raise ValueError(f"unsupported dispersion mode: {dispersion}")
    sc_tolerance = float(config.get("self_consistency_tolerance", 1.0e-10))
    sc_mixing = float(config.get("self_consistency_mixing", 0.5))
    sc_max_iterations = int(config.get("self_consistency_max_iterations", 10000))
    sc_mesh = int(config.get("self_consistency_mesh", 24))
    xi_mev = float(config.get("xi_meV", abs(j_values["J1"]) / 100.0))
    temperatures = [float(value) for value in config.get("temperatures_K", TEMPERATURES)]
    scattering = str(config.get("scattering_definition", "legacy_four_channel"))
    if not math.isfinite(xi_mev) or xi_mev <= 0:
        raise ValueError("xi_meV must be finite and positive")
    if len(temperatures) != 4 or len(set(temperatures)) != 4 or not all(math.isfinite(t) and t > 0 for t in temperatures):
        raise ValueError("temperatures_K must contain four distinct finite positive temperatures")
    if not math.isfinite(sc_tolerance) or sc_tolerance <= 0 or not 0 < sc_mixing <= 1 or sc_max_iterations < 1 or sc_mesh < 2:
        raise ValueError("invalid self-consistency controls")
    model: dict[str, object] = {
        **model_definitions(scattering),
        "scattering_definition": scattering,
        "mode": "luo_full_hp",
        "run_class": str(config.get("run_class", "CONDITIONAL_J123_FULL_HP")),
        "physical_release": False,
        "hamiltonian": "H=+sum_n J_n sum_<ij>_n S_i.S_j-A sum_i (S_i^z)^2",
        "spin_S": spin_s,
        "J1_meV": j_values["J1"],
        "J2_meV": j_values["J2"],
        "J3_meV": j_values["J3"],
        "A_meV": a_mev,
        "anisotropy_source": str(config.get("anisotropy_source", "explicit input configuration")),
        "gap_meV": (2.0 * spin_s - 1.0) * a_mev,
        "xi_meV": xi_mev,
        "temperatures_K": temperatures,
        "dispersion": dispersion,
        "self_consistency_equations": "Luo_PRB108_094432_Eqs_6_7_12_14_16",
        "self_consistency_tolerance": sc_tolerance,
        "self_consistency_mixing": sc_mixing,
        "self_consistency_max_iterations": sc_max_iterations,
        "self_consistency_mesh": sc_mesh,
        "delta": "lorentzian",
        "j_source_file": str(source),
        "j_source_sha256": sha256(source),
        "kernel_source": str(SOURCE),
        "kernel_source_sha256": sha256(SOURCE),
    }
    return model, source


def compiler_command(name: str, output: Path) -> list[str]:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"Fortran compiler not found: {name}")
    if "ifort" in Path(executable).name or "ifx" in Path(executable).name:
        flags = ["-O3", "-qopenmp"]
    else:
        flags = ["-O3", "-fopenmp", "-ffree-line-length-none"]
    return [executable, str(SOURCE), *flags, "-o", str(output)]


def ensure_binary(binary: Path, compiler: str) -> None:
    stamp = binary.with_name(binary.name + ".source_sha256")
    expected = sha256(SOURCE)
    if not binary.exists() or not stamp.exists() or stamp.read_text().strip() != expected:
        subprocess.run(compiler_command(compiler, binary), check=True)
        stamp.write_text(expected + "\n")


@contextmanager
def calculation_directory(work_dir: Path, create: bool = False):
    """One writer/executable per directory; no shared luo_scatt.in race."""
    if create:
        work_dir.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(work_dir, os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another calculation is using {work_dir}") from error
        yield
    finally:
        os.close(descriptor)


def bind_calculation(work_dir: Path, source: Path, model: dict[str, object] | None = None,
                     mesh: int | None = None, create: bool = False) -> dict[str, object]:
    """Call under calculation_directory. Identity is created once, never rewritten."""
    identity = work_dir / "calculation_identity.json"
    displayed = work_dir / "spin_model_frozen.json"
    message = f"calculation identity missing or changed in {work_dir}; use a new work directory"
    if identity.exists():
        try:
            recorded = load_json(identity)
            if load_json(displayed) != recorded or (model is not None and model != recorded):
                raise ValueError(message)
        except (OSError, ValueError) as error:
            raise ValueError(message) from error
    else:
        if not create or model is None or displayed.exists() or any(work_dir.glob("luo_*")) or (work_dir / "path_points").exists():
            raise ValueError(message)
        recorded = model
    definitions = model_definitions(str(recorded.get("scattering_definition", "legacy_four_channel")))
    if any(recorded.get(key) != value for key, value in definitions.items()):
        raise ValueError(message)
    if recorded.get("kernel_source_sha256") != sha256(source):
        raise ValueError(message)
    try:
        if recorded.get("j_source_sha256") != sha256(Path(str(recorded["j_source_file"]))):
            raise ValueError(message)
    except (KeyError, OSError) as error:
        raise ValueError(message) from error
    if mesh is not None and recorded.get("internal_fgr_mesh") != [mesh, mesh]:
        raise ValueError(message)
    if not identity.exists():
        content = json.dumps(recorded, indent=2, sort_keys=True, allow_nan=False) + "\n"
        # O_EXCL also fails closed if a writer bypasses our directory lock.
        with identity.open("x", encoding="utf-8") as stream:
            stream.write(content)
        with displayed.open("x", encoding="utf-8") as stream:
            stream.write(content)
    return recorded


def requested_model(args: argparse.Namespace) -> dict[str, object]:
    if args.mesh < 2:
        raise ValueError("mesh must exceed one")
    model, _ = frozen_model(Path(args.config).resolve())
    model["internal_fgr_mesh"] = [args.mesh, args.mesh]
    return model


def write_input(
    work_dir: Path,
    model: dict[str, object],
    mesh: int,
    iq_start: int,
    iq_end: int,
    suffix: str,
    compute_rates: bool,
) -> None:
    band_stem = "luo_bands_finite_t" if model["dispersion"] == "self_consistent_mean_field" else "luo_band_t0"
    lines = [
        f"nqx={mesh}",
        f"nqy={mesh}",
        f"iq_start={iq_start}",
        f"iq_end={iq_end}",
        f"spin_S={model['spin_S']}",
        f"J1_meV={model['J1_meV']}",
        f"J2_meV={model['J2_meV']}",
        f"J3_meV={model['J3_meV']}",
        f"A_meV={model['A_meV']}",
        f"xi_meV={model['xi_meV']}",
        f"hbar_meV_s={model['hbar_meV_s']}",
        f"scattering_definition={model.get('scattering_definition', 'legacy_four_channel')}",
        f"spectral_file=full_sunset_spectral{suffix}.tsv",
        "temperatures_K=" + " ".join(str(t) for t in model["temperatures_K"]),
        f"dispersion_mode={model['dispersion']}",
        f"sc_tolerance={model['self_consistency_tolerance']}",
        f"sc_mixing={model['self_consistency_mixing']}",
        f"sc_max_iterations={model['self_consistency_max_iterations']}",
        f"sc_nqx={model['self_consistency_mesh']}",
        f"sc_nqy={model['self_consistency_mesh']}",
        f"compute_rates={'.true.' if compute_rates else '.false.'}",
        f"rate_file=luo_four_rates{suffix}.tsv",
        f"band_file={band_stem}{suffix}.tsv",
        f"diag_file=luo_vertex_diagnostic{suffix}.tsv",
        f"summary_file=luo_scattering_summary{suffix}.out",
        f"run_class={model['run_class']}",
    ]
    (work_dir / "luo_scatt.in").write_text("\n".join(lines) + "\n", encoding="utf-8")


def command_prepare(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir).resolve()
    model = requested_model(args)
    with calculation_directory(work_dir, create=True):
        bind_calculation(work_dir, SOURCE, model, args.mesh, create=True)
    print(work_dir / "spin_model_frozen.json")


def command_compile(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir).resolve()
    output = work_dir / "cal_luo_full_hp_scattering"
    with calculation_directory(work_dir):
        bind_calculation(work_dir, SOURCE)
        subprocess.run(compiler_command(args.compiler, output), check=True)
        output.with_name(output.name + ".source_sha256").write_text(sha256(SOURCE) + "\n")
    print(output)


def command_run(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir).resolve()
    model = requested_model(args)
    nq = args.mesh * args.mesh
    iq_start = args.iq_start
    iq_end = args.iq_end or nq
    if not 1 <= iq_start <= iq_end <= nq:
        raise ValueError(f"invalid iq range {iq_start}:{iq_end} for mesh {args.mesh}")
    suffix = "" if iq_start == 1 and iq_end == nq else f".shard_{iq_start:04d}_{iq_end:04d}"
    with calculation_directory(work_dir, create=True):
        bind_calculation(work_dir, SOURCE, model, args.mesh, create=True)
        write_input(work_dir, model, args.mesh, iq_start, iq_end, suffix, True)
        binary = work_dir / "cal_luo_full_hp_scattering"
        ensure_binary(binary, args.compiler)
        subprocess.run([str(binary)], cwd=work_dir, check=True)
    stem = "full_sunset_spectral" if model.get("scattering_definition") == "full_sunset" else "luo_four_rates"
    print(work_dir / f"{stem}{suffix}.tsv")


def command_band(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir).resolve()
    model = requested_model(args)
    nq = args.mesh * args.mesh
    with calculation_directory(work_dir, create=True):
        bind_calculation(work_dir, SOURCE, model, args.mesh, create=True)
        write_input(work_dir, model, args.mesh, 1, nq, "", False)
        binary = work_dir / "cal_luo_full_hp_scattering"
        ensure_binary(binary, args.compiler)
        subprocess.run([str(binary)], cwd=work_dir, check=True)
    band_name = "luo_bands_finite_t.tsv" if model["dispersion"] == "self_consistent_mean_field" else "luo_band_t0.tsv"
    print(work_dir / band_name)


def data_lines(path: Path) -> Iterable[str]:
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip() and not raw.startswith("#"):
            yield raw


def command_merge(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir).resolve()
    with calculation_directory(work_dir):
        model = bind_calculation(work_dir, SOURCE, mesh=args.mesh)
        merge_rates(work_dir, args.mesh, model["temperatures_K"], str(model.get("scattering_definition", "legacy_four_channel")))


def merge_rates(work_dir: Path, mesh: int, temperatures: list[float], scattering: str = "legacy_four_channel") -> None:
    full = scattering == "full_sunset"
    stem = "full_sunset_spectral" if full else "luo_four_rates"
    files = sorted(work_dir.glob(f"{stem}.shard_*.tsv"))
    if not files:
        raise RuntimeError(f"no rate shards in {work_dir}")
    rows: dict[tuple[float, int], str] = {}
    header = files[0].read_text(encoding="utf-8").splitlines()[0]
    for path in files:
        for line in data_lines(path):
            fields = line.split()
            finite_fields = fields[:13] + fields[15:] if full else fields
            if len(fields) != (31 if full else 13) or not all(math.isfinite(float(value)) for value in finite_fields):
                raise RuntimeError(f"invalid rate row in {path}: {line}")
            if full and any(float(fields[i]) <= 0 or math.isnan(float(fields[i])) for i in (13, 14)):
                raise RuntimeError(f"invalid population lifetime in {path}: {line}")
            key = (float(fields[0]), int(fields[1]))
            if key in rows:
                raise RuntimeError(f"duplicate temperature/iq row {key}")
            ix, iy = divmod(key[1] - 1, mesh)
            if key[0] not in temperatures or not 1 <= key[1] <= mesh**2 or any(
                not math.isclose(float(value), expected, rel_tol=0, abs_tol=1e-12)
                for value, expected in zip(fields[2:5], (ix / mesh, iy / mesh, 0.0))
            ):
                raise RuntimeError(f"rate temperature/q does not match frozen mesh: {path}: {line}")
            rows[key] = line
    expected = {(temperature, iq) for temperature in temperatures for iq in range(1, mesh**2 + 1)}
    if set(rows) != expected:
        missing = sorted(expected - set(rows))[:10]
        raise RuntimeError(f"incomplete shard set; first missing rows: {missing}")
    output = work_dir / f"{stem}.tsv"
    ordered = [rows[(temperature, iq)] for temperature in temperatures for iq in range(1, mesh**2 + 1)]
    output.write_text(header + "\n" + "\n".join(ordered) + "\n", encoding="utf-8")
    print(output)


def path_iq(mesh: int) -> list[int]:
    if mesh % 6:
        raise ValueError("plot mesh must be divisible by 6 so Gamma, K, and M lie on it")
    points = [(t, t) for t in range(mesh // 3 + 1)]
    points += [(mesh // 3 + s, mesh // 3 - 2 * s) for s in range(1, mesh // 6 + 1)]
    points += [(mesh // 2 - s, 0) for s in range(1, mesh // 2 + 1)]
    return [ix * mesh + iy + 1 for ix, iy in points]


def command_plot(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir).resolve()
    with calculation_directory(work_dir):
        model = bind_calculation(work_dir, SOURCE, mesh=args.mesh)
        plot_rates(args, work_dir, model)


def plot_rates(args: argparse.Namespace, work_dir: Path, model: dict[str, object]) -> None:
    if model.get("scattering_definition") == "full_sunset":
        raise ValueError("full_sunset has a spectral table; use HWHM/FWHM columns with the production analysis")
    rate_path = work_dir / "luo_four_rates.tsv"
    rows: dict[tuple[float, int], list[float]] = {}
    for line in data_lines(rate_path):
        fields = [float(value) for value in line.split()]
        rows[(fields[0], int(fields[1]))] = fields
    iq_path = path_iq(args.mesh)
    ticks = [0, args.mesh // 3, args.mesh // 2, args.mesh]
    selected_temperatures = tuple(args.temperatures or model["temperatures_K"])
    available_temperatures = {key[0] for key in rows}
    missing_temperatures = set(selected_temperatures) - available_temperatures
    if missing_temperatures:
        raise ValueError(f"temperatures absent from rate table: {sorted(missing_temperatures)}")
    width, height = 1000, 720
    colors = ("#111111",) if len(selected_temperatures) == 1 else (
        "#0072B2", "#009E73", "#E69F00", "#CC79A7"
    )
    panels: list[str] = []
    panels.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    panels.append('<rect width="100%" height="100%" fill="white"/>')
    panels.append('<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.curve{fill:none;stroke-width:2}</style>')
    dispersion = str(model.get("dispersion", "fixed_zero_temperature"))
    default_title = (
        "CrI3 Luo full-HP V1-V4, self-consistent finite-T bands"
        if dispersion == "self_consistent_mean_field"
        else "CrI3 Luo full-HP V1-V4, fixed T=0 bands"
    )
    title = args.title or default_title
    run_class = escape(str(model["run_class"]))
    panels.append(f'<text x="500" y="26" text-anchor="middle" font-size="17">{escape(title)}</text>')
    panels.append(f'<text x="500" y="48" text-anchor="middle" font-size="12">A={model["A_meV"]:.4f} meV, xi={model["xi_meV"]:.5f} meV | {run_class}</text>')

    def symlog(value: float) -> float:
        return math.copysign(math.log10(1.0 + abs(value) / 1.0e6), value)

    for channel in range(1, 5):
        col, row = (channel - 1) % 2, (channel - 1) // 2
        left, top = 75 + col * 480, 85 + row * 300
        plot_w, plot_h = 390, 225
        column = 6 + channel
        series = {
            temperature: [rows[(temperature, iq)][column] for iq in iq_path]
            for temperature in selected_temperatures
        }
        raw_values = [value for values in series.values() for value in values]
        if args.y_scale == "linear":
            transformed = raw_values
            ymin = min(0.0, min(transformed))
            ymax = max(transformed)
            if math.isclose(ymin, ymax):
                ymax = ymin + 1.0
            ymax += 0.08 * (ymax - ymin)
            magnitude = max(abs(ymin), abs(ymax))
            exponent = int(math.floor(math.log10(magnitude))) if magnitude > 0.0 else 0
            tick_scale = 10.0**exponent
            axis_label = f"rate (10^{exponent} s^-1)"
        else:
            transformed = [symlog(value) for value in raw_values]
            ymin, ymax = min(transformed), max(transformed)
            if math.isclose(ymin, ymax):
                ymin -= 1.0
                ymax += 1.0
            margin = 0.08 * (ymax - ymin)
            ymin -= margin
            ymax += margin
            tick_scale = 1.0
            axis_label = "symlog rate (s^-1)"
        panels.append(f'<text x="{left + plot_w/2:.1f}" y="{top - 12}" text-anchor="middle" font-size="16">eta({channel})</text>')
        for fraction in (0.0, 0.5, 1.0):
            y = top + plot_h * (1.0 - fraction)
            panels.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}"/>')
            label = (ymin + fraction * (ymax - ymin)) / tick_scale
            panels.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="10">{label:.2f}</text>')
        panels.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')
        panels.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>')
        for tick, label in zip(ticks, ("G", "K", "M", "G")):
            x_tick = left + plot_w * tick / args.mesh
            panels.append(f'<line class="grid" x1="{x_tick:.1f}" y1="{top}" x2="{x_tick:.1f}" y2="{top+plot_h}"/>')
            panels.append(f'<text x="{x_tick:.1f}" y="{top+plot_h+18}" text-anchor="middle" font-size="12">{label}</text>')
        panels.append(f'<text x="{left-52}" y="{top+plot_h/2:.1f}" transform="rotate(-90 {left-52} {top+plot_h/2:.1f})" text-anchor="middle" font-size="11">{axis_label}</text>')
        for color, temperature in zip(colors, selected_temperatures):
            points = []
            for index, value in enumerate(series[temperature]):
                x_point = left + plot_w * index / max(len(iq_path) - 1, 1)
                y_value = value if args.y_scale == "linear" else symlog(value)
                y_point = top + plot_h * (ymax - y_value) / (ymax - ymin)
                points.append(f"{x_point:.2f},{y_point:.2f}")
            panels.append(f'<polyline class="curve" stroke="{color}" points="{" ".join(points)}"/>')
        if channel == 1:
            for index, (color, temperature) in enumerate(zip(colors, selected_temperatures)):
                lx, ly = left + 12, top + 16 + 18 * index
                panels.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+24}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
                panels.append(f'<text x="{lx+31}" y="{ly+4}" font-size="11">{temperature:g} K</text>')
    band_label = "self-consistent finite-T bands" if dispersion == "self_consistent_mean_field" else "fixed T=0 bands"
    footer = (
        f"Linear rate axis; {band_label}."
        if args.y_scale == "linear"
        else f"Vertical labels show sign(value) log10(1+|value|/1e6); {band_label}."
    )
    panels.append(f'<text x="500" y="707" text-anchor="middle" font-size="10">{footer}</text>')
    panels.append('</svg>')
    output = work_dir / args.output
    output.write_text("\n".join(panels) + "\n", encoding="utf-8")
    print(output)


def command_plot_bands(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir).resolve()
    with calculation_directory(work_dir):
        model = bind_calculation(work_dir, SOURCE, mesh=args.mesh)
        plot_bands(args, work_dir, model)


def plot_bands(args: argparse.Namespace, work_dir: Path, model: dict[str, object]) -> None:
    band_path = work_dir / "luo_bands_finite_t.tsv"
    rows: dict[tuple[float, int], list[float]] = {}
    for line in data_lines(band_path):
        fields = [float(value) for value in line.split()]
        rows[(fields[0], int(fields[1]))] = fields
    selected_temperatures = tuple(args.temperatures or model["temperatures_K"])
    iq_path = path_iq(args.mesh)
    missing = {(temperature, iq) for temperature in selected_temperatures for iq in iq_path} - set(rows)
    if missing:
        raise ValueError(f"finite-temperature band table is incomplete: {sorted(missing)[:5]}")

    colors = ("#0072B2", "#009E73", "#E69F00", "#CC79A7")
    series: list[tuple[str, str, list[float], str]] = []
    first_temperature = selected_temperatures[0]
    series.append(("T=0 beta", "#222222", [rows[(first_temperature, iq)][9] for iq in iq_path], "5,4"))
    series.append(("T=0 alpha", "#222222", [rows[(first_temperature, iq)][8] for iq in iq_path], "5,4"))
    for temperature, color in zip(selected_temperatures, colors):
        series.append((f"{temperature:g} K beta", color, [rows[(temperature, iq)][6] for iq in iq_path], ""))
        series.append((f"{temperature:g} K alpha", color, [rows[(temperature, iq)][5] for iq in iq_path], ""))

    width, height = 1000, 650
    left, top, plot_w, plot_h = 90, 75, 700, 500
    ymax = 1.06 * max(value for _, _, values, _ in series for value in values)
    ymin = 0.0
    panels = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.curve{fill:none;stroke-width:2}</style>',
        '<text x="500" y="30" text-anchor="middle" font-size="18">Luo mean-field finite-temperature magnon bands</text>',
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * (1.0 - fraction)
        value = ymin + fraction * (ymax - ymin)
        panels.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}"/>')
        panels.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-size="11">{value:.1f}</text>')
    panels.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')
    panels.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>')
    for tick, label in zip((0, args.mesh // 3, args.mesh // 2, args.mesh), ("G", "K", "M", "G")):
        x = left + plot_w * tick / args.mesh
        panels.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_h}"/>')
        panels.append(f'<text x="{x:.2f}" y="{top+plot_h+22}" text-anchor="middle" font-size="13">{label}</text>')
    panels.append(f'<text x="25" y="{top+plot_h/2:.2f}" transform="rotate(-90 25 {top+plot_h/2:.2f})" text-anchor="middle" font-size="13">Energy (meV)</text>')
    for label, color, values, dash in series:
        points = []
        for index, value in enumerate(values):
            x = left + plot_w * index / max(len(iq_path)-1, 1)
            y = top + plot_h * (ymax-value) / (ymax-ymin)
            points.append(f"{x:.2f},{y:.2f}")
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        panels.append(f'<polyline class="curve" stroke="{color}"{dash_attr} points="{" ".join(points)}"/>')
    legend_x, legend_y = 815, 95
    panels.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+28}" y2="{legend_y}" stroke="#222" stroke-width="2" stroke-dasharray="5,4"/>')
    panels.append(f'<text x="{legend_x+36}" y="{legend_y+4}" font-size="12">T=0 baseline</text>')
    for index, (temperature, color) in enumerate(zip(selected_temperatures, colors), start=1):
        y = legend_y + 25*index
        panels.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+28}" y2="{y}" stroke="{color}" stroke-width="2"/>')
        panels.append(f'<text x="{legend_x+36}" y="{y+4}" font-size="12">{temperature:g} K</text>')
    panels.append('<text x="500" y="632" text-anchor="middle" font-size="11">Both acoustic beta and optical alpha branches; finite-T curves use the same color.</text>')
    panels.append('</svg>')
    output = work_dir / args.output
    output.write_text("\n".join(panels) + "\n", encoding="utf-8")
    print(output)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("config")
    prepare.add_argument("--work-dir", required=True)
    prepare.add_argument("--mesh", type=int, default=6)
    prepare.set_defaults(func=command_prepare)

    compile_cmd = sub.add_parser("compile")
    compile_cmd.add_argument("--work-dir", required=True)
    compile_cmd.add_argument("--compiler", default="gfortran")
    compile_cmd.set_defaults(func=command_compile)

    run = sub.add_parser("run")
    run.add_argument("config")
    run.add_argument("--work-dir", required=True)
    run.add_argument("--mesh", type=int, default=6)
    run.add_argument("--iq-start", type=int, default=1)
    run.add_argument("--iq-end", type=int, default=0)
    run.add_argument("--compiler", default="gfortran")
    run.set_defaults(func=command_run)

    band = sub.add_parser("band")
    band.add_argument("config")
    band.add_argument("--work-dir", required=True)
    band.add_argument("--mesh", type=int, default=24)
    band.add_argument("--compiler", default="gfortran")
    band.set_defaults(func=command_band)

    merge = sub.add_parser("merge")
    merge.add_argument("--work-dir", required=True)
    merge.add_argument("--mesh", type=int, required=True)
    merge.set_defaults(func=command_merge)

    plot = sub.add_parser("plot")
    plot.add_argument("--work-dir", required=True)
    plot.add_argument("--mesh", type=int, required=True)
    plot.add_argument("--y-scale", choices=("linear", "symlog"), default="symlog")
    plot.add_argument("--temperatures", type=float, nargs="+")
    plot.add_argument("--output", default="rates_v1_v4.svg")
    plot.add_argument("--title")
    plot.set_defaults(func=command_plot)

    plot_bands = sub.add_parser("plot-bands")
    plot_bands.add_argument("--work-dir", required=True)
    plot_bands.add_argument("--mesh", type=int, required=True)
    plot_bands.add_argument("--temperatures", type=float, nargs="+")
    plot_bands.add_argument("--output", default="bands_finite_t.svg")
    plot_bands.set_defaults(func=command_plot_bands)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
