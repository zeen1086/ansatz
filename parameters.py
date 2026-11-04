"""Read the one authoritative parameter file; generate build-time Fortran values."""
from pathlib import Path
import math
import os
import shlex
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 hosts may use an already installed tomli.

ROOT = Path(__file__).resolve().parent
PATH_KEYS = {"func_dir", "nscf_dir", "rvr_dir", "super_h_dir", "pseudo_dir",
             "native_data_root", "qe_root", "mkl_include", "native_libpw", "native_gradcorr"}


def load(path: Path | None = None) -> dict:
    path = Path(path or os.environ.get("ANSATZ_PARAMETER", ROOT / "parameter")).resolve()
    sections = tomllib.loads(path.read_text())
    values = {}
    for section in sections.values():
        for key, value in section.items():
            if key in values:
                raise ValueError(f"duplicate global parameter: {key}")
            values[key] = value
    positive = ("nwan", "nspin", "nr1", "nr2", "nr3", "nk1", "nk2", "nk3",
                "nx", "ny", "nz", "period_x", "period_y", "period_z", "n_file", "n_kind",
                "band_energy_points", "dos_points", "threads", "rvr_chunk", "rvr_channel_block")
    for key in positive:
        if type(values[key]) is not int or values[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key, value in values.items():
        for number in value if isinstance(value, list) else [value]:
            if isinstance(number, float) and not math.isfinite(number):
                raise ValueError(f"{key} must be finite")
    if values["nspin"] not in (1, 2) or values["n_func"] not in (18, 36):
        raise ValueError("the shared spin assembly supports nspin=1/2 and n_func=18/36")
    if values["fit_nspinor"] != 6 or values["split_nspinor"] != 2:
        raise ValueError("the supplied CrI3 site maps require fit_nspinor=6 and split_nspinor=2")
    if not 1 <= values["wan_min"] <= values["wan_max"] <= values["nwan"]:
        raise ValueError("require 1 <= wan_min <= wan_max <= nwan")
    if (not 1 <= values["first_band"] <= values["last_band"] <= values["nbnd"]
            or values["num_bands"] != values["last_band"]-values["first_band"]+1
            or values["num_bands"] < values["nwan"]):
        raise ValueError("Wannier band window, num_bands, nwan and nbnd are inconsistent")
    for axis, k in zip("xyz", ("nk1", "nk2", "nk3")):
        extent = values[f"convert_size_{axis}"]
        if type(extent) is not int or extent < 0 or 2*extent+1 > values[k]:
            raise ValueError(f"invalid Wannier support on {axis}")
        if values[f"n{axis}"] % values[f"period_{axis}"]:
            raise ValueError(f"n{axis} must be divisible by period_{axis}")
    if (not values["broadening"] or any(x <= 0 for x in values["broadening"])
            or not 1 <= values["selected_eta"] <= len(values["broadening"])):
        raise ValueError("positive broadening and a valid selected_eta are required")
    if not 0 < values["weight_floor"] < 1 or values["spin_s"] <= 0:
        raise ValueError("invalid weight_floor or spin_s")
    if values["angle_degrees"] <= 0 or values["cell_residual_limit"] <= 0:
        raise ValueError("angle_degrees and cell_residual_limit must be positive")
    if len(values["channel_mask"]) != 6 or any(type(x) is not int or abs(x) > 1 for x in values["channel_mask"]):
        raise ValueError("channel_mask needs six integer values in -1..1")
    if len(values["channel_source"]) != 6 or any(type(x) is not int or not 1 <= x <= 6 for x in values["channel_source"]):
        raise ValueError("channel_source needs six integer values in 1..6")
    if len(values["trace_r"]) != 3 or any(type(x) is not int for x in values["trace_r"]):
        raise ValueError("trace_r needs three integers")
    if not isinstance(values["prefix"], str) or not values["prefix"].replace("_", "").isalnum():
        raise ValueError("prefix must contain letters, digits or underscores")
    for key in ("ef", "eta", "a0", "celldm3", "ecutwfc", "conv_thr", "fit_tolerance"):
        if isinstance(values[key], bool) or not isinstance(values[key], (int, float)):
            raise ValueError(f"{key} must be numeric")
    if values["a0"] <= 0 or values["celldm3"] <= 0 or values["conv_thr"] <= 0:
        raise ValueError("lattice constants and conv_thr must be positive")
    for key in PATH_KEYS:
        values[key] = str((path.parent / values[key]).resolve())
    return values


def fortran_module(values: dict) -> str:
    lines = ["! Generated from parameter; do not edit.", "module parameter_mod", "  implicit none"]
    for key, value in values.items():
        name = "p_" + key
        if isinstance(value, list):
            integers = all(type(x) is int for x in value)
            kind = "integer" if integers else "real(8)"
            lines.append(f"  {kind}, parameter :: {name}({len(value)}) = [ &")
            lines.extend((str(x) if integers else f"{float(x):.17e}_8") + (", &" if i < len(value)-1 else " ]")
                         for i, x in enumerate(value))
        elif isinstance(value, bool):
            lines.append(f"  logical, parameter :: {name} = .{str(value).lower()}.")
        elif isinstance(value, int):
            lines.append(f"  integer, parameter :: {name} = {value}")
        elif isinstance(value, float):
            lines.append(f"  real(8), parameter :: {name} = {value:.17e}_8")
        else:
            text = value + ("/" if key in PATH_KEYS else "")
            chunks = [text[i:i+64].replace("'", "''") for i in range(0, len(text), 64)] or [""]
            lines.append(f"  character(len=*), parameter :: {name} = &")
            lines.extend("    '" + chunk + "'" + (" // &" if i < len(chunks)-1 else "")
                         for i, chunk in enumerate(chunks))
    lines += ["contains", "  subroutine read_row_index(row)", "    integer, intent(out) :: row",
              "    integer :: ios", "    character(len=32) :: argument",
              "    call get_command_argument(1, argument)",
              "    read(argument, *, iostat=ios) row",
              "    if (ios /= 0) error stop 'First argument must be a Wannier/RVR row index'",
              "    if (row < 1 .or. row > p_nwan*2/p_nspin) error stop 'row index out of range'",
              "  end subroutine read_row_index", "  function row_text(row) result(value)",
              "    integer, intent(in) :: row", "    character(len=20) :: value",
              "    write(value, '(I0)') row", "  end function row_text"]
    return "\n".join(lines + ["end module parameter_mod", ""])


if __name__ == "__main__":
    values = load()
    if sys.argv[1:] == ["--shell"]:
        for key, value in values.items():
            if not isinstance(value, list):
                print(f"{key.upper()}={shlex.quote(str(value))}")
    else:
        for key in sys.argv[1:]:
            print(values[key])
