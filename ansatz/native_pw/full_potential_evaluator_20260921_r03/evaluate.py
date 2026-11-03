#!/usr/bin/env python3
"""Callable tangent magnetic + nonlinear scalar potential; no torque release."""
from pathlib import Path
import argparse
import importlib.util
import json
import math
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

R = Path(__file__).resolve().parent
P = R.parent
spec = importlib.util.spec_from_file_location('model', P / 'full_spatial_candidate_20260921_r03/model.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def max_difference(a, b):
    assert a.shape == b.shape
    return max(float(np.max(abs(a[i:i + 32768] - b[i:i + 32768])))
               for i in range(0, len(a), 32768))


def integral(a, b, dv):
    assert a.shape == b.shape
    return math.fsum(float(np.sum(a[i:i + 32768] * b[i:i + 32768]))
                     for i in range(0, len(a), 32768)) * dv


def magnetic_field(cell, spins, vfunc, fm):
    """Assemble serialized B42 and its fixed offset before the shared SD lift."""
    field = np.zeros((3 * g.N, 4))
    matrix, tiles, _ = g.CELLS[cell]
    inverse = np.linalg.inv(matrix)
    for tile in tiles:
        sites = []
        for tx, ty, b in g.CHANNELS:
            matches = [j for j, t in enumerate(tiles)
                       if np.allclose((np.array([tx, ty]) + tile - t) @ inverse,
                                      np.rint((np.array([tx, ty]) + tile - t) @ inverse), atol=1e-10)]
            assert len(matches) == 1
            sites.append(2 * matches[0] + b)
        for start in range(0, g.N, 32768):
            ids = np.arange(start, min(start + 32768, g.N))
            values = np.einsum('ifcs,cs->if', vfunc[ids], spins[sites], optimize=True)
            values[:, :2] += fm[ids, 1:3]
            field[g.index(cell, tile, ids), 1:] = values
    return g.lift_sd(field) if cell == 'sd' else field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cell', choices=g.CELLS)
    parser.add_argument('spins_json', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--native-dir', type=Path, help='Reuse matching density, nonlinear HXC and ENERGY_COMPONENTS.dat; no QE invocation.')
    args = parser.parse_args()
    spins_file = args.spins_json.resolve()
    spins = np.asarray(json.loads(spins_file.read_text()), float)
    assert spins.shape == (6, 3) and np.isfinite(spins).all()
    assert np.allclose(np.linalg.norm(spins, axis=1), 1, atol=1e-10, rtol=0)
    assert np.min(spins[:, 2]) >= math.cos(math.radians(3)) - 1e-12
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.native_dir:
        native_dir = args.native_dir.resolve(strict=True)
        g.evaluate(args.cell, spins, out / 'DENSITY_ANSATZ.bin')
        (out / 'native').symlink_to(native_dir, target_is_directory=True)
    else:
        native_dir = out / 'native'
        subprocess.run([sys.executable, str(g.R / 'generate_potential.py'), args.cell,
                        str(spins_file), str(native_dir)], check=True)
        shutil.copy2(native_dir / 'DENSITY_ANSATZ.bin', out / 'DENSITY_ANSATZ.bin')
    rho, shape = g.read(out / 'DENSITY_ANSATZ.bin')
    native_rho, native_shape = g.read(native_dir / 'DENSITY_ANSATZ.bin')
    assert native_shape == shape
    density_error = max_difference(rho, native_rho)
    assert density_error == 0., ('Requested spins do not reproduce reused native density', density_error)
    native, native_shape = g.read(native_dir / 'FROZEN_HXC.bin')
    assert native_shape == shape and np.isfinite(native).all()
    assert shape == g.native_shape(args.cell), ('Incorrect native grid', shape)
    reference = g.reference_tmp(args.cell) / 'CrI3.save/data-file-schema.xml'
    xml = ET.parse(reference)
    cell = np.array([np.fromstring(c.text, sep=' ') for c in xml.find('.//output/atomic_structure/cell')])
    dv = abs(np.linalg.det(cell)) / len(rho)
    source_xml = native_dir / 'tmp/CrI3.save/data-file-schema.xml'
    if source_xml.is_file():
        source_cell = np.array([np.fromstring(c.text, sep=' ') for c in ET.parse(source_xml).find('.//output/atomic_structure/cell')])
        assert np.allclose(source_cell, cell, atol=1e-10, rtol=0), 'Reused native cell differs.'
    vfunc_path = g.R / 'V_func_tmp'
    size = g.N * 3 * 14 * 3 * 8
    with vfunc_path.open('rb') as f:
        assert struct.unpack('<7i', f.read(28)) == (0, 0, 0, 0, 0, 0, size)
        f.seek(-12, 2)
        assert struct.unpack('<3i', f.read(12)) == (size, 0, 0)
    assert vfunc_path.stat().st_size == size + 40
    vfunc = np.memmap(vfunc_path, dtype='<f8', mode='r', offset=28, shape=(g.N, 3, 14, 3), order='F')
    fm = np.load(g.R / 'potential_FM.npy', mmap_mode='r')
    field = np.array(native)
    magnetic = magnetic_field(args.cell, spins, vfunc, fm)
    assert magnetic.shape == field.shape
    field[:, 1:] = magnetic[:, 1:]
    del magnetic
    assert np.isfinite(field).all() and max_difference(field[:, :1], native[:, :1]) == 0.
    with (out / 'FROZEN_HXC.bin').open('xb') as f:
        f.write(struct.pack('<7i', 16, *shape, 4, 16, field.nbytes))
        for column in field.T:
            f.write(np.asarray(column, dtype='<f8').tobytes())
        f.write(struct.pack('<i', field.nbytes))
    assert max_difference(g.read(out / 'FROZEN_HXC.bin')[0], field) == 0.
    energies = np.loadtxt(native_dir / 'ENERGY_COMPONENTS.dat')
    assert energies.shape == (6,) and np.isfinite(energies).all()
    eh, exc, exc_core, native_u_reported, charge, vtxc = map(float, energies)
    native_u = integral(rho, native, dv)
    assert abs(native_u - native_u_reported) < 1e-8, ('Native energy/density/potential mismatch', native_u, native_u_reported)
    assert abs(float(np.sum(rho[:, 0])) * dv - charge) < 1e-7
    u_tilde = integral(rho, field, dv)
    hxc = math.fsum([eh, exc, -exc_core])
    nonband = math.fsum([hxc, -u_tilde])
    ion = 2 * float(xml.findtext('.//output/total_energy/ewald'))
    result = {'physical_release': False, 'torque_interface_complete': False,
              'model': 'Serialized 42-function magnetic tangent plus native nonlinear scalar HXC of the same trial density.',
              'SD_mapping': 'Physical-G lift of complete assembled B42 plus fixed transverse offset; scalar HXC evaluated on fine grid.',
              'cell': args.cell, 'spins': spins.tolist(), 'native_directory': str(native_dir),
              'serialized_magnetic_potential': str(vfunc_path), 'fixed_transverse_offset': str(g.R / 'potential_FM.npy'),
              'reference_xml': str(reference), 'grid': list(shape), 'voxel_volume_bohr3': dv,
              'native_reused': bool(args.native_dir), 'native_density_max_difference': density_error,
              'U_native_Ry': native_u, 'U_native_reported_Ry': native_u_reported, 'U_tilde_Ry': u_tilde,
              'EH_plus_Exc_minus_Excc_Ry': hxc, 'electronic_nonband_Ry': nonband,
              'fixed_ion_Ry': ion, 'nonband_with_fixed_ion_Ry': math.fsum([nonband, ion]),
              'energy_definition': 'Etotal_Ry = Eband_Ry[Vtilde] + nonband_with_fixed_ion_Ry; Eband includes all occupied states with native k weights.',
              'limits': ['No electronic diagonalization or torque calculation in this wrapper.',
                         'Calls allowed for six unit spins within3degrees of FM in s31,s13,sd; this is an interface bound, not full-range physical validation.',
                         'Native source retained under native/; no constraint field or penalty energy is added.',
                         'Validation scope remains the existing s31 single-site benchmarks; no new physical acceptance.']}
    with (out / 'ENERGY.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
