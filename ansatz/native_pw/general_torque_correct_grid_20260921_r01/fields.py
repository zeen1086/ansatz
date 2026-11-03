#!/usr/bin/env python3
"""Six local-axis derivatives of the full potential and its FM-scalar ablation."""
from pathlib import Path
import argparse
import importlib.util
import json
import math
import struct
import subprocess
import sys

import numpy as np

R = Path(__file__).resolve().parent
P = R.parent
spec = importlib.util.spec_from_file_location('potential_evaluator', P / 'full_potential_evaluator_20260921_r03/evaluate.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
g = ev.g


def rotate(spin, axis, angle):
    return spin * math.cos(angle) + np.cross(axis, spin) * math.sin(angle) + axis * np.dot(axis, spin) * (1 - math.cos(angle))


def write_field(path, field, shape):
    assert field.shape == (math.prod(shape), 4) and np.isfinite(field).all()
    with path.open('xb') as f:
        f.write(struct.pack('<7i', 16, *shape, 4, 16, field.nbytes))
        for column in field.T:
            f.write(np.asarray(column, dtype='<f8').tobytes())
        f.write(struct.pack('<i', field.nbytes))
    actual, actual_shape = g.read(path)
    assert actual_shape == shape and ev.max_difference(actual, field) == 0.


def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def generate_native(cell, spins, directory):
    spinfile = directory.with_suffix('.spins.json')
    write_json(spinfile, spins.tolist())
    subprocess.run([sys.executable, str(g.R / 'generate_potential.py'), cell,
                    str(spinfile), str(directory)], check=True)
    assert 'CRI3_VARIATIONAL_FIELD_EXPORT_COMPLETE' in (directory / 'potential.out').read_text()
    return directory


def site_maps(cell):
    matrix, tiles, _ = g.CELLS[cell]
    inverse = np.linalg.inv(matrix)
    result = []
    for tile in tiles:
        sites = []
        for tx, ty, b in g.CHANNELS:
            hits = [j for j, t in enumerate(tiles)
                    if np.allclose((np.array([tx, ty]) + tile - t) @ inverse,
                                   np.rint((np.array([tx, ty]) + tile - t) @ inverse), atol=1e-10)]
            assert len(hits) == 1
            sites.append(2 * hits[0] + b)
        result.append((tile, sites))
    return result


def analytic_derivatives(cell, site, de, vfunc, density_kernel):
    dv = np.zeros((3 * g.N, 4))
    drho = np.zeros_like(dv)
    for tile, sites in site_maps(cell):
        channels = [c for c, source_site in enumerate(sites) if source_site == site]
        for start in range(0, g.N, 32768):
            ids = np.arange(start, min(start + 32768, g.N))
            ix = g.index(cell, tile, ids)
            magnetic = np.zeros((len(ids), 3))
            for channel in channels:
                magnetic += np.einsum('ifs,s->if', vfunc[ids, :, channel, :], de)
            dv[ix, 1:] = magnetic
            drho[ix, 1:] = density_kernel[ids][:, channels].sum(axis=1)[:, None] * de
    if cell == 'sd':
        dv, drho = g.lift_sd(dv), g.lift_sd(drho)
    assert dv.shape == drho.shape == (math.prod(g.native_shape(cell)), 4)
    return dv, drho


def self_check():
    spin = np.array([.02, .01, math.sqrt(1 - .02 ** 2 - .01 ** 2)])
    axis = np.array([0., 1., 0.])
    h = 1e-5
    de = np.cross(axis, spin)
    assert np.max(abs((rotate(spin, axis, h) - rotate(spin, axis, -h)) / (2 * h) - de)) < 1e-9
    assert abs(np.linalg.norm(rotate(spin, axis, .1)) - 1) < 1e-14

    def energy(angle, frozen):
        e = rotate(spin, axis, angle)
        rho = np.r_[.7, e * [.2, .3, .4]]
        v = np.array([.7 + .3 * rho[3] ** 2, 2 * rho[1], rho[2] + .05 * rho[3], .8 * rho[3]])
        if frozen:
            v[0] = .7 + .3 * .4 ** 2
        band = v[0] - np.linalg.norm(v[1:] + [.2, 0, 0])
        return band - rho @ v + .5 * (rho @ rho)

    rho = np.r_[.7, spin * [.2, .3, .4]]
    drho = np.r_[0., de * [.2, .3, .4]]
    for frozen in (False, True):
        v = np.array([.7 + .3 * rho[3] ** 2, 2 * rho[1], rho[2] + .05 * rho[3], .8 * rho[3]])
        dv = np.array([.6 * rho[3] * drho[3], 2 * drho[1], drho[2] + .05 * drho[3], .8 * drho[3]])
        if frozen:
            v[0], dv[0] = .7 + .3 * .4 ** 2, 0.
        field = v[1:] + [.2, 0, 0]
        rho_out = np.r_[1., -field / np.linalg.norm(field)]
        tau = -(rho_out - rho) @ dv + (v - rho) @ drho
        assert abs(tau + (energy(h, frozen) - energy(-h, frozen)) / (2 * h)) < 1e-9
    print('PASS: Rodrigues derivative and dynamic/frozen scalar total-energy torque identities')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cell', choices=g.CELLS)
    parser.add_argument('spins_json', type=Path)
    parser.add_argument('axes_json', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--native-dir', type=Path)
    args = parser.parse_args()
    spins = np.asarray(json.loads(args.spins_json.read_text()), float)
    axes = np.asarray(json.loads(args.axes_json.read_text()), float)
    for vectors in (spins, axes):
        assert vectors.shape == (6, 3) and np.isfinite(vectors).all()
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-10, rtol=0)
    h = math.radians(.01)
    assert np.min(spins[:, 2]) >= math.cos(math.radians(3)) - 1e-12
    for site in range(6):
        for sign in (-1, 1):
            assert rotate(spins[site], axes[site], sign * h)[2] >= math.cos(math.radians(3)) - 1e-12, 'Derivative stencil exceeds declared3degree domain.'
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    dynamic, frozen = out / 'dynamic', out / 'frozen'
    command = [sys.executable, str(ev.R / 'evaluate.py'), args.cell, str(args.spins_json.resolve()), str(dynamic)]
    if args.native_dir:
        command += ['--native-dir', str(args.native_dir.resolve())]
    subprocess.run(command, check=True)
    rho, shape = g.read(dynamic / 'DENSITY_ANSATZ.bin')
    native, native_shape = g.read(dynamic / 'native/FROZEN_HXC.bin')
    field, field_shape = g.read(dynamic / 'FROZEN_HXC.bin')
    assert shape == native_shape == field_shape == g.native_shape(args.cell)
    energy = json.loads((dynamic / 'ENERGY.json').read_text())
    dv_grid = energy['voxel_volume_bohr3']
    fm_spins = np.tile([0., 0., 1.], (6, 1))
    g.evaluate(args.cell, fm_spins, out / 'FM_DENSITY_REQUESTED.bin')
    fm_rho, fm_shape = g.read(out / 'FM_DENSITY_REQUESTED.bin')
    fm_directory = g.R / ('FM_POTENTIAL_' + args.cell)
    fm_reused = False
    if (fm_directory / 'DENSITY_ANSATZ.bin').is_file() and (fm_directory / 'FROZEN_HXC.bin').is_file():
        old_rho, old_shape = g.read(fm_directory / 'DENSITY_ANSATZ.bin')
        fm_reused = old_shape == fm_shape and ev.max_difference(fm_rho, old_rho) == 0.
    if fm_reused:
        (out / 'fm_native').symlink_to(fm_directory, target_is_directory=True)
    else:
        fm_directory = generate_native(args.cell, fm_spins, out / 'fm_native')
    fm_field, fm_field_shape = g.read(fm_directory / 'FROZEN_HXC.bin')
    check_rho, check_shape = g.read(fm_directory / 'DENSITY_ANSATZ.bin')
    assert shape == fm_field_shape == check_shape and ev.max_difference(check_rho, fm_rho) == 0.
    frozen.mkdir()
    frozen_field = np.array(field)
    frozen_field[:, 0] = fm_field[:, 0]
    write_field(frozen / 'FROZEN_HXC.bin', frozen_field, shape)
    (frozen / 'DENSITY_ANSATZ.bin').symlink_to(dynamic / 'DENSITY_ANSATZ.bin')
    (frozen / 'native').symlink_to(dynamic / 'native', target_is_directory=True)
    frozen_energy = dict(energy)
    frozen_energy['model'] = 'Same trial density and serialized B42; scalarHXC fixed at FM.'
    frozen_energy['U_tilde_Ry'] = ev.integral(rho, frozen_field, dv_grid)
    frozen_energy['electronic_nonband_Ry'] = math.fsum([energy['EH_plus_Exc_minus_Excc_Ry'], -frozen_energy['U_tilde_Ry']])
    frozen_energy['nonband_with_fixed_ion_Ry'] = math.fsum([frozen_energy['electronic_nonband_Ry'], energy['fixed_ion_Ry']])
    frozen_energy.update(fixed_scalar_native_directory=str(fm_directory), fm_density_verified=True)
    write_json(frozen / 'ENERGY.json', frozen_energy)
    vfunc = np.memmap(g.R / 'V_func_tmp', dtype='<f8', mode='r', offset=28, shape=(g.N, 3, 14, 3), order='F')
    density_kernel = np.load(g.R / 'density_kernel.npy', mmap_mode='r')
    derivatives = out / 'fresh_scalar_derivatives'
    derivatives.mkdir()
    rows = {'dynamic': [], 'frozen': []}
    generated = []
    for site in range(6):
        derivative, drho = analytic_derivatives(args.cell, site, np.cross(axes[site], spins[site]), vfunc, density_kernel)
        sides = []
        density_sides = []
        for sign, label in [(-1, 'minus'), (1, 'plus')]:
            rotated = spins.copy()
            rotated[site] = rotate(spins[site], axes[site], sign * h)
            directory = generate_native(args.cell, rotated, derivatives / f'site{site}_{label}')
            generated.append({'site': site, 'sign': sign, 'angle_step_deg': sign * .01,
                              'spins': rotated.tolist(), 'native_directory': str(directory)})
            scalar, scalar_shape = g.read(directory / 'FROZEN_HXC.bin')
            side_rho, side_shape = g.read(directory / 'DENSITY_ANSATZ.bin')
            assert scalar_shape == side_shape == shape
            sides.append(scalar)
            density_sides.append(side_rho)
        derivative[:, 0] = (sides[1][:, 0] - sides[0][:, 0]) / (2 * h)
        # The density ansatz is linear in spin, so Rodrigues central differences equal sinc(h)*drho.
        density_fd = (density_sides[1] - density_sides[0]) / (2 * h)
        density_bridge = ev.max_difference(density_fd, drho * math.sin(h) / h)
        assert density_bridge < 1e-9, ('Analytic density derivative mapping mismatch', site, density_bridge)
        del density_fd
        for label, directory, value in [('dynamic', dynamic, field), ('frozen', frozen, frozen_field)]:
            if label == 'frozen':
                derivative[:, 0] = 0.
            write_field(directory / f'DERIVATIVE_{site}.bin', derivative, shape)
            input_work = ev.integral(rho, derivative, dv_grid)
            mismatch = ev.integral(value - native, drho, dv_grid)
            rows[label].append({'site': site, 'input_work': input_work,
                                'potential_mismatch_work': mismatch, 'total': math.fsum([input_work, mismatch]),
                                'analytic_density_derivative_bridge_max_error': density_bridge})
        print('GENERATED_SITE', site, flush=True)
    for label, directory in [('dynamic', dynamic), ('frozen', frozen)]:
        write_json(directory / 'COUNTERTERMS.json', {
            'physical_release': False, 'cell': args.cell, 'spins': spins.tolist(), 'axes': axes.tolist(),
            'model': label, 'derivative_step_deg': .01, 'voxel_volume_bohr3': dv_grid,
            'rows_Ry': rows[label], 'counterterm_torque_Ry': [row['total'] for row in rows[label]],
            'definition': 'tau_total=tau_band+counterterm_torque_Ry; tau_band=-Tr(rho_out*dH/dtheta).',
            'counterterm_definition': 'int(rho_in*dVtilde)+int((Vtilde-Vnative)*drho_in); scalar density derivative is zero.',
            'fm_scalar_source': str(fm_directory) if label == 'frozen' else None,
            'fresh_scalar_derivative_directory': str(derivatives),
            'limits': ['No electronic diagonalization or total torque in this generator.',
                       'Fresh centered scalar derivative uses0.01degree; no step-convergence claim.',
                       'General coordinate support is callable, not an independent physics acceptance.']})
    write_json(out / 'GENERATED.json', {'physical_release': False, 'cell': args.cell,
               'spins': spins.tolist(), 'axes': axes.tolist(), 'native_displacements': generated,
               'fm_native_directory': str(fm_directory), 'fm_native_reused': fm_reused,
               'dynamic_directory': str(dynamic), 'frozen_directory': str(frozen)})
    print('GENERAL_TORQUE_FIELDS_COMPLETE', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-check']:
        self_check()
    else:
        main()
