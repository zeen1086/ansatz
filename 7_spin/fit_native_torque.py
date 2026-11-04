#!/usr/bin/env python3
"""Read native three-term torques and fit a declared J1--J3 Hamiltonian."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parameters import load
PARAM = load()

import numpy as np

ROOT = Path(__file__).resolve().parent
CONVENTION = "H=+sum_<ij> J_ij S_i.S_j"
RY_MEV = 13605.693122994


def native_root(project):
    return project / '7_spin/native_pw'


def geometry_module(project):
    path = native_root(project) / 'three_cell_torque_20260920_r01/analyze.py'
    spec = importlib.util.spec_from_file_location('native_geometry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def read_torque(path):
    row = json.loads(path.read_text())
    spins, axes = (np.asarray(row[key], float) for key in ('spins', 'axes'))
    if spins.shape != (6, 3) or axes.shape != (6, 3):
        raise ValueError(f'{path}: expected six ordered spins and rotation axes')
    for value in (spins, axes):
        if not np.isfinite(value).all() or not np.allclose(np.linalg.norm(value, axis=1), 1, atol=1e-10, rtol=0):
            raise ValueError(f'{path}: non-finite or non-unit coordinates')
    rows = row['rows_Ry']
    if [r['site'] for r in rows] != list(range(6)):
        raise ValueError(f'{path}: incomplete, duplicate or reordered site rows')
    terms = np.array([[r[key] for key in ('band', 'input_work', 'potential_mismatch_work', 'total')] for r in rows], float)
    torque = np.asarray(row['site_torque_Ry_per_rad'], float)
    if torque.shape != (6,) or not np.isfinite(terms).all() or not np.isfinite(torque).all():
        raise ValueError(f'{path}: non-finite or incomplete torque')
    if not np.allclose(terms[:, :3].sum(axis=1), torque, atol=1e-12, rtol=0) or not np.allclose(terms[:, 3], torque, atol=1e-12, rtol=0):
        raise ValueError(f'{path}: three-term torque accounting mismatch')
    gap = float(row.get('global_indirect_gap_Ry', row.get('gap_Ry', float('nan'))))
    if (not math.isfinite(gap) or gap <= 0 or row.get('occupied_states_per_k') != PARAM['native_occupied']
            or row.get('n_kpoints', 0) < 1 or not math.isclose(float(row.get('k_weight_sum', 0)), 1, abs_tol=1e-10)):
        raise ValueError(f'{path}: incomplete insulating occupied-state readout')
    return spins, axes, torque, row


def torque_design(neighbors, spins, axes, spin_s):
    # Each undirected bond occurs once in H. The directed row lists its incident bonds.
    angular = np.einsum('ik,ijk->ij', axes, np.cross(spins[:, None, :], spins[None, :, :]))
    return -spin_s**2 * np.einsum('ijs,ij->is', neighbors, angular) / RY_MEV


def fit(config_path, project):
    config = json.loads(config_path.read_text())
    base = config_path.parent
    for name, expected in config['input_sha256'].items():
        if sha256((base / name).resolve()) != expected:
            raise ValueError(f'input changed: {name}')
    geometry = geometry_module(project)
    spin_s = float(config['spin_S'])
    if not math.isfinite(spin_s) or spin_s <= 0:
        raise ValueError('spin_S must be positive')
    designs, targets, observations = [], [], []
    for pair in config['pairs']:
        records = [read_torque(base / pair[side]) for side in ('plus', 'minus')]
        matrix, sites, _ = geometry.geometry(pair['cell'])
        neighbors = geometry.counts(matrix, sites)[:, :, :3]
        h = math.radians(float(pair['step_degrees']))
        if not math.isfinite(h) or h <= 0:
            raise ValueError('positive finite angular step required')
        if not np.allclose(records[0][1], records[1][1], atol=1e-12, rtol=0):
            raise ValueError('rotation axes differ across endpoints')
        for side, record in zip(('plus', 'minus'), records):
            if record[3].get('case') != side or record[3].get('cell', pair['cell']) != pair['cell']:
                raise ValueError('endpoint case or cell mismatch')
        design = (torque_design(neighbors, *records[0][:2], spin_s) - torque_design(neighbors, *records[1][:2], spin_s)) / (2*h)
        response = (records[0][2] - records[1][2]) / (2*h)
        selected = pair['response_sites']
        if not selected or len(set(selected)) != len(selected) or any(i not in range(6) for i in selected):
            raise ValueError('invalid response_sites')
        designs.extend(design[selected]); targets.extend(response[selected])
        observations.append({'cell': pair['cell'], 'Phi_tau_Ry_per_rad2': response.tolist(), 'response_sites': selected})
    design, target = np.asarray(designs), np.asarray(targets)
    j, _, rank, singular = np.linalg.lstsq(design, target, rcond=1e-10)
    residual = target - design @ j
    cell_fits = []
    offset = 0
    for observation in observations:
        count = len(observation['response_sites'])
        block = slice(offset, offset + count)
        cell_fits.append({'cell': observation['cell'], 'rows': count,
                          'rank': int(np.linalg.matrix_rank(design[block], tol=1e-10)),
                          'relative_residual': float(np.linalg.norm(residual[block]) / max(np.linalg.norm(target[block]), 1e-300))})
        offset += count
    result = {'physical_release': False, 'exchange_convention': CONVENTION, 'spin_S': spin_s,
              'scope': 'Conditional isotropic J1--J3 finite-angle torque fit; higher shells and non-Heisenberg terms excluded',
              'rank': int(rank), 'singular_values': singular.tolist(),
              'J_meV': dict(zip(('J1', 'J2', 'J3'), j.tolist())) if rank == 3 else None,
              'observations': observations, 'cell_fits': cell_fits, 'residual_Ry_per_rad2': residual.tolist(),
              'relative_residual': float(np.linalg.norm(residual) / max(np.linalg.norm(target), 1e-300)),
              'source_config': str(config_path), 'source_config_sha256': sha256(config_path),
              'input_sha256': config['input_sha256'], 'anisotropy': 'NOT_EXTRACTED',
              'uncertainty': 'NOT_ESTABLISHED; fit residual is not physical uncertainty'}
    if rank == 2:
        _, _, vh = np.linalg.svd(design, full_matrices=True)
        combinations = np.array([[1., 0, 1], [0, 1, 0]])
        if np.max(abs(combinations @ vh[2:].T)) < 1e-8:
            result['identified_combinations_meV'] = dict(zip(('J1_plus_J3', 'J2'), (combinations @ j).tolist()))
    return result


def connect_full14(project, existing, output):
    """Reuse completed s31 endpoints only after comparing every folded coefficient."""
    current = project / 'data/current14'
    sys.path.insert(0, str(native_root(project) / 'physical_validation_20260924_r01'))
    import fit14 as f
    if json.loads((current / 'RESULT.json').read_text()).get('rank_with_FM') != 14:
        raise ValueError('current candidate is not full14')
    metrics, inputs = {}, {}
    for kind in ('potential', 'density'):
        kp, fp = current / f'{kind}_kernel.npy', current / f'{kind}_FM.npy'
        op, ofp = existing / f'fits/{kind}_actual.npy', existing / f'fits/{kind}_FM.npy'
        k, fm, old, oldfm = [np.load(p, mmap_mode='r') for p in (kp, fp, op, ofp)]
        if k.shape != (f.g.N, 14) or fm.shape != (f.g.N, 4) or old.shape != (3*f.g.N, 6) or oldfm.shape != (3*f.g.N, 4):
            raise ValueError('unexpected full14 or s31 array shape')
        err = norm = maximum = fmmaximum = 0.
        for start in range(0, f.g.N, PARAM['native_block']):
            ids = np.arange(start, min(start+PARAM['native_block'], f.g.N))
            for tile in f.g.CELLS['s31'][1]:
                sites = np.array(f.c.site_map('s31', tile)); ix = f.g.index('s31', tile, ids)
                value = np.column_stack([k[ids][:, sites == j].sum(axis=1) for j in range(6)])
                if not all(np.isfinite(a).all() for a in (value, old[ix], fm[ids], oldfm[ix])):
                    raise ValueError('non-finite folded fields')
                delta = value-old[ix]
                err += float(np.sum(delta*delta)); norm += float(np.sum(old[ix]**2))
                maximum = max(maximum, float(np.max(abs(delta))))
                fmmaximum = max(fmmaximum, float(np.max(abs(fm[ids]-oldfm[ix]))))
        metrics[kind] = {'relative_L2': math.sqrt(err/norm), 'max_abs_kernel': maximum, 'max_abs_FM': fmmaximum}
        if max(maximum, fmmaximum) > 1e-12 or metrics[kind]['relative_L2'] > 1e-12:
            raise ValueError(f'completed s31 fields do not match current full14: {metrics[kind]}')
        for path in (kp, fp, op, ofp):
            inputs[str(path)] = sha256(path)
    # Tie the completed electronic states to these fields, not just directory names.
    endpoint_metrics = {}
    for side in ('plus', 'minus'):
        endpoint = existing / f'endpoints/actual/{side}'
        spins, _, _, record = read_torque(endpoint / 'TORQUES.json')
        if record.get('case') != side or record.get('variant') != 'actual':
            raise ValueError('expected completed actual-coordinate s31 endpoints')
        for kind, filename in [('density', 'native/DENSITY_ANSATZ.bin'), ('potential', 'state/FROZEN_HXC.bin')]:
            if not (endpoint / filename).is_file():
                endpoint_metrics[f'{side}_{kind}'] = 'SAVED_ENDPOINT_FIELD_NOT_PRESENT_ON_THIS_HOST; kernel equivalence and completed torque provenance retained'
                continue
            kernel = np.load(existing / f'fits/{kind}_actual.npy', mmap_mode='r')
            fm = np.load(existing / f'fits/{kind}_FM.npy', mmap_mode='r')
            field, shape = f.g.read(endpoint / filename)
            if shape != (3*PARAM['nr1'], PARAM['nr2'], PARAM['nr3']):
                raise ValueError('s31 field grid mismatch')
            maximum = 0.
            for start in range(0, len(kernel), PARAM['native_block']):
                stop = min(start+PARAM['native_block'], len(kernel))
                predicted = kernel[start:stop] @ spins
                predicted[:, :2] += fm[start:stop, 1:3]
                maximum = max(maximum, float(np.max(abs(predicted-field[start:stop, 1:]))))
                if kind == 'density':
                    maximum = max(maximum, float(np.max(abs(fm[start:stop, 0]-field[start:stop, 0]))))
            if maximum > 1e-12 or not math.isfinite(maximum):
                raise ValueError(f'completed endpoint field does not match reused kernels: {side}/{kind}')
            endpoint_metrics[f'{side}_{kind}_max_abs'] = maximum
    # All comparisons completed before creating the compact current-data entry.
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for side in ('plus', 'minus'):
        src = existing / f'endpoints/actual/{side}/TORQUES.json'
        read_torque(src)
        shutil.copy2(src, output / f'{side}.json')
        sources[str(src)] = sha256(src)
    for name in ('RESULT.json', 'REFERENCE.json'):
        src = existing / name
        shutil.copy2(src, output / ('s31_'+name))
        sources[str(src)] = sha256(src)
    certificate = {'physical_release': False, 'status': 'FULL14_S31_FOLDED_FIELDS_EQUIVALENT',
                   'method': 'All grid points, six site columns, both FM four-component fields; atol and relative kernel tolerance 1e-12',
                   'metrics': metrics, 'endpoint_field_metrics': endpoint_metrics,
                   'source_sha256': {**inputs, **sources},
                   'limits': 's31 kernel equivalence only; absent saved endpoint fields cannot be rechecked on this host. No new electronic-state, larger-cell or higher-order physical acceptance.'}
    write_json(output / 'FULL14_EQUIVALENCE.json', certificate)
    pins = {name: sha256(output / name) for name in ('plus.json', 'minus.json', 'FULL14_EQUIVALENCE.json')}
    for path in current.glob('*.npy'):
        if path.name in ('potential_kernel.npy', 'density_kernel.npy', 'potential_FM.npy', 'density_FM.npy'):
            import os
            pins[os.path.relpath(path, output)] = inputs[str(path)]
    config = {'spin_S': PARAM['spin_s'], 'input_sha256': pins,
              'pairs': [{'cell': 's31', 'plus': 'plus.json', 'minus': 'minus.json',
                         'step_degrees': PARAM['angle_degrees'], 'response_sites': [1, 2, 3, 4, 5]}]}
    write_json(output / 'torque_input.json', config)
    result = fit(output / 'torque_input.json', project)
    if result['rank'] != 3:
        raise ValueError('s31 exchange design unexpectedly rank deficient')
    write_json(output / 'J123.json', result)
    print(json.dumps({'output': str(output), 'rank': result['rank'], 'J_meV': result['J_meV'], 'equivalence': metrics}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit_cmd = commands.add_parser('fit')
    fit_cmd.add_argument('config', type=Path); fit_cmd.add_argument('--output', type=Path, required=True)
    connect = commands.add_parser('connect-full14')
    connect.add_argument('--existing-s31', type=Path, required=True)
    connect.add_argument('--output', type=Path, required=True)
    for cmd in (fit_cmd, connect):
        cmd.add_argument('--project', type=Path, default=ROOT.parent)
    args = parser.parse_args()
    if args.command == 'fit':
        result = fit(args.config.resolve(), args.project.resolve())
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
    else:
        connect_full14(args.project.resolve(), args.existing_s31.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
