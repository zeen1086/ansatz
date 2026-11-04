#!/usr/bin/env python3
"""Combine native six-site torques and fit a conditional three-cell J1--J3 model."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from parameters import load
PARAM = load()
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import re

import numpy as np

R = Path(__file__).resolve().parent
P = R.parent
RY_MEV = 13605.693122994
S2 = PARAM['spin_s']**2
ANGLE = PARAM['angle_degrees']
AXIS = [-1 / math.sqrt(2), 1 / math.sqrt(2), 0.]
CELLS = ('s31', 's13', 'sd')
# All geometric radii through 7 a^2, including radii omitted in legacy J6 labels.
RADII3 = np.array([1, 3, 4, 7, 9, 12, 13, 16, 19, 21])
LEGACY8 = [0, 1, 2, 3, 4, 6, 7, 9]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


model = module('three_cell_model', P / 'full_spatial_candidate_20260918_r02/model.py')
mapping_dir = R / 'reference_2x2'
if not mapping_dir.is_dir():
    mapping_dir = P.parent / 'j2_energy_mapping_20260801'
mapping = module('old_mapping_geometry', mapping_dir / 'solve_J123.py')
BASIS = np.array([mapping.BASIS['A'], mapping.BASIS['B']])


def counts(matrix, sites, radius=8):
    """Directed periodic neighbor multiplicities; self-image terms are retained."""
    images = np.array(list(itertools.product(range(-radius, radius + 1), repeat=2))) @ matrix
    delta = sites[None, :, None, :] + images[None, None, :, :] - sites[:, None, None, :]
    d3 = 3 * (delta[..., 0] ** 2 - delta[..., 0] * delta[..., 1] + delta[..., 1] ** 2)
    return np.sum(abs(d3[..., None] - RADII3) < 1e-8, axis=2)


def geometry(cell):
    matrix, tiles, _ = model.CELLS[cell]
    sites = np.array([np.array(tile) + basis for tile in tiles for basis in BASIS])
    neighbors = counts(matrix, sites)
    # A periodic spin coordinate rotates its images too: its self-image bond energy is constant.
    offsite = neighbors[1:, 0, :]
    return matrix, sites, offsite


def nullspace(a):
    _, singular, vh = np.linalg.svd(a, full_matrices=True)
    rank = int(np.count_nonzero(singular > 1e-10))
    return rank, singular, vh[rank:].T


def fit(a, y):
    rank, singular, null = nullspace(a)
    solution = np.linalg.lstsq(a, y, rcond=1e-12)[0]
    residual = y - a @ solution
    out = {'rank': rank, 'parameters': a.shape[1], 'singular_values': singular.tolist(),
           'right_nullspace_columns': null.tolist(), 'prediction_Ry': (a @ solution).tolist(),
           'residual_Ry': residual.tolist(), 'rms_Ry': float(np.sqrt(np.mean(residual ** 2))),
           'max_abs_residual_meV': float(np.max(abs(residual)) * RY_MEV),
           'Jspin_meV': (-solution * RY_MEV / S2).tolist() if rank == a.shape[1] else None,
           'uncertainty': 'NOT_ESTABLISHED; fit residual is not a physical uncertainty.'}
    if a.shape[1] == 3 and rank == 2:
        combinations = np.array([[1., 0, 1], [0, 1, 0]])
        if np.max(abs(combinations @ null)) < 1e-10:
            out['identified_combinations_meV'] = dict(zip(
                ['J1_plus_J3', 'J2'], (-combinations @ solution * RY_MEV / S2).tolist()))
    return out


def fold_2x2():
    """Derive old mapping folds on all 256 collinear configurations, not just four."""
    sites = np.array([np.array(tile) + basis
                      for tile in itertools.product(range(2), repeat=2) for basis in BASIS])
    neighbors = counts(np.diag([2, 2]), sites)
    signs = np.array(list(itertools.product([-1, 1], repeat=8)))
    bonds = .5 * np.einsum('ni,nj,ijs->ns', signs, signs, neighbors)
    design = np.column_stack([np.ones(len(signs)), bonds[:, :3]])
    coefficients = np.linalg.lstsq(design, bonds, rcond=None)[0]
    assert np.max(abs(design @ coefficients - bonds)) < 1e-10
    return np.rint(coefficients).astype(int)


def aliases(a, fold):
    rank, singular, null = nullspace(a)
    projection = null @ null.T
    identifiable = np.linalg.norm(fold @ null, axis=1) < 1e-10
    witnesses = []
    for row, ok in zip(fold, identifiable):
        if ok:
            witnesses.append(None)
            continue
        direction = projection @ row
        direction /= np.max(abs(direction))
        assert np.max(abs(a @ direction)) < 1e-10
        witnesses.append({'delta_shell_Jspin_meV': direction.tolist(),
                          'delta_2x2_effective_Jspin_meV': (fold @ direction).tolist(),
                          'max_abs_observed_Phi_change_Ry': float(np.max(abs(a @ direction)) * S2 / RY_MEV)})
    return {'rank': rank, 'parameters': a.shape[1], 'singular_values': singular.tolist(),
            'right_nullspace_columns': null.tolist(),
            '2x2_fold_matrix': fold.tolist(), '2x2_effective_J_identifiable': identifiable.tolist(),
            'nonidentifiability_witnesses': witnesses,
            '2x2_effective_Jspin_meV': None,
            'status': 'NONIDENTIFIABLE' if not np.all(identifiable) else 'IDENTIFIABLE_IN_DECLARED_RANGE'}


def load(path, sources):
    raw = path.read_bytes()
    sources[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def combine(cell, sources):
    cases = {}
    electronic = {}
    for case, sign in [('plus', 1), ('minus', -1)]:
        record = load(R / cell / case / 'TORQUES.json', sources)
        assert record.get('case', case) == case
        assert record.get('cell', cell) == cell
        assert record.get('perturbed_site', 0) == 0
        assert abs(record['angle_deg'] - sign * ANGLE) < 1e-12
        if 'rotation_axis' in record:
            assert np.allclose(record['rotation_axis'], AXIS, rtol=0, atol=1e-12)
        assert np.allclose(record['axes'], np.tile(AXIS, (6, 1)), rtol=0, atol=1e-12)
        electronic[case] = {key: record[key] for key in
                            ('n_kpoints', 'k_weight_sum', 'global_indirect_gap_Ry', 'occupied_states_per_k')}
        rows = record['rows_Ry']
        assert len(rows) == 6
        assert [row['site'] for row in rows] == list(range(6))
        cases[case] = {key: np.array([row[key] for row in rows], float)
                       for key in ('band', 'input_work', 'potential_mismatch_work', 'total')}
        assert all(np.isfinite(value).all() for value in cases[case].values())
        assert np.max(abs(cases[case]['total'] - sum(cases[case][key] for key in
                          ('band', 'input_work', 'potential_mismatch_work')))) < 1e-12
    phi = {key: (cases['plus'][key] - cases['minus'][key]) / (2 * math.radians(ANGLE))
           for key in cases['plus']}
    _, _, a = geometry(cell)
    out = {'classification': 'NATIVE_FULL_K_FOLDED_TORQUE_COLUMN',
           'cell': cell, 'angle_degrees': ANGLE, 'perturbed_site': 0, 'rotation_axis': AXIS,
           'site_order': [[*tile, sublattice] for tile in model.CELLS[cell][1] for sublattice in (0, 1)],
           'electronic_k_mesh': [12, 4, 1] if cell == 's13' else [7, 7, 1],
           'endpoint_electronic_summary': electronic,
           'Phi_tau_Ry': {key: value.tolist() for key, value in phi.items()},
           'torques_Ry': {case: {key: value.tolist() for key, value in row.items()}
                          for case, row in cases.items()},
           'C2_Ry': float(-phi['total'][0]), 'column_sum_Ry': float(phi['total'].sum()),
           'even_total_torque_Ry': ((cases['plus']['total'] + cases['minus']['total']) / 2).tolist(),
           'fit_J123': fit(a[:, :3], phi['total'][1:]),
           'source_sha256': {key: value for key, value in sources.items() if f'/{cell}/' in key}}
    (R / cell / 'RESULT.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    return out


def old_mapping(sources):
    path = mapping_dir / 'RESULT.txt'
    raw = path.read_bytes()
    sources[str(path)] = hashlib.sha256(raw).hexdigest()
    text = raw.decode()
    j = np.array([float(re.search(rf'J{i}_paper\s*=\s*([-+\d.]+)', text)[1]) for i in (1, 2, 3)])
    energies = {cfg: float(re.search(rf'^\s*{cfg}\s+E=([-+\d.]+)', text, re.M)[1])
                for cfg in mapping.CONFIGS}
    patterns = load(mapping_dir / 'PATTERNS.json', sources)
    bonds = np.array([mapping.bond_sums(patterns[cfg]['cr_sites'], patterns[cfg]['signs'])
                      for cfg in mapping.CONFIGS])
    delta = bonds - bonds[0]
    dft = (np.array([energies[cfg] for cfg in mapping.CONFIGS]) - energies['fm']) * RY_MEV
    return j, delta, dft


def self_test():
    expected = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1], [0, 2, 0], [1, 0, 2]])
    assert np.array_equal(geometry('s31')[2][:, :3], expected)
    for cell in CELLS:
        matrix, sites, _ = geometry(cell)
        assert np.array_equal(counts(matrix, sites, 7), counts(matrix, sites, 8))
    native = json.loads((P / 'full_k_qr_torque_20260920_r01/RESULT.json').read_text())
    actual = fit(expected, np.array(native['Phi_tau_Ry']['total'][1:]))['Jspin_meV']
    assert np.max(abs(np.array(actual) - native['Jspin_meV']['total'])) < 1e-11
    a = np.concatenate([geometry(cell)[2] for cell in CELLS])
    assert nullspace(a[:, :3])[0] == 3 and nullspace(a[:, LEGACY8])[0] == 5
    fold = fold_2x2()
    assert np.array_equal(fold[1:, LEGACY8],
                          [[1, 0, 0, 2, 0, 2, 0, 0], [0, 1, 0, 0, 1, 0, 0, 2],
                           [0, 0, 1, 0, 0, 0, 1, 0]])
    test_j = np.array([-4.2, -.6, .1])
    recovered = fit(a[:, :3], a[:, :3] @ (-test_j * S2 / RY_MEV))['Jspin_meV']
    assert np.max(abs(recovered - test_j)) < 1e-12
    assert not any(aliases(a, fold[1:])['2x2_effective_J_identifiable'])
    print('PASS: real s31 J, cell geometry, image convergence, joint sign/units, 256-state 2x2 folds and nullspace witnesses')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
               (Path(__file__).resolve(), Path(model.__file__), Path(mapping.__file__))}
    data = {'s31': load(P / 'full_k_qr_torque_20260920_r01/RESULT.json', sources)}
    missing = []
    for cell in ('s13', 'sd'):
        paths = [R / cell / case / 'TORQUES.json' for case in ('plus', 'minus')]
        if all(path.is_file() for path in paths):
            data[cell] = combine(cell, sources)
        else:
            missing.extend(str(path) for path in paths if not path.is_file())
    old_j, delta_bonds, old_dft = old_mapping(sources)
    per_cell = {}
    for cell, result in data.items():
        matrix, sites, all_shells = geometry(cell)
        phi = np.asarray(result['Phi_tau_Ry']['total'], float)
        assert phi.shape == (6,) and np.isfinite(phi).all()
        assert abs(result['angle_degrees'] - ANGLE) < 1e-12
        a = all_shells[:, :3]
        previous_prediction = a @ (-old_j * S2 / RY_MEV)
        per_cell[cell] = {'cell_matrix_rows': matrix.tolist(), 'Cr_primitive_coordinates': sites.tolist(),
                          'shell_counts_J123_offsite': a.tolist(), 'Phi_tau_Ry': phi.tolist(),
                          'fit': fit(a, phi[1:]), 'C2_meV': float(-phi[0] * RY_MEV),
                          'column_sum_Ry': float(phi.sum()),
                          'old_mapping_truncated_J123_Phi_tau_Ry': [-float(previous_prediction.sum()), *previous_prediction.tolist()],
                          'old_mapping_truncated_J123_C2_meV': float(previous_prediction.sum() * RY_MEV),
                          'old_mapping_truncated_J123_offsite_residual_Ry': (phi[1:] - previous_prediction).tolist()}
    planned = np.concatenate([geometry(cell)[2] for cell in CELLS])
    fold = fold_2x2()
    joint = None
    comparison = None
    if not missing:
        y = np.concatenate([np.array(data[cell]['Phi_tau_Ry']['total'][1:]) for cell in CELLS])
        joint = fit(planned[:, :3], y)
        j = np.array(joint['Jspin_meV'])
        for index, cell in enumerate(CELLS):
            predicted_phi = np.asarray(joint['prediction_Ry'][5 * index:5 * (index + 1)])
            per_cell[cell]['joint_J123_Phi_tau_Ry'] = [-float(predicted_phi.sum()), *predicted_phi.tolist()]
            per_cell[cell]['joint_J123_offsite_residual_Ry'] = joint['residual_Ry'][5 * index:5 * (index + 1)]
        prediction = S2 * delta_bonds @ j
        comparison = {'condition': 'Isotropic bilinear J1--J3 truncation; not de-aliased effective J with longer shells.',
                      'configurations': mapping.CONFIGS, 'bond_sum_difference_from_FM': delta_bonds.tolist(),
                      'old_DFT_excitation_energy_meV_per_2x2_cell': old_dft.tolist(),
                      'joint_J123_predicted_excitation_energy_meV_per_2x2_cell': prediction.tolist(),
                      'energy_difference_meV_per_2x2_cell': (prediction - old_dft).tolist(),
                      'old_mapping_reported_Jspin_meV': old_j.tolist(),
                      'joint_minus_old_mapping_Jspin_meV': (j - old_j).tolist(),
                      'note': 'Large-angle energies here are spin-model predictions, not new ansatz or DFT energy scans.'}
    out = {'classification': 'WAITING_FOR_CELL_TORQUES' if missing else 'THREE_CELL_CONDITIONAL_J123',
           'missing_files': missing, 'cells': per_cell, 'joint_J123': joint,
           '2x2_conditional_comparison': comparison,
           'conventions': 'Phi_tau=dtau/dtheta=-C2; radians; H=+sum_unordered Jspin Si.Sj, S=1.5; Jspin=-JUnit/2.25.',
           'fit_policy': 'Equal weight per offsite response, 15 rows when complete; onsite response tests the sum, not an extra fitted row.',
           'geometric_identifiability': {
               'data_scope': 'Planned three cells, site0 column; geometry-only, independent of missing computations.',
               'all_shells_radius_squared_a2': (RADII3 / 3).tolist(),
               'all_shells_through_7a2': aliases(planned, fold[1:]),
               'legacy_eight_selected_radius_squared_a2': (RADII3[LEGACY8] / 3).tolist(),
               'legacy_eight_selected': aliases(planned[:, LEGACY8], fold[1:, LEGACY8]),
               '2x2_constant_bond_sums': fold[0].tolist()},
           'limits': ['Three determinant-3 cells do not determine all longer-range shells or the 2x2 folded J combinations.',
                      's31 uses established 48-k QR; s13 uses native 48-k and sd native 49-k quadratures. Cross-method residual is not an error bar.',
                      'One input site and one rotation axis per cell; reciprocal full tensor and independent DFT uncertainty remain unverified.',
                      'Phi4 deferred. Joint fit is a candidate effective spin model, not a physical release.'],
           'source_sha256': sources}
    encoded = json.dumps(out, indent=2, allow_nan=False) + '\n'
    (R / 'JOINT_RESULT.json').write_text(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
