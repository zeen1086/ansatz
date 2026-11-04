#!/usr/bin/env python3
"""Full-k native-plane-wave torque for the existing 42-function potential."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from parameters import load
PARAM = load()
import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

R = Path(__file__).resolve().parent
P = R.parent
spec = importlib.util.spec_from_file_location('general_fields', P / 'general_torque_correct_grid_20260921_r01/fields.py')
fields = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fields)
g, ev = fields.g, fields.ev


def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def source_stats(directory):
    return [{'name': f.name, 'size': f.stat().st_size, 'mtime_ns': f.stat().st_mtime_ns,
             'inode': f.stat().st_ino} for f in sorted(directory.iterdir())
            if f.is_file() and (f.name in ('charge-density.dat', 'data-file-schema.xml')
                               or f.name.startswith('wfc') or f.suffix == '.upf')]


def run_command(argv, directory, filename, input_text, marker):
    log = directory / filename
    if log.exists():
        text = log.read_text()
        assert marker in text and 'eigenvalues not converged' not in text, \
            f'Incomplete or failed stage retained: {log}; no automatic retry.'
        return
    with log.open('x') as stream:
        subprocess.run(argv, cwd=directory, input=input_text, text=True, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    text = log.read_text()
    assert marker in text and 'eigenvalues not converged' not in text, log


def bands_input(source_text, outdir):
    replacements = {'calculation': "'bands'", 'restart_mode': "'from_scratch'",
                    'outdir': repr(str(outdir)), 'disk_io': "'low'", 'wf_collect': '.true.',
                    'startingpot': "'file'", 'startingwfc': "'file'",
                    'constrained_magnetization': "'none'", 'diago_thr_init': str(PARAM['native_diago_thr']),
                    'max_seconds': str(PARAM['native_three_cell_seconds'])}
    for key, value in replacements.items():
        pattern = rf'(?m)^(\s*{key}\s*=)\s*[^\n,]+'
        source_text, count = re.subn(pattern, lambda m: m[1] + value, source_text)
        assert count == 1, (key, count)
    return source_text


def contractions(field_dir, rho):
    counter = json.loads((field_dir / 'COUNTERTERMS.json').read_text())
    dv = counter['voxel_volume_bohr3']
    rows = []
    for site, terms in enumerate(counter['rows_Ry']):
        assert terms['site'] == site
        derivative, _ = g.read(field_dir / f'DERIVATIVE_{site}.bin')
        band = -ev.integral(rho, derivative, dv)
        input_work, mismatch = terms['input_work'], terms['potential_mismatch_work']
        rows.append({'site': site, 'band': band, 'input_work': input_work,
                     'potential_mismatch_work': mismatch,
                     'total': math.fsum([band, input_work, mismatch])})
    assert len(rows) == 6
    return rows


def band_data(xml_path, expected_kpoints):
    xml = ET.parse(xml_path)
    band = xml.find('.//output/band_structure')
    assert int(band.findtext('nks')) == expected_kpoints
    assert abs(float(band.findtext('nelec')) - PARAM["native_occupied"]) < 1e-9
    states = band.findall('ks_energies')
    assert len(states) == expected_kpoints
    weights, energies, occupied_max, empty_min = [], [], [], []
    for state in states:
        weight = float(state.find('k_point').attrib['weight'])
        values = 2 * np.fromstring(state.findtext('eigenvalues'), sep=' ')
        occupations = np.fromstring(state.findtext('occupations'), sep=' ')
        assert len(values) == len(occupations) == PARAM["native_nbnd"]
        assert np.max(abs(occupations[:PARAM["native_occupied"]] - 1)) < 1e-10
        assert np.max(abs(occupations[PARAM["native_occupied"]:])) < 1e-10
        weights.append(weight)
        energies.append(weight * float(values @ occupations))
        occupied_max.append(values[(PARAM["native_occupied"]-1)])
        empty_min.append(values[PARAM["native_occupied"]])
    assert abs(math.fsum(weights) - 1) < 1e-10
    gap = float(min(empty_min) - max(occupied_max))
    assert gap > 0
    return {'band_energy_Ry': math.fsum(energies), 'n_kpoints': expected_kpoints,
            'k_weight_sum': math.fsum(weights), 'global_indirect_gap_Ry': gap,
            'occupied_states_per_k': PARAM["native_occupied"]}


def execute(cell, case):
    endpoint = R / cell / case
    endpoint.mkdir(parents=True, exist_ok=True)
    if (endpoint / 'TORQUES.json').exists():
        result = json.loads((endpoint / 'TORQUES.json').read_text())
        assert result['cell'] == cell and result['case'] == case
        print('COMPLETED_ENDPOINT_REUSED', endpoint, flush=True)
        return
    angle = math.radians(PARAM['angle_degrees'] * (1 if case == 'plus' else -1))
    spins = np.tile([0., 0., 1.], (6, 1))
    spins[0] = [math.sin(angle) / math.sqrt(2), math.sin(angle) / math.sqrt(2), math.cos(angle)]
    axes = np.tile([-1 / math.sqrt(2), 1 / math.sqrt(2), 0.], (6, 1))
    for name, value in [('spins', spins), ('axes', axes)]:
        target = endpoint / (name + '.json')
        if target.exists():
            assert np.array_equal(json.loads(target.read_text()), value)
        else:
            write_json(target, value.tolist())
    field_root = endpoint / 'fields'
    if not (field_root / 'GENERATED.json').exists():
        assert not field_root.exists(), 'Incomplete field stage retained; no automatic retry.'
        run_command([sys.executable, str(fields.R / 'fields.py'), cell,
                     str(endpoint / 'spins.json'), str(endpoint / 'axes.json'), str(field_root)],
                    endpoint, 'fields.out', '', 'GENERAL_TORQUE_FIELDS_COMPLETE')
    generated = json.loads((field_root / 'GENERATED.json').read_text())
    assert generated['cell'] == cell and np.array_equal(generated['spins'], spins)
    field_dir = field_root / 'dynamic'
    print('PHASE_FIELDS_COMPLETE', cell, case, flush=True)
    source = g.reference_tmp(cell).parent
    seed = source / 'tmp/CrI3.save'
    nk = {'s13': PARAM['s13_kpoints'], 'sd': PARAM['sd_kpoints']}[cell]
    assert len(list(seed.glob('wfc*.dat'))) == nk
    state = endpoint / 'state'
    save = state / 'tmp/CrI3.save'
    prepared = endpoint / 'PREPARED.json'
    if not prepared.exists():
        assert not state.exists(), 'Incomplete WFC copy retained; no automatic retry.'
        stats = source_stats(seed)
        need = sum(row['size'] for row in stats) + 5_000_000_000
        assert shutil.disk_usage(endpoint).free > need
        save.mkdir(parents=True)
        for row in stats:
            src, dst = seed / row['name'], save / row['name']
            assert not src.is_symlink(), src
            shutil.copy2(src, dst)
            assert not dst.is_symlink() and dst.stat().st_ino != src.stat().st_ino
            assert dst.stat().st_size == row['size']
        assert source_stats(seed) == stats
        shutil.copy2(field_dir / 'FROZEN_HXC.bin', state / 'FROZEN_HXC.bin')
        inp = bands_input((source / 'scf.in').read_text(), state / 'tmp')
        (state / 'bands.in').write_text(inp)
        write_json(prepared, {'cell': cell, 'case': case, 'source': str(seed),
                             'n_kpoints': nk, 'source_files': stats,
                             'diagonalization': 'Native full plane waves; no SCF, constraint or penalty.',
                             'copied_wfc': True})
    metadata = json.loads(prepared.read_text())
    assert metadata['cell'] == cell and metadata['case'] == case
    assert source_stats(seed) == metadata['source_files']
    for filename in seed.glob('wfc*.dat'):
        dst = save / filename.name
        assert dst.is_file() and not dst.is_symlink() and dst.stat().st_ino != filename.stat().st_ino
    print('PHASE_PREPARED', cell, case, flush=True)
    try:
        run_command(['mpirun', '-np', str(PARAM['native_mpi_ranks']), str(P / 'frozen_pw_projection_scan_r01/pw_frozen_wfc.x'),
                     '-npool', '1', '-ndiag', '1'], state, 'bands.out',
                    (state / 'bands.in').read_text(), 'JOB DONE.')
        bands_log = (state / 'bands.out').read_text()
        assert 'FROZEN_OVERRIDE_BAND_ENERGIES_ONLY' in bands_log
        assert 'End of band structure calculation' in bands_log, 'No completed band diagonalization.'
        xml_seed = next(row for row in metadata['source_files'] if row['name'] == 'data-file-schema.xml')
        assert (save / 'data-file-schema.xml').stat().st_mtime_ns != xml_seed['mtime_ns'], 'Stale seed XML.'
        print('PHASE_DIAGONALIZATION_COMPLETE', cell, case, flush=True)
        readout = endpoint / 'readout'
        readout.mkdir(exist_ok=True)
        inp = f"&inputpp\n outdir='{state / 'tmp'}'\n prefix='CrI3'\n/\n"
        run_command(['mpirun', '-np', str(PARAM['native_mpi_ranks']), str(P / 'full_k_density_readout_r01/pp_variational.x'),
                     '-npool', '1', '-ndiag', '1'], readout, 'export.out', inp,
                    'FROZEN_SPIN_COORDINATES_COMPLETE')
        rho, shape = g.read(readout / 'RHO_OUT.bin')
        field, field_shape = g.read(field_dir / 'FROZEN_HXC.bin')
        assert shape == field_shape and np.isfinite(rho).all()
        rows = contractions(field_dir, rho)
        electronic = band_data(save / 'data-file-schema.xml', nk)
        energy = json.loads((field_dir / 'ENERGY.json').read_text())
        result = {'cell': cell, 'case': case,
                  'classification': 'FULL_K_NATIVE_SERIALIZED_MAGNETIC_TANGENT_CONSISTENT_TOTAL_TORQUE',
                  'angle_deg': math.degrees(angle), 'perturbed_site': 0,
                  'spins': spins.tolist(), 'axes': axes.tolist(), 'grid': list(shape), 'candidate_model': str(g.R),
                  'rows_Ry': rows, 'site_torque_Ry_per_rad': [row['total'] for row in rows],
                  'sum_torque_Ry_per_rad': math.fsum(row['total'] for row in rows),
                  'source_wfc_unchanged': True, 'field_directory': str(field_dir),
                  'serialized_potential': energy['serialized_magnetic_potential'],
                  'nonband_energy_Ry': energy['nonband_with_fixed_ion_Ry'], **electronic,
                  'total_energy_Ry': math.fsum([electronic['band_energy_Ry'], energy['nonband_with_fixed_ion_Ry']]),
                  'coordinate': 'Unit spins; axis=(-sin45,cos45,0); angles in radians.',
                  'model': 'Same serialized B42 plus nonlinear scalar HXC of trial density; native rediagonalization.',
                  'torque_definition': '-int(rho_out*dVtilde)+int(rho_in*dVtilde)+int((Vtilde-Vnative)*drho_in).',
                  'limits': ['One column at +/-0.75 degrees, six readouts; no independent DFT acceptance.',
                             's13 uses12x4 electronic mesh; sd uses7x7 oblique mesh. No mesh-convergence claim.',
                             'Native full plane waves; this result does not create a new-cell Wannier interface.']}
        assert source_stats(seed) == metadata['source_files']
        write_json(endpoint / 'TORQUES.json', result)
        print('THREE_CELL_ENDPOINT_COMPLETE', cell, case, flush=True)
    finally:
        assert source_stats(seed) == metadata['source_files'], 'Original FM source changed during run.'


def self_check():
    # Source input retains the actual geometry and mesh while SCF/constraint controls change.
    for cell in ('s13', 'sd'):
        source = (P / 'spatial_completion_r01' / (cell + '_zero') / 'scf.in').read_text()
        target = bands_input(source, Path('/tmp/isolated-torque/tmp'))
        assert target.split('CELL_PARAMETERS')[1] == source.split('CELL_PARAMETERS')[1]
        assert "calculation='bands'" in target and "constrained_magnetization='none'" in target
        assert "startingwfc='file'" in target and "disk_io='low'" in target
        if cell == 'sd':
            fine, count = re.subn(r'(?m)^(\s*nr1\s*=)[^\n,]+', lambda m: m[1] + str(3*PARAM['nr1']), source)
            assert count == 1
            fine_target = bands_input(fine, Path('/tmp/isolated-fine-torque/tmp'))
            assert re.search(rf"(?m)^\s*nr1\s*=\s*{3*PARAM['nr1']}\s*,?$", fine_target)
            assert fine_target.split('CELL_PARAMETERS')[1] == source.split('CELL_PARAMETERS')[1]
    # Independent completed native result obeys the same three-term force bookkeeping.
    for case in ('plus', 'minus'):
        data = json.loads((P / 'tangent_consistent_energy_20260918_r01' / case / 'TORQUES.json').read_text())
        for row in data['rows_Ry']:
            assert abs(math.fsum(row[k] for k in ('band', 'input_work', 'potential_mismatch_work')) - row['total']) < 1e-15
    print('PASS: native geometry/mesh preservation and completed s31 torque bookkeeping')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-check']:
        self_check()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('cell', choices=('s13', 'sd'))
        parser.add_argument('case', choices=('plus', 'minus'))
        args = parser.parse_args()
        for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            os.environ[variable] = '1'
        execute(args.cell, args.case)
