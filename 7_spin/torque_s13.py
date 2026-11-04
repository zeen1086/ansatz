#!/usr/bin/env python3
"""Complete two full14 s13 endpoints using the existing native three-term runner."""
from pathlib import Path
import argparse
import json
import math
import os
import shutil
import subprocess
import sys

os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parameters import load
PARAM = load()
import fit_native_torque as reader
NATIVE = reader.native_root(ROOT)
sys.path.insert(0, str(NATIVE / 'physical_validation_20260924_r01'))
import torque_sd as shared

R = ROOT / 'runs/full14_s13_20260926'
CURRENT = ROOT / 'data/current14'
OLD = Path(PARAM['native_data_root'])
WFC = OLD / 'spatial_completion_continue_20260918_r01/s13_zero/tmp/CrI3.save'
SEED = R / 'seed/tmp/CrI3.save'
SHAPE = (PARAM['nr1'], 3*PARAM['nr2'], PARAM['nr3'])
l, f = shared.l, shared.f


def bands_input(save):
    text = (R / 'source_s13.in').read_text()
    values = {'calculation': "'bands'", 'restart_mode': "'from_scratch'",
              'outdir': repr(str(save.parent)), 'pseudo_dir': repr(str(SEED)),
              'disk_io': "'low'", 'wf_collect': '.true.', 'startingpot': "'file'",
              'startingwfc': "'file'", 'constrained_magnetization': "'none'",
              'diago_thr_init': str(PARAM['native_diago_thr']), 'max_seconds': str(PARAM['native_max_seconds'])}
    for key, value in values.items():
        text = shared.campaign.replace(text, key, value)
    return text


def check():
    ids = np.arange(3*3*5)
    mapped = np.concatenate([f.g.index('s13', tile, ids, (3, 3, 5)) for tile in f.g.CELLS['s13'][1]])
    assert np.array_equal(np.sort(mapped), np.arange(3*len(ids)))
    for tile in f.g.CELLS['s13'][1]:
        assert set(f.c.site_map('s13', tile)) == set(range(6))
    matrix, sites, design = shared.a.geometry('s13')
    assert np.linalg.matrix_rank(design[:, :3]) == 3
    expected = np.array([[2, 0, 0], [0, 2, 0], [1, 0, 2], [0, 2, 0], [0, 0, 1]])
    assert np.array_equal(design[:, :3], expected)
    print('PASS: s13 grid bijection, site map and rank3 shell design')


def prepare():
    assert os.environ.get('PBS_JOBID'), 'Compute allocation required'
    check()
    R.mkdir(parents=True, exist_ok=False)
    (R / 'fits').mkdir(); SEED.mkdir(parents=True)
    source = WFC.parents[1] / 'scf.in'
    shutil.copy2(source, R / 'source_s13.in')
    assert len(list(WFC.glob('wfc*.dat'))) == PARAM['s13_kpoints']
    for path in WFC.iterdir():
        if path.name in ('data-file-schema.xml', 'charge-density.dat') or path.suffix.lower() == '.upf':
            shutil.copy2(path, SEED / path.name)
    assert (SEED / 'data-file-schema.xml').is_file() and (SEED / 'charge-density.dat').is_file()
    pins = {}
    for kind in ('potential', 'density'):
        kp, fp = CURRENT / f'{kind}_kernel.npy', CURRENT / f'{kind}_FM.npy'
        kernel, reference = np.load(kp, mmap_mode='r'), np.load(fp, mmap_mode='r')
        folded = np.lib.format.open_memmap(R / f'fits/{kind}_actual.npy', mode='w+', dtype='<f8', shape=(3*f.g.N, 6))
        fm = np.lib.format.open_memmap(R / f'fits/{kind}_FM.npy', mode='w+', dtype='<f8', shape=(3*f.g.N, 4))
        for start in range(0, f.g.N, PARAM['native_block']):
            ids = np.arange(start, min(start+PARAM['native_block'], f.g.N))
            for tile in f.g.CELLS['s13'][1]:
                sites = np.array(f.c.site_map('s13', tile)); ix = f.g.index('s13', tile, ids)
                folded[ix] = np.column_stack([kernel[ids][:, sites == site].sum(axis=1) for site in range(6)])
                fm[ix] = reference[ids]
        folded.flush(); fm.flush()
        assert np.isfinite(folded).all() and np.max(abs(folded.sum(axis=1)-fm[:, 3])) < 1e-10
        pins[str(kp)] = reader.sha256(kp); pins[str(fp)] = reader.sha256(fp)
        print('FOLDED', kind, flush=True)
    rows = {}
    for side in ('plus', 'minus'):
        src = OLD / f'three_cell_torque_20260920_r01/s13/{side}/TORQUES.json'
        spins, axes, _, _ = reader.read_torque(src)
        assert np.allclose(axes, np.tile(shared.a.AXIS, (6, 1)), atol=1e-12, rtol=0)
        rows[side] = {'moments': spins.tolist(), 'coordinate_source': str(src),
                      'note': 'Reuse specified unit-spin geometry only; old model torque is not reused'}
    reader.write_json(R / 'REFERENCE.json', {'rows': rows, 'physical_release': False})
    reader.write_json(R / 'fits/RESULT.json', {'rank_source': 14, 'physical_release': False,
                      'method': 'Exact s13 fold of current actual-direction full14; no refit'})
    reader.write_json(R / 'INPUTS.json', {'current_kernel_sha256': pins, 'new_scf': False,
                      'endpoints': 2, 'spin_step_degrees': PARAM['angle_degrees'], 'WFC_seed': str(WFC),
                      'shape': SHAPE, 'k_mesh': [12, 4, 1], 'physical_release': False})
    text = bands_input(Path('/tmp/s13_input_check/tmp/CrI3.save'))
    assert "calculation='bands'" in text and "startingwfc='file'" in text and "constrained_magnetization='none'" in text
    assert text.split('CELL_PARAMETERS')[1] == source.read_text().split('CELL_PARAMETERS')[1]
    print('FULL14_S13_PREPARED', flush=True)


def execute(binary, directory, text, logname, marker, ranks):
    assert os.environ.get('PBS_JOBID'), 'Compute allocation required'
    if binary.name == 'pw_frozen_wfc.x':
        files = sorted(WFC.glob('wfc*.dat')); assert len(files) == PARAM['s13_kpoints']
        assert shutil.disk_usage(directory).free > sum(p.stat().st_size for p in files) + 20_000_000_000
        save = directory / 'tmp/CrI3.save'
        for src in files:
            dst = save / src.name; assert not dst.exists()
            before = src.stat(); shutil.copy2(src, dst)
            assert dst.stat().st_ino != before.st_ino and dst.stat().st_size == before.st_size and src.stat().st_mtime_ns == before.st_mtime_ns
        print('WFC_COPY_COMPLETE', flush=True)
    argv = ['mpirun', '-np', str(ranks), str(binary)]
    if ranks > 1:
        argv += ['-npool', '1', '-ndiag', '1']
    with (directory / logname).open('x') as stream:
        subprocess.run(argv, cwd=directory, input=text, text=True, stdout=stream, stderr=subprocess.STDOUT, check=True)
    log = (directory / logname).read_text()
    assert marker in log and 'eigenvalues not converged' not in log


def endpoint(side):
    for name, value in json.loads((R / 'INPUTS.json').read_text())['current_kernel_sha256'].items():
        assert reader.sha256(Path(name)) == value, 'current model changed'
    l.R = R; l.SEED = SEED; l.SHAPE = SHAPE
    l.execute = execute; l.bands_input = bands_input
    l.band_data = lambda save: shared.w.band_data(save / 'data-file-schema.xml', PARAM['s13_kpoints'])
    original_save = l.save_json
    def metadata(path, value):
        if path.name == 'TORQUES.json':
            value.update(cell='s13', grid=list(SHAPE), source_candidate=str(CURRENT),
                         startingwfc='file; private copy of existing s13 FM wavefunctions',
                         scope='Current full14 s13 defined-spin finite-difference endpoint; three-term native torque')
        original_save(path, value)
    l.save_json = metadata
    l.endpoint('actual', side)
    reader.read_torque(R / f'endpoints/actual/{side}/TORQUES.json')
    print('FULL14_S13_TORQUE_COMPLETE', side, flush=True)


def analyze():
    sources = {'s31': CURRENT / 's31_response',
               's13': R / 'endpoints/actual',
               'sd': ROOT / 'data/reference/physical_validation_20260924_r01/torque_sd_20260925/endpoints/actual'}
    inputs = {}
    for cell, base in sources.items():
        for side in ('plus', 'minus'):
            path = base / (f'{side}.json' if cell == 's31' else f'{side}/TORQUES.json')
            reader.read_torque(path)
            inputs[cell, side] = path
    output = CURRENT / 'three_cell_response'; output.mkdir(exist_ok=False)
    config = {'spin_S': PARAM['spin_s'], 'input_sha256': {}, 'pairs': []}
    for cell in sources:
        pair = {'cell': cell, 'step_degrees': PARAM['angle_degrees'], 'response_sites': [1, 2, 3, 4, 5]}
        for side in ('plus', 'minus'):
            name = f'{cell}_{side}.json'; shutil.copy2(inputs[cell, side], output / name)
            pair[side] = name; config['input_sha256'][name] = reader.sha256(output / name)
        config['pairs'].append(pair)
    for path, value in json.loads((R / 'INPUTS.json').read_text())['current_kernel_sha256'].items():
        config['input_sha256'][os.path.relpath(path, output)] = value
    source = output / 'torque_input.json'; reader.write_json(source, config)
    result = reader.fit(source, ROOT)
    assert result['rank'] == 3 and len(result['observations']) == 3
    result['scope'] = 'Joint s31+s13+SD current full14 conditional J1--J3; equally weighted offsite response rows'
    reader.write_json(output / 'J123.json', result)
    # The conditional fit is an output; downstream models are configured separately.
    valid = all(row['relative_residual'] <= PARAM['cell_residual_limit'] for row in result['cell_fits'])
    reader.write_json(output / 'STATUS.json', {'status': 'THREE_CELL_FIT_COMPLETE' if valid else 'THREE_CELL_RESPONSE_MISMATCH',
                      'response_criterion_passed': valid, 'physical_release': False,
                      'criterion': f"each cell offsite response residual <= {PARAM['cell_residual_limit']}; not independent DFT acceptance"})
    print(json.dumps({'J_meV': result['J_meV'], 'cell_fits': result['cell_fits'], 'response_criterion_passed': valid}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'prepare', 'endpoint', 'analyze'))
    parser.add_argument('side', nargs='?', choices=('plus', 'minus'))
    args = parser.parse_args()
    if args.command == 'endpoint':
        if args.side is None: parser.error('endpoint requires plus or minus')
        endpoint(args.side)
    else:
        globals()[args.command]()
