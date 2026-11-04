"""Authorized s31 calibration only: full-grid folded fits and four frozen endpoints."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from parameters import load
PARAM = load()
import importlib.util
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

R = Path(__file__).resolve().parent
P = R.parent
spec = importlib.util.spec_from_file_location('original_model', P/'full_spatial_candidate_20260918_r02/model.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
SEED = P/'tangent_consistent_energy_20260918_r01/plus/state/tmp/CrI3.save'
SHAPE = (3*PARAM['nr1'], PARAM['nr2'], PARAM['nr3'])
H = math.sin(math.radians(PARAM['angle_degrees']))


def save_json(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def write_field(path, a):
    assert a.shape == (3*g.N, 4) and np.isfinite(a).all()
    with path.open('xb') as f:
        f.write(struct.pack('<7i', 16, *SHAPE, 4, 16, a.nbytes))
        for col in a.T:
            f.write(np.asarray(col, dtype='<f8').tobytes())
        f.write(struct.pack('<i', a.nbytes))
    back, shape = g.read(path)
    assert shape == SHAPE and np.array_equal(back, a)


def folded_maps():
    maps = []
    for tile in g.CELLS['s31'][1]:
        f = np.zeros((6, 14))
        for c, (tx, ty, b) in enumerate(g.CHANNELS):
            f[2*((tx+tile[0]) % 3)+b, c] = 1
        maps.append((tile, f))
    return maps


def designs():
    _, rows, nominal, _ = g.setup('potential')
    selected = [i for i, row in enumerate(rows) if row[0] != 'sd']
    actual = np.load(R/'DESIGN.npz')['actual'][selected]
    x = {'nominal': nominal[selected], 'actual': actual}
    proj = np.eye(14)-np.ones((14, 14))/14
    inverses = {}
    for name, matrix in x.items():
        _, s, vt = np.linalg.svd(matrix@proj, full_matrices=False)
        assert np.sum(s > s[0]*1e-8) == 9
        basis = vt[:9].T
        null = proj-basis@basis.T
        for tile, f in folded_maps():
            assert np.max(abs(f@null)) < 1e-12
        inverses[name] = basis@np.linalg.pinv(matrix@basis, rcond=1e-10)
    return selected, x, inverses


def fit():
    selected, x, inverses = designs()
    out = R/'fits'
    out.mkdir()
    metrics = {}
    for kind in ('potential', 'density'):
        zeros, all_rows, _, _ = g.setup(kind)
        rows = [all_rows[i] for i in selected]
        paths = {zeros[c] for c in ('s31', 's13')}
        paths.update(p for row in rows for p in row[2:4] if p is not None)
        data = {p: g.read(p)[0] for p in paths}
        kernels = {name: np.lib.format.open_memmap(out/f'{kind}_{name}.npy', mode='w+', dtype='<f8', shape=(3*g.N, 6)) for name in x}
        fm = np.lib.format.open_memmap(out/f'{kind}_FM.npy', mode='w+', dtype='<f8', shape=(3*g.N, 4))
        rss = {name: 0. for name in x}
        yn = 0.
        for start in range(0, g.N, PARAM['native_block']):
            ids = np.arange(start, min(start+PARAM['native_block'], g.N))
            ix = {(c, u): g.index(c, u, ids) for c in ('s31', 's13') for u in g.CELLS[c][1]}
            ref = np.mean([data[zeros['s31']][ix['s31', u]] for u in g.CELLS['s31'][1]], axis=0)
            base = np.repeat(ref[:, 3, None]/14, 14, axis=1)
            y = np.column_stack([(data[plus if plus else zeros[c]][ix[c, u], comp]-data[minus][ix[c, u], comp])/((2 if plus else 1)*H) for c, mode, plus, minus, u, comp in rows])
            yn += float(np.sum(y*y))
            for name in x:
                coefficients = base+(y-base@x[name].T)@inverses[name].T
                assert np.max(abs(coefficients.sum(axis=1)-ref[:, 3])) < 1e-10
                residual = coefficients@x[name].T-y
                rss[name] += float(np.sum(residual*residual))
                for tile, f in folded_maps():
                    kernels[name][ix['s31', tile]] = coefficients@f.T
                    fm[ix['s31', tile]] = ref
            print('FIT_CHUNK', kind, start, flush=True)
        fm.flush()
        for k in kernels.values():
            k.flush()
        metrics[kind] = {name: {'relative_residual_L2': math.sqrt(rss[name]/yn)} for name in x}
    save_json(out/'RESULT.json', {'physical_release': False, 'rank_with_FM': 10, 'unidentified_kernel_directions': 4,
              'folded_s31_independent_of_nullspace': True, 'samples': g.N, 'metrics': metrics,
              'scope': 's31 folded fields only; no unique14kernel or V_func_tmp emitted; same labels and FM for both fits'})


def execute(binary, directory, text, logname, marker, ranks):
    assert os.environ.get('SLURM_JOB_ID'), 'Compute allocation required'
    with (directory/logname).open('x') as out:
        argv = ['srun', '--exclusive', '-N', '1', '-n', str(ranks), str(binary)]
        if ranks > 1:
            argv += ['-npool', '1', '-ndiag', '1']
        subprocess.run(argv,
                       input=text, text=True, cwd=directory, stdout=out, stderr=subprocess.STDOUT, check=True)
    log = (directory/logname).read_text()
    assert marker in log and 'eigenvalues not converged' not in log, logname


def pp_input(directory):
    return f"&inputpp\n outdir='{directory}'\n prefix='CrI3'\n/\n"


def density(spins, kernel, fm):
    result = np.array(fm)
    result[:, 1:] = kernel@spins
    result[:, 1:3] += fm[:, 1:3]
    return result


def native(spins, kernel, fm, directory):
    directory.mkdir()
    write_field(directory/'DENSITY_ANSATZ.bin', density(spins, kernel, fm))
    execute(P/'density_ansatz_harris_r02/pp_variational.x', directory, pp_input(SEED.parent),
            'export.out', 'CRI3_VARIATIONAL_FIELD_EXPORT_COMPLETE', 1)
    return g.read(directory/'FROZEN_HXC.bin')[0]


def rotate(e, axis, angle):
    return e*math.cos(angle)+np.cross(axis, e)*math.sin(angle)+axis*(axis@e)*(1-math.cos(angle))


def bands_input(save):
    text = (P/'variational_qx_triplet_continue_r01/zero/scf.in').read_text()
    replacements = {'calculation': "'bands'", 'restart_mode': "'from_scratch'", 'outdir': repr(str(save.parent)),
                    'pseudo_dir': repr(str(SEED)), 'disk_io': "'low'", 'wf_collect': '.true.',
                    'startingpot': "'file'", 'startingwfc': "'atomic+random'", 'constrained_magnetization': "'none'",
                    'diago_thr_init': str(PARAM['native_diago_thr']), 'max_seconds': str(PARAM['native_layer1_seconds'])}
    for key, value in replacements.items():
        text, count = re.subn(rf'(?m)^(\s*{key}\s*=)\s*[^\n,]+', lambda m: m[1]+value, text)
        assert count == 1, (key, count)
    return text


def band_data(save):
    band = ET.parse(save/'data-file-schema.xml').find('.//output/band_structure')
    assert int(band.findtext('nks')) == PARAM['s31_kpoints'] and abs(float(band.findtext('nelec'))-PARAM["native_occupied"]) < 1e-9
    weights, high, low = [], [], []
    for row in band.findall('ks_energies'):
        e = 2*np.fromstring(row.findtext('eigenvalues'), sep=' ')
        o = np.fromstring(row.findtext('occupations'), sep=' ')
        assert len(e) == len(o) == PARAM["native_nbnd"] and np.max(abs(o[:PARAM["native_occupied"]]-1)) < 1e-10 and np.max(abs(o[PARAM["native_occupied"]:])) < 1e-10
        weights.append(float(row.find('k_point').attrib['weight']))
        high.append(e[(PARAM["native_occupied"]-1)]); low.append(e[PARAM["native_occupied"]])
    assert len(weights) == PARAM['s31_kpoints'] and abs(sum(weights)-1) < 1e-10 and min(low) > max(high)
    return {'n_kpoints': PARAM['s31_kpoints'], 'k_weight_sum': sum(weights), 'occupied_states_per_k': PARAM["native_occupied"], 'gap_Ry': min(low)-max(high)}


def endpoint(name, case, resume=False, axes_by_name=None):
    assert name in ('nominal', 'actual') and case in ('plus', 'minus')
    assert (R/'fits/RESULT.json').exists()
    out = R/'endpoints'/name/case
    if not resume:
        out.mkdir(parents=True)
    reference = json.loads((R/'REFERENCE.json').read_text())['rows'][case]
    moments = np.array(reference['moments'])
    spins = moments/np.linalg.norm(moments, axis=1)[:, None]
    assert np.isfinite(spins).all() and np.min(spins[:, 2]) >= math.cos(math.radians(3))
    axes = np.tile([-1/math.sqrt(2), 1/math.sqrt(2), 0], (6, 1))
    kernels = {kind: np.load(R/f'fits/{kind}_{name}.npy', mmap_mode='r') for kind in ('density', 'potential')}
    fm = {kind: np.load(R/f'fits/{kind}_FM.npy', mmap_mode='r') for kind in kernels}
    native0 = (g.read(out/'native/FROZEN_HXC.bin')[0] if resume else
               native(spins, kernels['density'], fm['density'], out/'native'))
    rhoin = g.read(out/'native/DENSITY_ANSATZ.bin')[0]
    field = np.array(native0)
    field[:, 1:] = kernels['potential']@spins
    field[:, 1:3] += fm['potential'][:, 1:3]
    cell = np.array([np.fromstring(c.text, sep=' ') for c in ET.parse(SEED/'data-file-schema.xml').find('.//output/atomic_structure/cell')])
    dv = abs(np.linalg.det(cell))/(3*g.N)
    state = out/'state'
    save = state/'tmp/CrI3.save'
    if resume:
        oldlog = (state/'bands.out').read_text()
        assert 'Maximum CPU time exceeded' in oldlog
        assert 'eigenvalues not converged' not in oldlog
        assert 'End of band structure calculation' not in oldlog
        assert (state/'bands.in').read_text() == bands_input(save)
        restart_files = list(save.parent.glob('CrI3.restart_k*'))
        wfcs = list(save.parent.glob('CrI3.wfc*'))
        assert len(restart_files) == len(wfcs) == 16
        assert all(f.stat().st_size > 0 for f in restart_files+wfcs)
        checkpoints = [f.read_text().split() for f in restart_files]
        assert all(len(row) == 3+PARAM['native_nbnd']*PARAM['s31_kpoints'] and 1 <= int(row[0]) < PARAM['s31_kpoints'] for row in checkpoints)
        assert len({row[0] for row in checkpoints}) == 1
        print('RESUME_AFTER_KPOINT', checkpoints[0][0], flush=True)
        evidence = out/'restart_evidence_20260924'; evidence.mkdir()
        for f in restart_files+[save/'data-file-schema.xml']:
            shutil.copy2(f, evidence/f.name)
        save_json(evidence/'FILES.json', {str(f.relative_to(state)): f.stat().st_size for f in wfcs})
        inp = re.sub(r"(restart_mode\s*=\s*)'from_scratch'", r"\1'restart'", (state/'bands.in').read_text())
        assert inp != (state/'bands.in').read_text()
        assert np.array_equal(g.read(state/'FROZEN_HXC.bin')[0], field)
        logname = 'bands.resume_20260924.out'
        (state/'bands.resume_20260924.in').write_text(inp)
    else:
        save.mkdir(parents=True)
        for f in SEED.iterdir():
            assert not f.name.startswith('wfc'), 'Seed must not contain transferred wavefunctions'
            shutil.copy2(f, save/f.name)
        write_field(state/'FROZEN_HXC.bin', field)
        inp = bands_input(save)
        (state/'bands.in').write_text(inp)
        logname = 'bands.out'
    execute(P/'frozen_pw_projection_scan_r01/pw_frozen_wfc.x', state, inp,
            logname, 'End of band structure calculation', PARAM['native_mpi_ranks'])
    assert 'FROZEN_OVERRIDE_BAND_ENERGIES_ONLY' in (state/logname).read_text()
    electronic = band_data(save)
    readout = out/'readout'; readout.mkdir()
    execute(P/'full_k_density_readout_r01/pp_variational.x', readout, pp_input(save.parent),
            'export.out', 'FROZEN_SPIN_COORDINATES_COMPLETE', PARAM['native_mpi_ranks'])
    rho, shape = g.read(readout/'RHO_OUT.bin')
    assert shape == SHAPE and np.isfinite(rho).all()
    if axes_by_name is None:
        axes_by_name = {'': axes}
    for axis_name, axes in axes_by_name.items():
        assert axis_name in ('', 'x', 'y')
        axes = np.asarray(axes, float)
        assert axes.shape == (6, 3) and np.isfinite(axes).all()
        assert np.allclose(np.linalg.norm(axes, axis=1), 1, atol=1e-12, rtol=0)
        rows = []
        h = math.radians(.01)
        for site in range(6):
            sides = []
            for sign, label in ((-1, 'minus'), (1, 'plus')):
                e = spins.copy(); e[site] = rotate(spins[site], axes[site], sign*h)
                sides.append(native(e, kernels['density'], fm['density'], out/f'{axis_name}site{site}_{label}'))
            de = np.cross(axes[site], spins[site])
            deriv = np.empty_like(field)
            deriv[:, 0] = (sides[1][:, 0]-sides[0][:, 0])/(2*h)
            deriv[:, 1:] = kernels['potential'][:, site, None]*de
            drho = kernels['density'][:, site, None]*de
            band = -float(np.sum(rho*deriv))*dv
            work = float(np.sum(rhoin*deriv))*dv
            mismatch = float(np.sum((field[:, 1:]-native0[:, 1:])*drho))*dv
            rows.append({'site': site, 'band': band, 'input_work': work, 'potential_mismatch_work': mismatch,
                         'total': math.fsum((band, work, mismatch))})
            print('TORQUE', name, case, rows[-1], flush=True)
        save_json(out/('TORQUES' + ('_'+axis_name if axis_name else '') + '.json'), {'physical_release': False, 'variant': name, 'case': case,
                  'spins': spins.tolist(), 'axes': axes.tolist(), 'rows_Ry': rows,
                  'site_torque_Ry_per_rad': [r['total'] for r in rows], **electronic,
                  'new_scf': False, 'startingwfc': 'atomic+random; same fixed Hamiltonian and convergence threshold',
                  'scope': 'identifiable s31 folded model, actual test directions, calibration only'})


def analyze():
    fits = json.loads((R/'REFERENCE_FITS.json').read_text())
    ref = json.loads((R/'REFERENCE.json').read_text())
    design = np.array(fits['fits']['old_central_achieved']['design'])
    jref = np.array(fits['fits']['old_central_achieved']['Jspin_meV'])
    h = math.radians(fits['angle_degrees'])
    dft = (np.array(ref['rows']['plus']['site_torque_Ry'])-ref['rows']['minus']['site_torque_Ry'])/(2*h)
    result = {'physical_release': False, 'calibration_only': True, 'DFT_Jspin_meV': jref.tolist(), 'variants': {}}
    for name in ('nominal', 'actual'):
        rows = [json.loads((R/'endpoints'/name/c/'TORQUES.json').read_text()) for c in ('plus', 'minus')]
        for row, case in zip(rows, ('plus', 'minus')):
            m = np.array(ref['rows'][case]['moments']); e = m/np.linalg.norm(m, axis=1)[:, None]
            assert np.allclose(row['spins'], e, atol=1e-14, rtol=0)
        phi = (np.array(rows[0]['site_torque_Ry_per_rad'])-rows[1]['site_torque_Ry_per_rad'])/(2*h)
        coefficients, _, rank, _ = np.linalg.lstsq(design, phi[1:], rcond=None)
        assert rank == 3
        j = -coefficients*13605.693122994/2.25
        relative = abs(j-jref)/abs(jref)
        sites = [1, 2, 4]
        error = float(np.linalg.norm(phi[sites]-dft[sites])/np.linalg.norm(dft[sites]))
        result['variants'][name] = {'Jspin_meV': j.tolist(), 'J_relative_error': relative.tolist(),
                 'dominant_response_error': error, 'response_Ry': phi.tolist(),
                 'pass': bool(max(relative[:2]) <= .02 and error <= .02)}
    result['actual_minus_nominal_Jspin_meV'] = (np.array(result['variants']['actual']['Jspin_meV'])-result['variants']['nominal']['Jspin_meV']).tolist()
    save_json(R/'RESULT.json', result)
    print(json.dumps(result, indent=2))


def check():
    selected, x, inverses = designs()
    sample = np.load(R/'REDUCED_FIT.npz')
    basis = sample['identifiable_basis']
    for kind in ('potential', 'density'):
        y = sample[kind+'_labels']
        base = np.repeat(sample[kind+'_FM'][:, None]/14, 14, axis=1)
        for name in x:
            current = base+(y-base@x[name].T)@inverses[name].T
            expected = base+sample[f'{kind}_{name}_coefficients']@basis.T
            for tile, f in folded_maps():
                assert np.max(abs((current-expected)@f.T)) < 1e-12
    e = np.array([.01, .02, math.sqrt(1-.0005)])
    axis = np.array([0., 1., 0.]); h = 1e-5
    assert np.max(abs((rotate(e, axis, h)-rotate(e, axis, -h))/(2*h)-np.cross(axis, e))) < 1e-9
    print('PASS: saved real-field sampled fits reproduced; nullspace eliminated; local-axis derivative correct')


if __name__ == '__main__':
    command = sys.argv[1]
    if command == 'check': check()
    elif command == 'fit': fit()
    elif command == 'endpoint': endpoint(*sys.argv[2:])
    elif command == 'resume': endpoint(*sys.argv[2:], resume=True)
    elif command == 'analyze': analyze()
    else: raise ValueError(command)
