#!/usr/bin/env python3
"""Direct three-cell transverse response; frozen J, no SCF and no refitting."""
from pathlib import Path
import argparse
import json
import math
import os
import shutil
import sys
import types

os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import numpy as np
import torque_s13 as t

ROOT, OLD, reader, shared, l = t.ROOT, t.OLD, t.reader, t.shared, t.l
R = ROOT / 'runs/full14_response_matrix_20260926_r01'
JFILE = ROOT / 'data/current14/three_cell_response/J123.json'
PARAM = t.PARAM
JHASH = PARAM['response_j_sha256']
AXES = {'x': np.tile([0., 1., 0.], (6, 1)), 'y': np.tile([-1., 0., 0.], (6, 1))}
CELLS = {
    's31': ((3*PARAM['nr1'], PARAM['nr2'], PARAM['nr3']), OLD/'variational_qx_triplet_continue_r01/zero', OLD/'coordinate_layer1_issp_20260923_r01/fits', PARAM['s31_kpoints']),
    's13': ((PARAM['nr1'], 3*PARAM['nr2'], PARAM['nr3']), OLD/'spatial_completion_continue_20260918_r01/s13_zero', ROOT/'runs/full14_s13_20260926/fits', PARAM['s13_kpoints']),
    'sd': ((3*PARAM['nr1'], 3*PARAM['nr2'], PARAM['nr3']), OLD/'sd_fft_repair_20260921_r01/sd_zero', ROOT/'data/current14/sd_folded', PARAM['sd_kpoints']),
}


def cases():
    # 72 full-matrix endpoints plus one half-step column pair per cell.
    half_angle = PARAM['angle_degrees']/2
    return [{'cell': c, 'axis': a, 'site': i, 'degrees': h, 'side': side,
             'id': f'{c}_{a}{i}_{"half" if h == half_angle else "full"}_{side}'}
            for c in CELLS for a, i, h in
            [(a, i, PARAM['angle_degrees']) for a in AXES for i in range(6)] + [('x', 0, PARAM['angle_degrees']/2)]
            for side in ('plus', 'minus')]


def spins(case):
    e = np.tile([0., 0., 1.], (6, 1))
    sign = 1 if case['side'] == 'plus' else -1
    e[case['site']] = l.rotate(e[case['site']], AXES[case['axis']][case['site']], sign*math.radians(case['degrees']))
    return e


def model(cell, J, S):
    m, sites, _ = shared.a.geometry(cell)
    n = shared.a.counts(m, sites)[:, :, :3]
    adjacency = n @ J
    k = S*S*(adjacency-np.diag(adjacency.sum(axis=1)))
    return np.kron(np.eye(2), k), n


def check():
    J = np.array([-4.7, -.55, .09]); S = 1.5; h = 1e-5
    assert len(cases()) == 78 and len({c['id'] for c in cases()}) == 78
    for cell in CELLS:
        K, n = model(cell, J, S)
        direct = np.zeros((12, 12))
        for col in range(12):
            a, site = ('x' if col < 6 else 'y'), col % 6
            e = np.tile([0., 0., 1.], (6, 1)); ep=e.copy(); em=e.copy()
            ep[site]=l.rotate(e[site],AXES[a][site],h);em[site]=l.rotate(e[site],AXES[a][site],-h)
            direct[:,col] = -reader.RY_MEV*np.concatenate([
                (reader.torque_design(n,ep,ax,S)-reader.torque_design(n,em,ax,S))@J/(2*h) for ax in AXES.values()])
        assert np.max(abs(direct-K)) < 1e-8
        assert np.max(abs(K-K.T)) < 1e-12 and np.max(abs(K[:6,:6].sum(axis=1))) < 1e-12
        assert np.linalg.eigvalsh(K).min() > -1e-9
    # New readout shares the unchanged three-term native kernel, once per axis.
    import inspect
    assert 'axes_by_name' in inspect.signature(l.endpoint).parameters
    print('PASS: 78 unique endpoints, transverse ordering, torque sign/S^2, periodic self-bond cancellation')


def prepare():
    assert os.environ.get('PBS_JOBID')
    check(); assert reader.sha256(JFILE) == JHASH
    R.mkdir(parents=True, exist_ok=False)
    shutil.copy2(JFILE, R/'J123_frozen.json')
    for path in (Path(__file__), Path(l.__file__), Path(t.__file__)):
        shutil.copy2(path, R/('source_'+path.name))
    pins = {str(JFILE): JHASH}
    eq = json.loads((ROOT/'data/current14/s31_response/FULL14_EQUIVALENCE.json').read_text())
    for path, value in eq['source_sha256'].items():
        if path.endswith('.npy'):
            assert reader.sha256(Path(path)) == value
    for cell, (shape, base, fits, nk) in CELLS.items():
        d=R/cell; d.mkdir(); seed=d/'seed/tmp/CrI3.save';seed.mkdir(parents=True)
        src=base/'tmp/CrI3.save'; files=list(src.glob('wfc*.dat'));assert len(files)==nk
        for p in src.iterdir():
            if p.name in ('data-file-schema.xml','charge-density.dat') or p.suffix.lower()=='.upf':
                shutil.copy2(p,seed/p.name)
        assert (seed/'charge-density.dat').is_file() and (seed/'data-file-schema.xml').is_file()
        shutil.copy2(base/'scf.in',d/'source.in')
        (d/'fits').mkdir()
        for kind in ('potential','density'):
            for suffix in ('actual','FM'):
                path=fits/f'{kind}_{suffix}.npy'; a=np.load(path,mmap_mode='r')
                assert a.shape == (math.prod(shape),6 if suffix=='actual' else 4)
                pins[str(path)] = reader.sha256(path)
                (d/'fits'/path.name).symlink_to(path)
        reader.write_json(d/'fits/RESULT.json',{'method':'Reuse current full14 folds; no fit'})
    for c in cases():
        d=R/c['id'];d.mkdir();(d/'fits').symlink_to(R/c['cell']/'fits')
        reader.write_json(d/'REFERENCE.json',{'rows':{c['side']:{'moments':spins(c).tolist()}}})
    reader.write_json(R/'INPUTS.json',{'pins':pins,'cases':cases(),'J_sha256':JHASH,'new_scf':False,
        'coordinates':'x0..x5,y0..y5; axes +y,-x; K=-d(tau)/d(theta), meV/rad^2',
        'A_meV_external':.3627,'A_in_direct_response':0.,
        'scope':'72 direct matrix endpoints; 6 half-step endpoints. No new fit, no Phi4/rates.',
        'storage':'Private newly copied WFC files removed only after both torque readouts validate; fields and logs retained.'})
    print('RESPONSE_MATRIX_PREPARED',flush=True)


def endpoint(index):
    assert os.environ.get('PBS_JOBID')
    c=cases()[index]; shape, base, fits, nk=CELLS[c['cell']]
    manifest=json.loads((R/'INPUTS.json').read_text())
    for path, value in manifest['pins'].items():
        if path==str(JFILE) or Path(path).parent==fits:
            assert reader.sha256(Path(path))==value, path
    d=R/c['id'];seed=R/c['cell']/'seed/tmp/CrI3.save'
    t.R=R/c['cell'];t.SEED=seed
    def bands_input(save):
        text=(R/c['cell']/'source.in').read_text()
        for key,value in {'calculation':"'bands'",'restart_mode':"'from_scratch'",'outdir':repr(str(save.parent)),
            'pseudo_dir':repr(str(seed)),'disk_io':"'low'",'wf_collect':'.true.','startingpot':"'file'",
            'startingwfc':"'file'",'constrained_magnetization':"'none'",'diago_thr_init':str(PARAM['native_diago_thr']),'max_seconds':str(PARAM['native_max_seconds'])}.items():
            text=shared.campaign.replace(text,key,value)
        return text
    def execute(binary,directory,text,logname,marker,ranks):
        if binary.name=='pw_frozen_wfc.x':
            files=list((base/'tmp/CrI3.save').glob('wfc*.dat'));assert len(files)==nk
            assert shutil.disk_usage(d).free>sum(p.stat().st_size for p in files)+30_000_000_000
            for src in files:
                dst=directory/'tmp/CrI3.save'/src.name;assert not dst.exists()
                before=src.stat();shutil.copy2(src,dst)
                assert dst.stat().st_ino!=before.st_ino and dst.stat().st_size==before.st_size
                assert src.stat().st_mtime_ns==before.st_mtime_ns
        # The existing checked MPI executor handles all native binaries and markers.
        if binary.name=='pw_frozen_wfc.x':
            import subprocess
            with (directory/logname).open('x') as stream:
                subprocess.run(['mpirun','-np',str(ranks),str(binary),'-npool','1','-ndiag','1'],cwd=directory,
                    input=text,text=True,stdout=stream,stderr=subprocess.STDOUT,check=True)
            log=(directory/logname).read_text()
            assert marker in log and 'eigenvalues not converged' not in log
        else:
            shared.execute(binary,directory,text,logname,marker,ranks)
    l.R=d;l.SEED=seed;l.SHAPE=shape
    l.g=types.SimpleNamespace(read=shared.f.g.read,N=math.prod(shape)//3)
    l.execute=execute;l.bands_input=bands_input
    l.band_data=lambda save:shared.w.band_data(save/'data-file-schema.xml',nk)
    original_save=l.save_json
    def metadata(path,value):
        if path.name.startswith('TORQUES'):
            value.update(cell=c['cell'],grid=list(shape),source_candidate=str(ROOT/'data/current14'),
                         startingwfc='private copy of existing FM wavefunctions',scope='Direct two-axis fixed-J validation',
                         perturbation=c,J_sha256=JHASH)
        original_save(path,value)
    l.save_json=metadata
    l.endpoint('actual',c['side'],axes_by_name=AXES)
    out=d/f'endpoints/actual/{c["side"]}'
    for a in AXES:
        e,ax,_,_=reader.read_torque(out/f'TORQUES_{a}.json')
        assert np.allclose(e,spins(c),atol=1e-14,rtol=0) and np.array_equal(ax,AXES[a])
    # Only disposable WFC copies created by this job, never a source directory.
    save=out/'state/tmp/CrI3.save';assert save.is_relative_to(d) and not save.is_symlink()
    for path in save.glob('wfc*.dat'):
        assert path.is_file() and not path.is_symlink();path.unlink()
    reader.write_json(d/'COMPLETE.json',{'case':c,'private_WFC_removed':True,
        'torque_sha256':{a:reader.sha256(out/f'TORQUES_{a}.json') for a in AXES}})
    print('RESPONSE_ENDPOINT_COMPLETE',c['id'],flush=True)


def analyze():
    assert reader.sha256(JFILE)==JHASH
    j=json.loads((R/'J123_frozen.json').read_text());J=np.array([j['J_meV'][f'J{i}'] for i in (1,2,3)]);S=j['spin_S']
    result={'J_refitted':False,'J_sha256':JHASH,'cells':{},'missing':[]}
    values={}
    for c in cases():
        d=R/c['id'];p=d/'COMPLETE.json'
        if not p.exists():result['missing'].append(c['id']);continue
        cert=json.loads(p.read_text());v=[]
        for a in AXES:
            path=d/f'endpoints/actual/{c["side"]}/TORQUES_{a}.json'
            assert reader.sha256(path)==cert['torque_sha256'][a]
            e,ax,tau,_=reader.read_torque(path)
            assert np.allclose(e,spins(c),atol=1e-14,rtol=0) and np.array_equal(ax,AXES[a])
            v.extend(tau)
        values[c['cell'],c['axis'],c['site'],c['degrees'],c['side']]=np.array(v)
    for cell in CELLS:
        selected=[c for c in cases() if c['cell']==cell]
        if any(c['id'] in result['missing'] for c in selected):continue
        def column(a,i,h):
            return -reader.RY_MEV*(values[cell,a,i,h,'plus']-values[cell,a,i,h,'minus'])/(2*math.radians(h))
        K=np.column_stack([column(a,i,PARAM['angle_degrees']) for a in AXES for i in range(6)])
        pred,n=model(cell,J,S);norm=np.linalg.norm(K)
        symplectic=np.block([[np.zeros((6,6)),np.eye(6)],[-np.eye(6),np.zeros((6,6))]])
        # Raw matrix is retained; no symmetrization or sum-rule projection before comparison.
        modes={}
        for label,k in [('direct',K),('fixed_J',pred)]:
            for A in (0.,PARAM['response_anisotropy_mev']):
                w=np.linalg.eigvals(1j*symplectic@(k+S*(2*S-1)*A*np.eye(12))/S)
                modes[f'{label}_A{A}']={'eigenvalues_real_meV':sorted(w.real.tolist()),'max_imaginary_meV':float(max(abs(w.imag)))}
        uniform=np.column_stack([np.r_[np.ones(6),np.zeros(6)],np.r_[np.zeros(6),np.ones(6)]])/math.sqrt(6)
        result['cells'][cell]={'K_direct_meV':K.tolist(),'K_fixed_J_meV':pred.tolist(),
            'relative_error':float(np.linalg.norm(K-pred)/norm),
            'raw_reciprocity_error':float(np.linalg.norm(K-K.T)/norm),
            'uniform_rotation_residual':float(np.linalg.norm(K@uniform)/norm),
            'xy_block_difference':float(np.linalg.norm(K[:6,:6]-K[6:,6:])/norm),
            'cross_xy_fraction':float(np.linalg.norm(K[:6,6:])/norm),
            'half_step_column_error':float(np.linalg.norm(column('x',0,PARAM['angle_degrees']/2)-K[:,0])/np.linalg.norm(K[:,0])),
            'modes':modes}
    result['status']='COMPLETE' if not result['missing'] else 'INCOMPLETE'
    result['limits']=['Direct matrices test full14-to-J compression, not independent DFT physical validity.',
        'Only supercell-compatible modes are checked. Literature A is external; no SOC or Phi4 validation.',
        'Half-step check samples x0 only; full elementwise step convergence is not asserted.']
    reader.write_json(R/'RESULT.json',result)
    print(json.dumps({'status':result['status'],'missing':len(result['missing']),'cells':{k:{n:v for n,v in row.items() if 'error' in n} for k,row in result['cells'].items()}}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('check','prepare','endpoint','analyze'));p.add_argument('index',type=int,nargs='?');a=p.parse_args()
    if a.action=='endpoint':
        if a.index is None or not 0<=a.index<len(cases()):p.error('endpoint index must be 0..77')
        endpoint(a.index)
    else:globals()[a.action]()
