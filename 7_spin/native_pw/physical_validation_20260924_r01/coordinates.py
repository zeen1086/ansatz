"""Rebuild r02 coordinates only; read existing XML/fields, never submit or refit."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from parameters import load
PARAM = load()
import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.dont_write_bytecode = True
R = Path(__file__).resolve().parent
P = R.parent
REMOTE = PARAM['native_remote_root']
MODEL = P / 'full_spatial_candidate_20260918_r02/model.py'
spec = importlib.util.spec_from_file_location('original_r02', MODEL)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
H = math.sin(math.radians(PARAM['angle_degrees']))
SAMPLES = np.array([0, 123, 7919, 100003, 400001, 800003, 1200001, g.N - 1])


def relative(path):
    return str(path.relative_to(P))


def endpoint(field):
    rel = relative(field)
    if rel.startswith('variational_field_export_r01/'):
        case = field.parent.name
        return {'plus': 'groupA_pool1_ndiag1_plus_r01',
                'minus': 'variational_qx_triplet_continue_r01/minus',
                'zero': 'variational_qx_triplet_continue_r01/zero'}[case]
    if rel.startswith('density_ansatz_harris_r01/roundtrip_check/'):
        return 'variational_qx_triplet_continue_r01/zero'
    assert field.parent.name == 'fields'
    return relative(field.parent.parent)


def collect():
    """Read-only SSH snapshot: whole XML, field record envelopes and selected values."""
    request = {'root': REMOTE, 'fields': {}, 'endpoints': {},
               'model': relative(MODEL)}
    for kind in ('potential', 'density'):
        zeros, rows, _, _ = g.setup(kind)
        for cell, _, plus, minus, u, _ in rows:
            for path in (zeros[cell], plus, minus):
                if path is None:
                    continue
                ep = endpoint(path)
                request['endpoints'][ep] = cell
                request['fields'].setdefault(relative(path), {'endpoint': ep, 'samples': {}})
                request['fields'][relative(path)]['samples'][str(u)] = g.index(cell, u, SAMPLES).tolist()
    code = r'''
from pathlib import Path
import json,struct,hashlib,datetime,sys
q=json.loads(sys.stdin.read());p=Path(q['root'])
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'25_li','root':str(p),'endpoints':{},'fields':{}}
out['model_sha256']=hashlib.sha256((p/q['model']).read_bytes()).hexdigest()
out['density_FM_XML_copies']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p/'density_ansatz_harris_r01/zero/tmp/CrI3.save/data-file-schema.xml',p/'density_ansatz_harris_r01/zero/SCF_REFERENCE.xml']}
for ep,cell in q['endpoints'].items():
 d=p/ep;x=d/'tmp/CrI3.save/data-file-schema.xml'
 row={'cell':cell,'xml_path':str(x),'status':'MISSING'}
 if x.is_file():
  b=x.read_bytes();row.update(status='PRESENT',xml=b.decode(),xml_sha256=hashlib.sha256(b).hexdigest(),xml_mtime_ns=x.stat().st_mtime_ns)
  log=d/'scf.out'
  if log.is_file():
   t=log.read_text(errors='replace');row['scf_log_converged']='convergence has been achieved' in t;row['scf_log_job_done']='JOB DONE.' in t
 out['endpoints'][ep]=row
for name,req in q['fields'].items():
 f=p/name;row={'endpoint':req['endpoint'],'remote_path':str(f),'status':'MISSING'}
 if f.is_file():
  stat=f.stat()
  with f.open('rb') as h:
   header=struct.unpack('<7i',h.read(28));shape=header[1:5];n=shape[0]*shape[1]*shape[2]*shape[3]
   h.seek(-4,2);tail=struct.unpack('<i',h.read(4))[0]
   assert header[0]==header[5]==16 and header[6]==tail==n*8 and stat.st_size==n*8+32
   samples={};ng=n//4
   for tile,ids in req['samples'].items():
    a=[]
    for idx in ids:
     v=[]
     for comp in range(4):
      h.seek(28+8*(idx+comp*ng));v.append(struct.unpack('<d',h.read(8))[0])
     a.append(v)
    samples[tile]=a
  assert f.stat().st_mtime_ns==stat.st_mtime_ns
  row.update(status='PRESENT',shape=shape,size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,samples=samples)
  row['export_evidence']={}
  for name2 in ('print.in','export.in','EXPORT_STATUS.json','export.out'):
   f2=f.parent/name2
   if f2.is_file():
    t=f2.read_text(errors='replace')
    row['export_evidence'][name2]=t if name2!='export.out' else '\n'.join(l for l in t.splitlines() if any(k in l for k in ('Reading xml','directory:','JOB DONE','COMPLETE','RELOAD_MAXDIFF')))
 out['fields'][name]=row
print(json.dumps(out))
'''
    # Input is data; the remote command is fixed Python code, shell-quoted once.
    import shlex
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30',
                             '25_li', 'python3 -c ' + shlex.quote(code)],
                            input=json.dumps(request), capture_output=True, text=True, check=True)
    snapshot = json.loads(result.stdout)
    assert snapshot['model_sha256'] == hashlib.sha256(MODEL.read_bytes()).hexdigest()
    with (R / 'SOURCE_SNAPSHOT.json').open('x') as f:
        json.dump(snapshot, f, indent=2)
    print('Snapshot:', len(snapshot['endpoints']), 'XML endpoints,', len(snapshot['fields']), 'field files')


def spins(record):
    assert record['status'] == 'PRESENT', record['xml_path']
    x = ET.fromstring(record['xml'])
    out = x.find('output')
    converged = out.findtext('.//convergence_achieved').strip() == 'true'
    if not converged:
        limit = record.get('preliminary_threshold', 0.)
        assert 0 < limit <= 1e-7 and int(out.findtext('.//n_scf_steps')) > 0
        assert 0 < 2*float(out.findtext('.//scf_error')) <= limit
    assert out.findtext('magnetization/spinorbit').strip() == 'false'
    sites = out.findall('magnetization/Site_Magnetizations/SiteMagnetization')
    cr = [s for s in sites if s.attrib['species'].startswith('Cr')]
    assert [int(s.attrib['atom']) for s in cr] == list(range(1, 7))
    moment = np.array([np.fromstring(s.text, sep=' ') for s in cr])
    norms = np.linalg.norm(moment, axis=1)
    assert moment.shape == (6, 3) and np.isfinite(moment).all() and min(norms) > 0
    e = moment / norms[:, None]
    atoms = out.find('atomic_structure')
    lattice = np.array([np.fromstring(v.text, sep=' ') for v in atoms.find('cell')])
    coords = [a for a in atoms.find('atomic_positions') if a.attrib['name'].startswith('Cr')]
    assert [int(a.attrib['index']) for a in coords] == list(range(1, 7))
    frac = np.array([np.fromstring(a.text, sep=' ') for a in coords]) @ np.linalg.inv(lattice)
    mat, tiles, _ = g.CELLS[record['cell']]
    primitive = frac[:, :2] @ mat
    expected = np.repeat(tiles, 2, axis=0) + np.tile([[1/3, 2/3], [2/3, 1/3]], (3, 1))
    delta = (primitive - expected) @ np.linalg.inv(mat)
    geometry_error = float(np.max(abs(delta - np.rint(delta))))
    assert geometry_error < 1e-8, (record['xml_path'], geometry_error)
    target = []
    species = x.findall('input/atomic_species/species')
    for atom in coords:
        s = next(s for s in species if s.attrib['name'] == atom.attrib['name'])
        # QE XML stores radians; omitted angles are the collinear +z default.
        theta, phi = [float(s.findtext(k, '0')) for k in ('spin_teta', 'spin_phi')]
        target.append([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)])
    target = np.array(target)
    return e, {'xml_path': record['xml_path'], 'xml_sha256': record['xml_sha256'],
               'true_scf': converged, 'scf_error_Ry': 2*float(out.findtext('.//scf_error')),
               'geometry_error_fractional': geometry_error, 'moments_muB': moment.tolist(),
               'unit_spins': e.tolist(), 'target_spins': target.tolist(),
               'moment_norms_muB': norms.tolist(),
               'max_direction_error_deg': float(np.degrees(np.max(np.arccos(np.clip(np.sum(e*target, axis=1), -1, 1)))))}


def site_map(cell, tile):
    mat, tiles, _ = g.CELLS[cell]
    result = []
    for tx, ty, b in g.CHANNELS:
        d = (np.array([tx, ty]) + tile - np.array(tiles)) @ np.linalg.inv(mat)
        matches = np.flatnonzero(np.max(abs(d-np.rint(d)), axis=1) < 1e-10)
        assert len(matches) == 1
        result.append(2*int(matches[0])+b)
    return result


def spectrum(a):
    s = np.linalg.svd(a, compute_uv=False)
    return {'shape': list(a.shape), 'singular_values': s.tolist(),
            'rank_relative': {str(t): int(np.sum(s > t*s[0])) for t in (1e-6, 1e-8, 1e-10, 1e-12)},
            'condition_full': float(s[0]/s[-1]), 'condition_nonnull_13': float(s[0]/s[12])}


def rebuild():
    snapshot = json.loads((R / 'SOURCE_SNAPSHOT.json').read_text())
    identity = snapshot.get('density_FM_XML_copies')
    if identity is None:
        identity = json.loads((R / 'FM_XML_IDENTITY.json').read_text())
    fm_hash = snapshot['endpoints']['variational_qx_triplet_continue_r01/zero']['xml_sha256']
    assert set(identity.values()) == {fm_hash}, 'Density FM copied XML differs from original SCF.'
    states, evidence = {}, {}
    for ep, rec in snapshot['endpoints'].items():
        states[ep], evidence[ep] = spins(rec)
    zeros, rows, nominal, _ = g.setup('potential')
    actual, raw, manifests = [], [], []
    sampled = {k: [] for k in ('potential', 'density')}
    for j, (cell, mode, plus, minus, tile, comp) in enumerate(rows):
        mapping = site_map(cell, tile)
        positive = plus if plus is not None else zeros[cell]
        denominator = (2 if plus is not None else 1)*H
        de = states[endpoint(positive)] - states[endpoint(minus)]
        vector = de[mapping, comp-1]
        actual.append(vector/denominator);raw.append(vector)
        pairs = {}
        for kind in sampled:
            z, rr, nx, _ = g.setup(kind)
            assert np.array_equal(nx, nominal)
            _, _, pp, mm, tt, cc = rr[j]
            pp = pp if pp is not None else z[cell]
            aa, bb = [snapshot['fields'][relative(p)] for p in (pp, mm)]
            assert aa['status'] == bb['status'] == 'PRESENT'
            for f in (aa, bb):
                rec = snapshot['endpoints'][f['endpoint']]
                assert f['mtime_ns'] >= rec['xml_mtime_ns']
                assert 'JOB DONE.' in f['export_evidence']['export.out']
            expected = [3*PARAM['nr1'], PARAM['nr2'], PARAM['nr3'], 4] if cell == 's31' else [PARAM['nr1'], 3*PARAM['nr2'], PARAM['nr3'], 4]
            assert aa['shape'] == bb['shape'] == expected
            assert (endpoint(pp), endpoint(mm)) == (endpoint(positive), endpoint(minus))
            y = (np.array(aa['samples'][str(tt)])[:,cc] - np.array(bb['samples'][str(tt)])[:,cc])/denominator
            assert np.isfinite(y).all()
            sampled[kind].append(y)
            pairs[kind] = {'positive_field': aa['remote_path'], 'negative_field': bb['remote_path'],
                           'field_component_zero_based': cc, 'denominator': denominator,
                           'positive_xml': evidence[endpoint(pp)]['xml_path'],
                           'negative_xml': evidence[endpoint(mm)]['xml_path']}
        manifests.append({'row': j, 'cell': cell, 'mode': mode, 'tile': list(tile),
                          'component': 'xy'[comp-1], 'Cr_atom_indices_one_based': [i+1 for i in mapping],
                          'difference': 'central' if plus is not None else 'FM-minus', 'pairs': pairs,
                          'longitudinal_difference_max': float(np.max(abs(de[:,2])))})
    actual, raw = np.array(actual), np.array(raw)
    # FM longitudinal anchor follows the SAME three-window average as the original field.
    fm = states[endpoint(zeros['s31'])]
    anchor = np.mean([fm[site_map('s31', u), 2] for u in g.CELLS['s31'][1]], axis=0)
    projector = np.eye(14) - np.outer(anchor, anchor)/(anchor@anchor)
    inverse = projector @ np.linalg.pinv(actual @ projector, rcond=1e-10)
    mode_metrics = {}
    for cell, mode in dict.fromkeys((r[0], r[1]) for r in rows):
        ix = [i for i,r in enumerate(rows) if r[:2] == (cell,mode)]
        a, n = actual[ix], nominal[ix]
        scale = np.sum(a*n)/np.sum(n*n)
        mode_metrics[cell+'_'+mode] = {'transverse_scale': float(scale),
                                      'nonuniform_residual_after_scale': float(np.linalg.norm(a-scale*n)/np.linalg.norm(a))}
    # One runnable regression: endpoint folding must reproduce every old analytic row;
    # arbitrary kernels must survive the actual constrained inverse and label units.
    for j, row in enumerate(rows):
        cell, mode, plus, minus, tile, comp = row
        pp = plus if plus is not None else zeros[cell]
        a = np.array(evidence[endpoint(pp)]['target_spins'])
        b = np.array(evidence[endpoint(minus)]['target_spins'])
        assert np.max(abs((a-b)[site_map(cell,tile),comp-1]/((2 if plus else 1)*H)-nominal[j])) < 1e-10
    kernel = np.arange(14, dtype=float)**2/13
    base = anchor*(anchor@kernel)/(anchor@anchor)
    recovered = base + inverse@(actual@kernel-actual@base)
    assert np.max(abs(recovered-kernel)) < 1e-10
    assert np.max(abs(raw/np.array([m['pairs']['potential']['denominator'] for m in manifests])[:,None]-actual)) < 1e-14
    assert np.max(abs(anchor-1)) < 1e-10
    no_sd = np.array([r[0] != 'sd' for r in rows])
    result = {'classification': 'ACTUAL_SPIN_TRAINING_DESIGN_DIAGNOSTIC',
              'source_model': str(MODEL), 'snapshot_utc': snapshot['utc'], 'training_rows': len(rows),
              'endpoints': len(evidence), 'fields': len(snapshot['fields']),
              'nominal': spectrum(np.vstack([nominal, np.ones(14)])),
              'actual_with_FM_anchor': spectrum(np.vstack([actual, anchor])),
              'actual_projected': spectrum(actual@projector), 'actual_transverse_only': spectrum(actual),
              'without_original_SD': spectrum(np.vstack([actual[no_sd], anchor])),
              'parameter_scope': '14 shared scalar kernels; 42 serialized functions have tied Cartesian blocks, not 42 independent fitted kernels.',
              'mode_metrics': mode_metrics, 'FM_anchor': anchor.tolist(),
              'FM_transverse_max': float(np.max(abs(fm[:,:2]))),
              'relative_matrix_change': float(np.linalg.norm(actual-nominal)/np.linalg.norm(nominal)),
              'nominal_projected': spectrum(nominal@projector),
              'density_FM_XML_copy_identity': identity,
              'synthetic_recovery_maxerror': float(np.max(abs(recovered-kernel))),
              'limits': ['Original r02 fields and representation only; SD original FFT issue is retained, not repaired by coordinates.',
                         'Rank is conditional on the existing FM anchor and isotropic pure-S representation.',
                         'One-sided optical responses remain one-sided; no new endpoints or independent validation.',
                         'Field payload pairing checked at eight primitive grid samples; full-field residual/refit NOT_RUN.',
                         'No new kernel, RVR, torque, Phi2 or J; no uniform rescaling of J.']}
    np.savez(R/'DESIGN.npz', actual=actual, nominal=nominal, raw_spin_differences=raw,
             FM_anchor=anchor, constrained_inverse=inverse, sample_primitive_indices=SAMPLES,
             **{k+'_sample_labels': np.array(v).T for k,v in sampled.items()})
    for filename, obj in [('RESULT.json', result), ('ENDPOINTS.json', evidence), ('ROW_PAIRING.json', manifests)]:
        (R/filename).write_text(json.dumps(obj, indent=2)+'\n')
    np.savetxt(R/'actual_design.csv', actual, delimiter=',', header=','.join(map(str,g.CHANNELS)))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args()
    if args.collect:
        collect()
    rebuild()
