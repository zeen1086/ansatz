"""Actual-coordinate full14 fit; held-out single-site SD field predictions never enter fit."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from parameters import load
PARAM = load()
import hashlib,json,math,sys
import numpy as np
import coordinates as c
from run import write
R=Path(__file__).resolve().parent;P=R.parent;g=c.g

def directory(name):
    if name=='sd_ac_minus' and (R/'DEADLINE_POLICY.json').exists():
        return R/'sd_retry'/name
    return R/('sd_retry' if name=='sd_ac_minus' and (R/'SD_RETRY.json').exists() else 'sd')/name

def state(name):
    p=directory(name)
    result=json.loads((p/'RESULT.json').read_text())
    threshold=result.get('acceptance_scf_threshold_Ry') or 1e-10
    assert threshold<=1e-7 and result['scf_error_Ry']<=threshold
    f=p/'tmp/CrI3.save/data-file-schema.xml';data=f.read_bytes()
    return c.spins({'status':'PRESENT','xml':data.decode(),'xml_path':str(f),'xml_sha256':hashlib.sha256(data).hexdigest(),'cell':'sd','preliminary_threshold':threshold if result.get('preliminary') else 0.})[0]

def design():
    _,rows,_,_=g.setup('potential')
    x=np.load(P/'coordinate_layer1_issp_20260923_r01/DESIGN.npz')['actual'].copy()
    states={name:state(name) for name in ('sd_ac_plus','sd_ac_minus','sd_op_plus','sd_op_minus')}
    for i,(cell,mode,pp,mm,tile,comp) in enumerate(rows):
        if cell=='sd':x[i]=(states['sd_'+mode+'_plus']-states['sd_'+mode+'_minus'])[c.site_map(cell,tile),comp-1]/(2*c.H)
    proj=np.eye(14)-np.ones((14,14))/14
    s=np.linalg.svd(x@proj,compute_uv=False);assert np.sum(s>s[0]*1e-8)==13
    return x,proj@np.linalg.pinv(x@proj,rcond=1e-8)

def blind_design():
    _,rows,_,_=g.setup('potential');a=np.load(P/'coordinate_layer1_issp_20260923_r01/DESIGN.npz')['actual']
    proj=np.eye(14)-np.ones((14,14))/14
    _,s,vt=np.linalg.svd(a[[i for i,row in enumerate(rows) if row[0]!='sd']]@proj,full_matrices=False)
    assert sum(s>s[0]*1e-8)==9
    null=proj-vt[:9].T@vt[:9]
    tests=[]
    for site in (0,1):
        for tile in g.CELLS['sd'][1]:
            tests.append(np.array([int(j==site) for j in c.site_map('sd',tile)]))
    coverage=np.array(tests)@null
    assert np.linalg.matrix_rank(coverage,tol=1e-8)==4
    return {'refs':[0,1],'angles_degrees':[-PARAM['angle_degrees'],PARAM['angle_degrees']],'old_nullspace_coverage_rank':4,'criteria':{'potential_response_relative_L2':.02,'density_response_relative_L2':.02},'role':'field-level independent predictions; downstream torque acceptance still required'}

def fit():
    out=R/'fit14';out.mkdir()
    x,inverse=design();singular=np.linalg.svd(np.vstack([x,np.ones(14)]),compute_uv=False)
    write(out/'FROZEN_DESIGN.json',{'rank_with_FM':14,'singular_values':singular.tolist(),'condition':float(singular[0]/singular[-1]),'blind_design':blind_design()})
    np.save(out/'actual_design.npy',x)
    metrics={}
    for kind in ('potential','density'):
        zeros,rows,_,_=g.setup(kind);suffix='RAW_DENSITY.bin' if kind=='density' else 'FROZEN_HXC.bin'
        updated=[]
        for cell,mode,pp,mm,tile,comp in rows:
            if cell=='sd':pp,mm=[directory('sd_'+mode+'_'+side)/'physical_samples'/suffix for side in ('plus','minus')]
            updated.append((cell,mode,pp,mm,tile,comp))
        paths={zeros['s31']}|{p if p else zeros[cell] for cell,mode,pp,mm,tile,comp in updated for p in (pp,mm)}
        data={p:g.read(p)[0] for p in paths}
        k=np.lib.format.open_memmap(out/(kind+'_kernel.npy'),mode='w+',dtype='<f8',shape=(g.N,14))
        fm=np.lib.format.open_memmap(out/(kind+'_FM.npy'),mode='w+',dtype='<f8',shape=(g.N,4))
        rss=yn=0.; rowrss=np.zeros(len(rows));rowyn=np.zeros(len(rows))
        for start in range(0,g.N,PARAM['native_block']):
            ids=np.arange(start,min(start+PARAM['native_block'],g.N));ix={(cell,tile):g.index(cell,tile,ids) for cell in g.CELLS for tile in g.CELLS[cell][1]}
            ref=np.mean([data[zeros['s31']][ix['s31',u]] for u in g.CELLS['s31'][1]],axis=0);fm[ids]=ref
            y=np.column_stack([(data[pp if pp else zeros[cell]][ix[cell,tile],comp]-data[mm][ix[cell,tile],comp])/((2 if pp else 1)*c.H) for cell,mode,pp,mm,tile,comp in updated])
            base=np.repeat(ref[:,3,None]/14,14,axis=1);coef=base+(y-base@x.T)@inverse.T
            assert np.max(abs(coef.sum(axis=1)-ref[:,3]))<1e-10
            k[ids]=coef;err=coef@x.T-y;rowrss+=np.sum(err*err,axis=0);rowyn+=np.sum(y*y,axis=0)
        k.flush();fm.flush();metrics[kind]={'relative_L2':float(np.sqrt(rowrss.sum()/rowyn.sum())),'row_relative_L2':np.sqrt(rowrss/np.maximum(rowyn,1e-300)).tolist()}
    write(out/'RESULT.json',{'preliminary':(R/'DEADLINE_POLICY.json').exists(),'rank_with_FM':14,'metrics':metrics,'scope':'full14 actual-coordinate fitted candidate; blind data excluded; no production interface serialized'})
    print('FULL14_FIT_COMPLETE',flush=True)

def validate():
    assert (R/'fit14/RESULT.json').exists()
    pairs={site:[state('blind'+str(site)+'_'+side) for side in ('plus','minus')] for site in (0,1)}
    result={'preliminary':(R/'DEADLINE_POLICY.json').exists(),'field_only':True,'blind_design':blind_design(),'metrics':{}}
    for kind in ('potential','density'):
        suffix='RAW_DENSITY.bin' if kind=='density' else 'FROZEN_HXC.bin'
        k=np.load(R/'fit14'/(kind+'_kernel.npy'),mmap_mode='r');rss=yn=0.
        for site in (0,1):
            fields=[g.read(R/'sd'/('blind'+str(site)+'_'+side)/'physical_samples'/suffix)[0] for side in ('plus','minus')]
            de=pairs[site][0]-pairs[site][1]
            for start in range(0,g.N,PARAM['native_block']):
                ids=np.arange(start,min(start+PARAM['native_block'],g.N))
                for tile in g.CELLS['sd'][1]:
                    ix=g.index('sd',tile,ids);delta=de[c.site_map('sd',tile),:2]
                    expected=fields[0][ix,1:3]-fields[1][ix,1:3];pred=k[ids]@delta
                    rss+=float(np.sum((pred-expected)**2));yn+=float(np.sum(expected**2))
        err=math.sqrt(rss/yn);result['metrics'][kind]={'relative_response_L2':err,'pass':err<=.02}
    result['pass']=all(row['pass'] for row in result['metrics'].values())
    write(R/'BLIND_FIELD_RESULT.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
    if sys.argv[1]=='check':print(json.dumps(blind_design(),indent=2))
    elif sys.argv[1]=='fit':fit()
    elif sys.argv[1]=='validate':validate()
    else:raise ValueError(sys.argv[1])
