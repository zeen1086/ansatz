"""14-channel candidate with corrected SD labels and explicit physical-G projection.

The serialized primitive grid and index() remain coarse. SD native evaluation
uses lift_sd() on the assembled density, magnetic field AND their derivatives.
This changes pointwise fields; it is not raw-HXC lossless interpolation.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from parameters import load
PARAM = load()
import sys,json,math,struct,importlib.util
from functools import lru_cache
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent
T=np.array([[0,0],[0,-1],[1,0],[1,1],[-1,0],[0,1],[-1,-1]])
CHANNELS=[(int(t[0]),int(t[1]),b) for t in T for b in range(2)]
CELLS={'s31':(np.array([[3,0],[0,1]]),[(0,0),(1,0),(2,0)],(1/3,0)), 's13':(np.array([[1,0],[0,3]]),[(0,0),(0,1),(0,2)],(0,1/3)), 'sd':(np.array([[1,-1],[0,3]]),[(0,0),(0,1),(0,-1)],(1/3,1/3))}
G=(PARAM['nr1'],PARAM['nr2'],PARAM['nr3']);N=math.prod(G)
REPAIR=P/'sd_fft_repair_20260921_r01'
spec=importlib.util.spec_from_file_location('sd_physical_resample',REPAIR/'resample.py')
resample=importlib.util.module_from_spec(spec);spec.loader.exec_module(resample)
def native_shape(cell):
 return resample.FINE if cell=='sd' else ((3*PARAM['nr1'],PARAM['nr2'],PARAM['nr3']) if cell=='s31' else (PARAM['nr1'],3*PARAM['nr2'],PARAM['nr3']))
def reference_tmp(cell):
 return (REPAIR/'sd_zero' if cell=='sd' else P/('variational_qx_triplet_continue_r01/zero' if cell=='s31' else 'spatial_completion_continue_20260918_r01/s13_zero'))/'tmp'
@lru_cache(maxsize=1)
def physical_millers():
 mill,_=resample.read_millers(REPAIR/'fixed_fm/SEED_CHARGE.dat')
 resample.validate_millers(mill,resample.COARSE,resample.FINE)
 return mill
def lift_sd(field,mill=None,coarse=resample.COARSE,fine=resample.FINE,metrics=None):
 """Same linear map for density, magnetic field (including FM offset), and derivatives."""
 assert field.shape==(math.prod(coarse),4) and field.dtype==np.dtype('float64')
 if mill is None:mill=physical_millers()
 out=np.empty((math.prod(fine),4),order='F')
 for comp in range(4):
  value,report=resample.convert(field[:,comp].reshape(coarse,order='F'),mill,'lift',coarse,fine)
  out[:,comp]=value.ravel(order='F')
  if metrics is not None:metrics.append({'component':comp,**report})
 return out
def write_field(dest,field,shape):
 assert field.shape==(math.prod(shape),4) and np.isfinite(field).all()
 with Path(dest).open('xb') as f:
  f.write(struct.pack('<7i',16,*shape,4,16,field.nbytes))
  for column in field.T:f.write(np.asarray(column,dtype='<f8').tobytes())
  f.write(struct.pack('<i',field.nbytes))
 actual,actual_shape=read(dest);assert actual_shape==shape and np.array_equal(actual,field)
def read(f):
 with Path(f).open('rb') as h:
  a=struct.unpack('<7i',h.read(28));shape=a[1:5];n=math.prod(shape)*8;h.seek(-4,2)
  assert a[0]==a[5]==16 and a[6]==n and struct.unpack('<i',h.read(4))[0]==n and Path(f).stat().st_size==n+32
 assert shape[3]==4
 return np.memmap(f,dtype='<f8',mode='r',offset=28,shape=(math.prod(shape[:3]),4),order='F'),shape[:3]
def index(cell,u,idx,g=G):
 mat=CELLS[cell][0];native=(g[0]*3,g[1],g[2]) if cell=='s31' else (g[0],g[1]*3,g[2])
 f=np.column_stack((idx%g[0]/g[0]+u[0],(idx//g[0])%g[1]/g[1]+u[1]))@np.linalg.inv(mat)
 xy=f*np.array(native[:2]);assert np.max(abs(xy-np.rint(xy)))<1e-9
 xy=np.rint(xy).astype(int)%np.array(native[:2]);z=idx//(g[0]*g[1])
 return xy[:,0]+native[0]*(xy[:,1]+native[1]*z)
def setup(kind):
 suffix='RAW_DENSITY.bin' if kind=='density' else 'FROZEN_HXC.bin'
 raw=P/'density_ansatz_harris_r01/roundtrip_check/raw_export/RAW_DENSITY.bin' if kind=='density' else P/'variational_field_export_r01/zero/KS_HARTREE_XC.bin'
 old=P/'variational_field_export_r01';of='DENSITY.bin' if kind=='density' else 'KS_HARTREE_XC.bin'
 spatial=P/'spatial_completion_continue_20260918_r01'
 zeros={'s31':raw,**{c:spatial/(c+'_zero')/'fields'/suffix for c in ['s13','sd']}}
 zeros['sd']=REPAIR/'sd_zero/physical_samples'/suffix
 modes=[('s31','ac',old/'plus'/of,old/'minus'/of),('s31','op',None,P/'qx_optical_training_r01/minus/fields'/suffix),('s31','go',None,P/'gamma_optical_training_r01/minus/fields'/suffix)]
 modes += [(c,m,spatial/(c+'_'+m+'_plus')/'fields'/suffix,spatial/(c+'_'+m+'_minus')/'fields'/suffix) for c in ['s13','sd'] for m in ['ac','op']]
 modes=[(c,m,REPAIR/'recovery_20260921_r01'/('sd_'+m+'_plus')/'physical_samples'/suffix,REPAIR/'recovery_20260921_r01'/('sd_'+m+'_minus')/'physical_samples'/suffix) if c=='sd' else (c,m,plus,minus) for c,m,plus,minus in modes]
 rows=[];design=[]
 for c,m,plus,minus in modes:
  q=np.zeros(2) if m=='go' else np.array(CELLS[c][2]);sign=np.array([1 if b==0 or m=='ac' else -1 for _,_,b in CHANNELS])
  for u in CELLS[c][1]:
   phase=math.pi/4+2*math.pi*(np.array([t[:2] for t in CHANNELS])+u)@q
   for comp,trig in [(1,np.cos),(2,np.sin)]:rows.append((c,m,plus,minus,u,comp));design.append(sign*trig(phase))
 x=np.array(design);proj=np.eye(14)-np.ones((14,14))/14
 assert np.linalg.matrix_rank(x@proj,tol=1e-10)==13
 inv=proj@np.linalg.pinv(x@proj,rcond=1e-10)
 k=np.arange(14)**2/13;bas=np.full(14,k.sum()/14)
 assert np.max(abs(bas+inv@(x@k-x@bas)-k))<1e-10
 return zeros,rows,x,inv

def build(kind):
 dest=R/(kind+'_kernel.npy');assert not dest.exists(),'Preserve existing candidate'
 zeros,rows,x,inv=setup(kind);paths=set(zeros.values())|{p for row in rows for p in row[2:4] if p is not None};data={p:read(p) for p in paths}
 for c,p in zeros.items():assert data[p][1]==((3*PARAM['nr1'],PARAM['nr2'],PARAM['nr3']) if c=='s31' else (PARAM['nr1'],3*PARAM['nr2'],PARAM['nr3']))
 for c,_,plus,minus,_,_ in rows:
  for path in (plus,minus):
   if path is not None:assert data[path][1]==((3*PARAM['nr1'],PARAM['nr2'],PARAM['nr3']) if c=='s31' else resample.COARSE)
 k=np.lib.format.open_memmap(dest,mode='w+',dtype='<f8',shape=(N,14));fm=np.lib.format.open_memmap(R/(kind+'_FM.npy'),mode='w+',dtype='<f8',shape=(N,4))
 rss=np.zeros(len(rows));yn=np.zeros(len(rows));fmerr={c:np.zeros(2) for c in CELLS};sumerr=0;h=math.sin(math.radians(PARAM['angle_degrees']))
 for start in range(0,N,PARAM['native_block']):
  ids=np.arange(start,min(start+PARAM['native_block'],N));ix={(c,u):index(c,u,ids) for c in CELLS for u in CELLS[c][1]}
  ref=np.mean([data[zeros['s31']][0][ix['s31',u]] for u in CELLS['s31'][1]],axis=0);fm[ids]=ref
  for c in CELLS:
   z=np.mean([data[zeros[c]][0][ix[c,u]] for u in CELLS[c][1]],axis=0);fmerr[c]+=np.array([np.sum((z-ref)**2),np.sum(ref**2)])
  y=np.empty((len(ids),len(rows)))
  for j,(c,m,plus,minus,u,comp) in enumerate(rows):
   im=ix[c,u];a=data[minus][0][im,comp]
   y[:,j]=(data[plus][0][im,comp]-a)/(2*h) if plus else (data[zeros[c]][0][im,comp]-a)/h
  base=np.repeat(ref[:,3,None]/14,14,axis=1);out=base+(y-base@x.T)@inv.T;k[ids]=out
  residual=out@x.T-y;rss+=np.sum(residual**2,axis=0);yn+=np.sum(y*y,axis=0);sumerr=max(sumerr,float(np.max(abs(out.sum(axis=1)-ref[:,3]))))
 k.flush();fm.flush();assert sumerr<1e-10
 back=np.load(dest,mmap_mode='r');assert np.array_equal(back[::7919],k[::7919])
 result={'kind':kind,'grid':G,'channels':CHANNELS,'rank_with_FM_sum':14,'training_rows':len(rows),'FM_sum_imposed':True,'FM_sum_maxerror':sumerr,'relative_training_residual':np.sqrt(rss/np.maximum(yn,1e-300)).tolist(),'row_labels':[(c,m,u,comp) for c,m,_,_,u,comp in rows],'FM_cross_cell_relative_difference':{c:math.sqrt(a/b) for c,(a,b) in fmerr.items()},'source_files':[str(p) for p in sorted(paths)],'limits':['Finite chosen14channel support; full rank is not physical sufficiency','Qx optical and Gamma optical one-sided; all other endpoints paired','Frozen scalar density and nonlinear HXC/Harris route required for energy','Magnetic V_func_tmp is only FM tangent, not independently valid band-only energy model','No independent full-spatial torque or Wannier acceptance yet']}
 result.update(SD_training='Paired lambda12 fineSCF endpoints: same lambda*dV as original lambda4 training; physicalG projection before coarse sampling.',native_grids={c:native_shape(c) for c in CELLS},
               FM_cross_cell_SD_scope={'comparable_constraint_strength':False,'physical_validation':False,'reason':'SD zero is a lambda4 fine-grid warm seed with one-third of the s31 effective lambda; its diagnostic difference is retained, not a matched-constraint FM test.','used_in_fitted_parameters':False,'fitted_FM_source':str(zeros['s31'])})
 (R/(kind+'_RESULT.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_files'},indent=2),flush=True)
 if kind=='potential':serialize()

def serialize():
 k=np.load(R/'potential_kernel.npy',mmap_mode='r');zeros=np.zeros(N,dtype='<f8');size=N*3*14*3*8
 with (R/'V_func_tmp').open('xb') as f:
  f.write(struct.pack('<6i',0,0,0,0,0,0));f.write(struct.pack('<i',size))
  for spincomp in range(3):
   for channel in range(14):
    for fieldcomp in range(3):f.write(np.asarray(k[:,channel] if spincomp==fieldcomp else zeros,dtype='<f8').tobytes())
  f.write(struct.pack('<3i',size,0,0))
 # Actual record readback, including every block (diagonal and off-diagonal).
 a=np.memmap(R/'V_func_tmp',dtype='<f8',mode='r',offset=28,shape=(N,3,14,3),order='F')
 for c in range(3):
  for d in range(3):assert np.array_equal(a[::7919,c,:,d],k[::7919] if c==d else np.zeros_like(k[::7919]))
 ids=np.arange(N);vec=np.column_stack((ids%G[0],ids//G[0]%G[1],ids//(G[0]*G[1]))).astype('<i4')
 with (R/'R_vec').open('xb') as f:f.write(struct.pack('<4i',4,N,4,vec.nbytes));f.write(vec.tobytes(order='F'));f.write(struct.pack('<i',vec.nbytes))
 (R/'INTERFACE.json').write_text(json.dumps({'n_function':42,'n_dep':14,'spin_normalization':'unit target spins','channel_order':CHANNELS,'V_func_tmp':'3empty records;float64(N,3,14,3) Fortran order;empty record;Ry','R_vec':'record N;record int32(N,3) Fortran order;zero based81x81x225 primitive grid','warning':'New isolated tangent artifact; not drop-in accepted legacy Wannier/band-energy model','density_interface':'model.py evaluate CELL spins.json output.bin; SD output is243x243x225. Then recompute native nonlinear HXC and Harris energy/torque.','coarse_interface':'evaluate_coarse retains serialized sampling layout; index() is coarse. lift_sd must also act on assembled magnetic potential plus fixedFMoffset and on all analytic derivatives.','SD_projection':'physical-G projection/lift; not lossless raw-HXC interpolation','SD_G_source':str(REPAIR/'fixed_fm/SEED_CHARGE.dat'),'native_grids':{c:native_shape(c) for c in CELLS}},indent=2)+'\n')

def density_coarse(cell,spins):
 e=np.asarray(spins,float);assert e.shape==(6,3) and np.isfinite(e).all() and np.allclose(np.linalg.norm(e,axis=1),1,atol=1e-10,rtol=0)
 assert np.min(e[:,2])>=math.cos(math.radians(3))-1e-12
 mat,tiles,q=CELLS[cell];inv=np.linalg.inv(mat);k=np.load(R/'density_kernel.npy',mmap_mode='r');fm=np.load(R/'density_FM.npy',mmap_mode='r');out=np.empty((N*3,4),order='F')
 for u in tiles:
  sites=[]
  for tx,ty,b in CHANNELS:
   matches=[j for j,v in enumerate(tiles) if np.allclose((np.array([tx,ty])+u-v)@inv,np.rint((np.array([tx,ty])+u-v)@inv),atol=1e-10)]
   assert len(matches)==1;sites.append(2*matches[0]+b)
  for start in range(0,N,PARAM['native_block']):
   ids=np.arange(start,min(start+PARAM['native_block'],N));a=np.array(fm[ids]);a[:,1:]=k[ids]@e[sites];a[:,1:3]+=fm[ids,1:3];out[index(cell,u,ids)]=a
 return out

def evaluate_coarse(cell,spins,dest):
 write_field(dest,density_coarse(cell,spins),(3*PARAM['nr1'],PARAM['nr2'],PARAM['nr3']) if cell=='s31' else resample.COARSE)

def evaluate(cell,spins,dest):
 assert not Path(dest).exists()
 if cell=='sd':assert not Path(str(dest)+'.lift.json').exists()
 out=density_coarse(cell,spins);metrics=[]
 if cell=='sd':out=lift_sd(out,metrics=metrics)
 write_field(dest,out,native_shape(cell))
 if cell=='sd':
  with Path(str(dest)+'.lift.json').open('x') as f:json.dump({'operation':'SD physical-G lift/projection','G_source':str(REPAIR/'fixed_fm/SEED_CHARGE.dat'),'components':metrics},f,indent=2);f.write('\n')

def check():
 for c in CELLS:
  ids=np.arange(3*3*5);mapped=np.concatenate([index(c,u,ids,(3,3,5)) for u in CELLS[c][1]])
  assert np.array_equal(np.sort(mapped),np.arange(3*len(ids)))
  # An analytic periodic scalar in primitive coordinates must agree after mapping.
  for u in CELLS[c][1]:
   native=(9,3,5) if c=='s31' else (3,9,5);ix=index(c,u,ids,(3,3,5));f=np.column_stack((ix%native[0]/native[0],ix//native[0]%native[1]/native[1]))@CELLS[c][0]
   assert np.allclose(np.cos(2*np.pi*f[:,0])+np.sin(2*np.pi*f[:,1]),np.cos(2*np.pi*(ids%3)/3)+np.sin(2*np.pi*(ids//3%3)/3),atol=1e-12)
 zeros,rows,x,inv=setup('density')
 assert zeros['sd']==REPAIR/'sd_zero/physical_samples/RAW_DENSITY.bin'
 assert all('/physical_samples/' in str(p) for c,_,plus,minus,_,_ in rows if c=='sd' for p in (plus,minus))
 # All six-site spin fields in one training cell are independent of the other
 # two cells' ac/op labels. This protects reuse of completed s31/s13 torque;
 # check the actual solver, not only a rank count or one response pattern.
 go=[j for j,r in enumerate(rows) if r[:2]==('s31','go')]
 base=np.ones(14)/14;fm_weights=base-inv@(x@base);gamma=None;cross_error=0.
 for c,(matrix,tiles,_) in CELLS.items():
  fold=np.zeros((len(tiles),6,14));inverse=np.linalg.inv(matrix)
  for tile,u in enumerate(tiles):
   for ch,(tx,ty,b) in enumerate(CHANNELS):
    hits=[j for j,t in enumerate(tiles) if np.allclose((np.array([tx,ty])+u-t)@inverse,np.rint((np.array([tx,ty])+u-t)@inverse),atol=1e-10)]
    assert len(hits)==1;fold[tile,2*hits[0]+b,ch]=1
  response=fold.reshape(-1,14)@inv
  other=[j for j,r in enumerate(rows) if r[0]!=c and r[1]!='go']
  cross_error=max(cross_error,float(np.max(abs(response[:,other]))))
  assert np.max(abs(response[:,other]))<1e-14
  # Shared FM is uniformly1/6 per site. Shared Gamma response averages source
  # translations, hence is primitive periodic; it does not mix reciprocal G.
  assert np.max(abs(fold.reshape(-1,14)@fm_weights-1/6))<1e-14
  shared=response[:,go].reshape(3,6,3,2)
  assert np.max(abs(shared-shared[0:1,:,0:1,:]))<1e-14
  if gamma is not None:assert np.max(abs(shared-gamma))<1e-14
  gamma=shared
 # Test the ACTUAL shared four-field lift, with physical frequency outside the
 # coarse principal FFT box; a coarse roundtrip alone cannot reveal this error.
 coarse,fine=(81,243,1),(243,243,1);m=np.array([50,-60,0]);mill=np.array([[0,0,0],m,-m])
 x=np.arange(243)[:,None,None]/243;y=np.arange(243)[None,:,None]/243
 oracle=np.cos(2*np.pi*(m[0]*x+m[1]*y));weights=np.array([.7,2.,-.3,1.2])
 field=(oracle[::3,...,None]*weights).reshape(-1,4,order='F')
 lifted=lift_sd(field,mill,coarse,fine)
 assert np.max(abs(lifted-(oracle[...,None]*weights).reshape(-1,4,order='F')))<2e-12
 wrong=(mill+np.array(coarse)//2)%coarse-np.array(coarse)//2
 coeff=resample.spectrum(oracle[::3])[resample.indices(mill,coarse)]
 naive=resample.from_coefficients(coeff,wrong,fine)
 assert np.max(abs(naive[::3]-oracle[::3]))<2e-12 and np.max(abs(naive-oracle))>1.
 print('PASS: coarse mappings,14channel design, SD lift, cross-cell ac/op decoupling max=',cross_error,'; primitive-periodic FM/Gamma; physical response NOT_RUN')
if __name__=='__main__':
 if sys.argv[1:]==['self-check']:check()
 elif sys.argv[1:]==['build']:check();build('density');build('potential')
 elif len(sys.argv)==5 and sys.argv[1]=='evaluate':evaluate(sys.argv[2],json.loads(Path(sys.argv[3]).read_text()),sys.argv[4])
 elif len(sys.argv)==5 and sys.argv[1]=='evaluate-coarse':evaluate_coarse(sys.argv[2],json.loads(Path(sys.argv[3]).read_text()),sys.argv[4])
 else:raise SystemExit('model.py self-check | build | evaluate[-coarse] CELL spins.json output.bin')
