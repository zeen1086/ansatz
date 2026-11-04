"""14-channel FM tangent candidate. Native grids; no interpolation or physical release."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from parameters import load
PARAM = load()
import sys,json,math,struct
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent
T=np.array([[0,0],[0,-1],[1,0],[1,1],[-1,0],[0,1],[-1,-1]])
CHANNELS=[(int(t[0]),int(t[1]),b) for t in T for b in range(2)]
CELLS={'s31':(np.array([[3,0],[0,1]]),[(0,0),(1,0),(2,0)],(1/3,0)), 's13':(np.array([[1,0],[0,3]]),[(0,0),(0,1),(0,2)],(0,1/3)), 'sd':(np.array([[1,-1],[0,3]]),[(0,0),(0,1),(0,-1)],(1/3,1/3))}
G=(PARAM['nr1'],PARAM['nr2'],PARAM['nr3']);N=math.prod(G)
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
 modes=[('s31','ac',old/'plus'/of,old/'minus'/of),('s31','op',None,P/'qx_optical_training_r01/minus/fields'/suffix),('s31','go',None,P/'gamma_optical_training_r01/minus/fields'/suffix)]
 modes += [(c,m,spatial/(c+'_'+m+'_plus')/'fields'/suffix,spatial/(c+'_'+m+'_minus')/'fields'/suffix) for c in ['s13','sd'] for m in ['ac','op']]
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
 (R/'INTERFACE.json').write_text(json.dumps({'n_function':42,'n_dep':14,'spin_normalization':'unit target spins','channel_order':CHANNELS,'V_func_tmp':'3empty records;float64(N,3,14,3) Fortran order;empty record;Ry','R_vec':'record N;record int32(N,3) Fortran order;zero based81x81x225 primitive grid','warning':'New isolated tangent artifact; not drop-in accepted legacy Wannier/band-energy model','density_interface':'model.py evaluate CELL spins.json output.bin; then native variationalHXC and Harris energy/torque'},indent=2)+'\n')

def evaluate(cell,spins,dest):
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
 shape=(3*PARAM['nr1'],PARAM['nr2'],PARAM['nr3'],4) if cell=='s31' else (PARAM['nr1'],3*PARAM['nr2'],PARAM['nr3'],4)
 with Path(dest).open('xb') as f:f.write(struct.pack('<7i',16,*shape,16,out.nbytes));f.write(out.tobytes(order='F'));f.write(struct.pack('<i',out.nbytes))
 assert np.array_equal(read(dest)[0],out)

def check():
 for c in CELLS:
  ids=np.arange(3*3*5);mapped=np.concatenate([index(c,u,ids,(3,3,5)) for u in CELLS[c][1]])
  assert np.array_equal(np.sort(mapped),np.arange(3*len(ids)))
  # An analytic periodic scalar in primitive coordinates must agree after mapping.
  for u in CELLS[c][1]:
   native=(9,3,5) if c=='s31' else (3,9,5);ix=index(c,u,ids,(3,3,5));f=np.column_stack((ix%native[0]/native[0],ix//native[0]%native[1]/native[1]))@CELLS[c][0]
   assert np.allclose(np.cos(2*np.pi*f[:,0])+np.sin(2*np.pi*f[:,1]),np.cos(2*np.pi*(ids%3)/3)+np.sin(2*np.pi*(ids//3%3)/3),atol=1e-12)
 setup('density');print('PASS: native lattice bijections, analytic field mapping and14channel synthetic recovery')
if __name__=='__main__':
 if sys.argv[1:]==['self-check']:check()
 elif sys.argv[1:]==['build']:check();build('density');build('potential')
 elif len(sys.argv)==5 and sys.argv[1]=='evaluate':evaluate(sys.argv[2],json.loads(Path(sys.argv[3]).read_text()),sys.argv[4])
 else:raise SystemExit('model.py self-check | build | evaluate CELL spins.json output.bin')
