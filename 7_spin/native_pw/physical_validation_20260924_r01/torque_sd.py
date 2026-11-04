"""Full14 SD blind torque, reusing the completed layer1 three-term native-PW runner."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from parameters import load
PARAM = load()
import importlib.util,json,math,os,re,shutil,subprocess,sys,types
import xml.etree.ElementTree as ET
import numpy as np
import fit14 as f
import run as campaign
from projection_consistency import s
V=Path(__file__).resolve().parent;P=V.parent;R=V/'torque_sd_20260925'
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
l=module('layer1_torque',P/'coordinate_layer1_issp_20260923_r01/run.py')
a=module('torque_analysis',P/'three_cell_torque_20260920_r01/analyze.py')
w=module('native_readout',P/'three_cell_torque_20260921_r02/run.py')
q=module('constraint_torque',P/'q0_constraint_torque.py')
SEED=R/'seed/tmp/CrI3.save'
WFC=P/'sd_fft_repair_20260921_r01/sd_zero/tmp/CrI3.save'

def prepare():
 assert os.environ.get('PBS_JOBID')
 R.mkdir(exist_ok=True);out=R/'fits';out.mkdir();SEED.mkdir(parents=True)
 src=V/'sd/blind0_plus/tmp/CrI3.save'
 for path in src.iterdir():
  if path.name in ('charge-density.dat','data-file-schema.xml') or path.suffix.lower()=='.upf':shutil.copy2(path,SEED/path.name)
 mill,_=s.read_millers(src/'charge-density.dat');s.validate_millers(mill,s.COARSE,s.FINE)
 # Folding and physical-G lifting are fixed linear maps; apply them to the kernels,
 # FM offsets and every later derivative through the same native-grid arrays.
 for kind in ('density','potential'):
  k=np.load(V/'fit14'/(kind+'_kernel.npy'),mmap_mode='r');fm=np.load(V/'fit14'/(kind+'_FM.npy'),mmap_mode='r')
  coarse=np.zeros((3*f.g.N,10))
  for start in range(0,f.g.N,PARAM['native_block']):
   ids=np.arange(start,min(start+PARAM['native_block'],f.g.N))
   for tile in f.g.CELLS['sd'][1]:
    ix=f.g.index('sd',tile,ids);sites=np.array(f.c.site_map('sd',tile))
    for site in range(6):coarse[ix,site]=k[ids][:,sites==site].sum(axis=1)
    coarse[ix,6:]=fm[ids]
  assert np.max(abs(coarse[:,:6].sum(axis=1)-coarse[:,9]))<1e-10
  kernel=np.lib.format.open_memmap(out/(kind+'_actual.npy'),mode='w+',dtype='<f8',shape=(math.prod(s.FINE),6))
  reference=np.lib.format.open_memmap(out/(kind+'_FM.npy'),mode='w+',dtype='<f8',shape=(math.prod(s.FINE),4))
  for col in range(10):
   freq=s.spectrum(coarse[:,col].reshape(s.COARSE,order='F'));coeff=freq[s.indices(mill,s.COARSE)]
   value=s.from_coefficients(coeff,mill,s.FINE)
   (kernel if col<6 else reference)[:,col if col<6 else col-6]=value.ravel(order='F')
   print('LIFTED',kind,col,flush=True)
  kernel.flush();reference.flush()
  assert np.max(abs(kernel.sum(axis=1)-reference[:,3]))<1e-10
  # Exact linear-map oracle: finite rotations through the native folded arrays.
  e=np.tile([0.,0.,1.],(6,1));e[0]=l.rotate(e[0],np.array(a.AXIS),.01)
  sample=np.arange(0,len(kernel),7919)
  h=1e-5;ep=e.copy();em=e.copy();ep[2]=l.rotate(e[2],np.array(a.AXIS),h);em[2]=l.rotate(e[2],np.array(a.AXIS),-h)
  assert np.max(abs((kernel[sample]@ep-kernel[sample]@em)/(2*h)-kernel[sample,2,None]*np.cross(a.AXIS,e[2])*math.sin(h)/h))<1e-9
  del coarse,kernel,reference
 ref={'rows':{}};axis=np.tile(a.AXIS,(6,1))
 for side in ('plus','minus'):
  d=V/'sd'/('blind0_'+side);state=json.loads((d/'RESULT.json').read_text());x=ET.parse(d/'tmp/CrI3.save/data-file-schema.xml')
  m=np.array(state['moments_muB']);e,details=f.c.spins({'status':'PRESENT','xml':ET.tostring(x.getroot(),encoding='unicode'),'xml_path':str(d),'xml_sha256':'not_needed_for_this_diagnostic','cell':'sd'})
  target=3*np.array(details['target_spins']);tau=q.torque(m,target,axis,12*state['dv'])
  ref['rows'][side]={'moments':m.tolist(),'site_torque_Ry':tau.tolist(),'scf_error_Ry':state['scf_error_Ry'],'achieved_spins':e.tolist()}
 l.save_json(R/'REFERENCE.json',ref)
 l.save_json(out/'RESULT.json',{'source':str(V/'fit14'),'method':'Exact SD fold plus existing physical-G lift; no refit','rank_source':14})
 l.save_json(R/'CONTRACT.json',{'scope':'Blind SD site0 plus/minus; actual directions; J1+J3 and J2 conditional readout','criteria':{'J1_plus_J3_relative':.02,'J2_relative':.02,'offsite_response_relative_L2':.02},'h_degrees':PARAM['angle_degrees'],'new_SCF':False,'band_diago_thr_init':1e-10,'source_wavefunctions':'Independent copy of existing SD FM WFC, never modified in place','torque_definition':'-int(rho_out dV)+int(rho_in dV)+int((V-Vnative) drho)'})
 print('PREPARE_COMPLETE',flush=True)

def bands_input(save):
 text=(V/'sd/blind0_plus/scf.in').read_text()
 values={'calculation':"'bands'",'restart_mode':"'from_scratch'",'outdir':repr(str(save.parent)),'pseudo_dir':repr(str(SEED)),'disk_io':"'low'",'wf_collect':'.true.','startingpot':"'file'",'startingwfc':"'file'",'constrained_magnetization':"'none'",'diago_thr_init':str(PARAM['sd_diago_thr']),'max_seconds':str(PARAM['native_max_seconds'])}
 for key,value in values.items():text=campaign.replace(text,key,value)
 return text

def execute(binary,directory,text,logname,marker,ranks):
 assert os.environ.get('PBS_JOBID')
 if binary.name=='pw_frozen_wfc.x':
  save=directory/'tmp/CrI3.save';files=sorted(WFC.glob('wfc*.dat'));assert len(files)==PARAM['sd_kpoints']
  assert shutil.disk_usage(directory).free>sum(x.stat().st_size for x in files)+10_000_000_000
  for src in files:
   dst=save/src.name;assert not dst.exists();before=src.stat();shutil.copy2(src,dst)
   assert dst.stat().st_ino!=before.st_ino and dst.stat().st_size==before.st_size and src.stat().st_mtime_ns==before.st_mtime_ns
  print('WFC_COPY_COMPLETE',flush=True)
 cmd=['mpirun','-np',str(ranks),str(binary)]
 if ranks>1:cmd+=['-npool','1','-ndiag','1']
 with (directory/logname).open('x') as out:subprocess.run(cmd,input=text,text=True,cwd=directory,stdout=out,stderr=subprocess.STDOUT,check=True)
 log=(directory/logname).read_text();assert marker in log and 'eigenvalues not converged' not in log

def endpoint(side):
 l.R=R;l.SEED=SEED;l.SHAPE=s.FINE;l.g=types.SimpleNamespace(read=f.g.read,N=math.prod(s.FINE)//3)
 l.execute=execute;l.bands_input=bands_input;l.band_data=lambda save:w.band_data(save/'data-file-schema.xml',PARAM['sd_kpoints'])
 original_save=l.save_json
 def save_metadata(path,value):
  if path.name=='TORQUES.json':
   value.update(cell='sd',grid=list(s.FINE),startingwfc='file; copied existing SD FM wavefunctions',scope='Full14 actual-coordinate blind SD candidate; same physical-G field and derivative representation',source_candidate=str(V/'fit14'))
  original_save(path,value)
 l.save_json=save_metadata
 l.endpoint('actual',side)
 print('SD_TORQUE_COMPLETE',side,flush=True)

def analyze():
 ref=json.loads((R/'REFERENCE.json').read_text());h=math.radians(PARAM['angle_degrees']);matrix,sites,_=a.geometry('sd');neighbors=a.counts(matrix,sites)[:,:,:3]
 responses={};design=[]
 for side in ('plus','minus'):
  e=np.array(ref['rows'][side]['achieved_spins']);th=np.arctan2((e[:,0]+e[:,1])/math.sqrt(2),e[:,2])
  design.append(-np.einsum('ijs,ij->is',neighbors,np.sin(th[:,None]-th[None,:]))/h)
 design=(design[0]-design[1])[1:]/2
 candidate=[];dft=[]
 for side in ('plus','minus'):
  row=json.loads((R/'endpoints/actual'/side/'TORQUES.json').read_text())
  assert np.allclose(row['spins'],ref['rows'][side]['achieved_spins'],rtol=0,atol=1e-14)
  candidate.append(row['site_torque_Ry_per_rad']);dft.append(ref['rows'][side]['site_torque_Ry'])
 for name,values in [('candidate',candidate),('DFT',dft)]:
  phi=(np.array(values[0])-values[1])/(2*h);fit=a.fit(design,phi[1:]);assert fit['rank']==2 and fit['identified_combinations_meV']
  responses[name]={'Phi_tau_Ry':phi.tolist(),'fit':fit}
 cp=responses['candidate'];df=responses['DFT'];error=float(np.linalg.norm(np.array(cp['Phi_tau_Ry'])[1:]-np.array(df['Phi_tau_Ry'])[1:])/np.linalg.norm(np.array(df['Phi_tau_Ry'])[1:]))
 rel={k:abs(cp['fit']['identified_combinations_meV'][k]/v-1) for k,v in df['fit']['identified_combinations_meV'].items()}
 l.save_json(R/'RESULT.json',{'preliminary':True,'responses':responses,'relative_J_combination_errors':rel,'offsite_response_relative_L2':error,'diagnostic_pass':max([error,*rel.values()])<=.02,'limits':['One achieved-spin path, finite constraint and angle; no full Hessian or physical uncertainty','SD identifies J1+J3 and J2 only; no standalone J1 or J3','Native fixed-potential band calculation, not additional SCF']})
 print('SD_TORQUE_ANALYSIS_COMPLETE',flush=True)

if __name__=='__main__':
 cmd=sys.argv[1]
 if cmd=='prepare':prepare()
 elif cmd=='endpoint':endpoint(sys.argv[2])
 elif cmd=='analyze':analyze()
 elif cmd=='check':
  q.self_check();matrix,sites,_=a.geometry('sd');n=a.counts(matrix,sites)[:,:,:3];assert np.array_equal(n,n.transpose(1,0,2));assert np.linalg.matrix_rank(n[1:,0])==2
  e=np.array([.01,0.,math.sqrt(1-.0001)]);axis=np.array(a.AXIS);h=1e-5
  assert np.max(abs((l.rotate(e,axis,h)-l.rotate(e,axis,-h))/(2*h)-np.cross(axis,e)))<1e-9
  text=bands_input(Path('/tmp/sd-check/tmp/CrI3.save'))
  assert "calculation='bands'" in text and "startingwfc='file'" in text and "constrained_magnetization='none'" in text
  assert text.split('CELL_PARAMETERS')[1]==(V/'sd/blind0_plus/scf.in').read_text().split('CELL_PARAMETERS')[1]
  print('PASS: SD geometry rank2, torque sign, local spin derivative and fixed-band input')
 else:raise ValueError(cmd)
