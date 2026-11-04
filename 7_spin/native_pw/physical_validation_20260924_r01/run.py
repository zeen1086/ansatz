"""SD constrained-SCF and matched-vector 2x2 calculations."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from parameters import load
PARAM = load()
import importlib.util
import json, math, os, re, shutil, subprocess, sys
import xml.etree.ElementTree as ET
import numpy as np
R=Path(__file__).resolve().parent
P=R.parent
PW=P/'qx_variational_plus_ethr_floor_r01/pw_variational_gga.x'
PP=P/'density_ansatz_harris_r01/roundtrip_check/pp_raw.x'
SD=P/'sd_fft_repair_20260921_r01/recovery_20260921_r01'
PSEUDO=P/'tangent_consistent_energy_20260918_r01/plus/state/tmp/CrI3.save'

def write(path,value):
    with path.open('x') as f: json.dump(value,f,indent=2,allow_nan=False); f.write('\n')

def replace(text,key,value):
    text,n=re.subn(r'(?m)^(\s*'+re.escape(key)+r'\s*=)[^\n,]+',lambda m:m[1]+value,text)
    assert n==1,(key,n)
    return text

def execute(binary,directory,inp,log,ranks,pools=1):
    assert os.environ.get('SLURM_JOB_ID')
    cmd=['srun','--exclusive','-N','1','-n',str(ranks),str(binary)]
    if ranks>1: cmd+=['-npool',str(pools),'-ndiag','1']
    with (directory/log).open('x') as f:
        code=subprocess.run(cmd,cwd=directory,input=inp,text=True,stdout=f,stderr=subprocess.STDOUT).returncode
    write(directory/(log+'.execution.json'),{'returncode':code,'command':cmd})
    return code,(directory/log).read_text()

def xml_data(save, preliminary_threshold=None):
    x=ET.parse(save/'data-file-schema.xml'); out=x.find('output')
    converged=out.findtext('.//convergence_achieved')=='true'
    if not converged:
        assert preliminary_threshold is not None
        assert int(out.findtext('.//n_scf_steps'))>0
        assert 0<2*float(out.findtext('.//scf_error'))<=preliminary_threshold
    error=2*float(out.findtext('.//scf_error'))
    assert math.isfinite(error) and error>=0
    sites=out.findall('magnetization/Site_Magnetizations/SiteMagnetization')
    assert sites, 'XML omits site moments; cannot infer achieved spin directions from an SCF error alone'
    m=np.array([np.fromstring(s.text,sep=' ') for s in sites if s.attrib['species'].startswith('Cr')])
    assert np.isfinite(m).all() and np.min(np.linalg.norm(m,axis=1))>0
    cell=np.array([np.fromstring(c.text,sep=' ') for c in out.find('atomic_structure/cell')])
    grid=tuple(int(out.find('basis_set/fft_grid').attrib['nr'+str(i)]) for i in (1,2,3))
    return x,{'xml_converged':converged,'preliminary':preliminary_threshold is not None,'acceptance_scf_threshold_Ry':preliminary_threshold,'scf_error_Ry':error,'moments_muB':m.tolist(),'grid':grid,'dv':abs(float(np.linalg.det(cell)))/math.prod(grid),'energy_Ry':2*float(out.findtext('total_energy/etot'))}

def seed(source,target):
    target.mkdir(parents=True)
    for name in ('data-file-schema.xml','charge-density.dat'):
        shutil.copy2(source/name,target/name)
    for f in PSEUDO.glob('*.upf'): shutil.copy2(f,target/f.name)

def sd(case, blind=False, retry=False):
    assert case in (('blind0_plus','blind0_minus','blind1_plus','blind1_minus') if blind else ('sd_ac_plus','sd_ac_minus','sd_op_plus','sd_op_minus'))
    out=R/('sd_retry' if retry else 'sd')/case; out.mkdir(parents=True)
    source=SD/('sd_ac_plus' if blind else case); save=out/'tmp/CrI3.save'
    seed(source/'tmp/CrI3.save',save)
    inp=(source/'scf.in').read_text()
    for k,v in {'outdir':repr(str(save.parent)),'pseudo_dir':repr(str(PSEUDO))}.items(): inp=replace(inp,k,v)
    is_new=blind or case=='sd_ac_minus'
    if blind:
        site=int(case[5]);sign=1 if case.endswith('plus') else -1
        for i in range(6):
            inp=replace(inp,f'angle1({i+1})',str(sign*PARAM['angle_degrees'] if i==site else 0.))
            inp=replace(inp,f'angle2({i+1})','45.0')
    if is_new:
        if not blind: assert ET.parse(save/'data-file-schema.xml').findtext('.//output//convergence_achieved')=='false'
        for k,v in {'startingwfc':"'atomic+random'",'startingpot':"'file'",'restart_mode':"'from_scratch'",'electron_maxstep':'24','max_seconds':str(PARAM['native_scf_seconds']),'diago_thr_init':str(PARAM['native_diago_thr']),'disk_io':"'low'",'wf_collect':'.true.'}.items(): inp=replace(inp,k,v)
        (out/'scf.in').write_text(inp)
        code,log=execute(PW,out,inp,'scf.out',16,1)
        assert 'Error in routine' not in log and 'convergence has been achieved' in log
    else:
        (out/'scf.in').write_text(inp)
    finish_sd(out,source,is_new)

def finish_sd(out,source,is_new,threshold=1e-10,preliminary=False):
    case=out.name; save=out/'tmp/CrI3.save'
    _,data=xml_data(save,threshold if preliminary else None)
    assert data['scf_error_Ry']<=threshold and tuple(data['grid'])==(3*PARAM['nr1'],3*PARAM['nr2'],PARAM['nr3'])
    fields=out/'fields'; fields.mkdir()
    (fields/'DENSITY_ANSATZ.bin').symlink_to('RAW_DENSITY.bin')
    code,log=execute(PP,fields,f"&inputpp\n outdir='{save.parent}'\n prefix='CrI3'\n/\n",'export.out',1)
    assert code==0 and 'CRI3_VARIATIONAL_FIELD_EXPORT_COMPLETE' in log and 'JOB DONE.' in log
    reload=re.search(r'DENSITY_RELOAD_MAXDIFF\s+([^\n]+)',log)
    assert reload and max(abs(float(v.replace('D','e'))) for v in reload[1].split())<1e-12
    spec=importlib.util.spec_from_file_location('resample',P/'sd_fft_repair_20260921_r01/resample.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    samples=out/'physical_samples';samples.mkdir()
    for name in ('RAW_DENSITY.bin','FROZEN_HXC.bin'):
        metrics=mod.process('restrict',fields/name,save/'charge-density.dat',samples/name,samples/(name+'.metrics.json'))
        assert max(c['in_sphere_coefficient_relative_l2'] for c in metrics['components'])<1e-11
    write(out/'RESULT.json',{'new_SCF':is_new,**data,'input_lambda':12,'source':str(source)})
    print('SD_COMPLETE',case,flush=True)

def path(case, retry=False):
    assert case in ('fm','neel','mid90')
    angle={'fm':0.,'neel':180.,'mid90':90.}[case]
    out=R/('path_retry' if retry else 'path')/case;out.mkdir(parents=True)
    source=P/'regularized_2x2_mapping_20260921_r01'/('neel' if case=='neel' else 'fm')/'tmp/CrI3.save'
    save=out/'tmp/CrI3.save';seed(source,save)
    inp=(R/'optical_template.in').read_text()
    settings={'outdir':repr(str(save.parent)),'pseudo_dir':repr(str(PSEUDO)),'angle1(1)':str(angle/2),'angle1(2)':str(-angle/2),'diago_thr_init':str(PARAM['native_diago_thr']),'max_seconds':str(PARAM['native_scf_seconds']),'disk_io':"'low'",'wf_collect':'.true.'}
    for k,v in settings.items(): inp=replace(inp,k,v)
    (out/'scf.in').write_text(inp)
    code,log=execute(PW,out,inp,'scf.out',32 if retry else 64,2 if retry else 4)
    assert 'Error in routine' not in log and 'convergence has been achieved' in log
    finish_path(out,log,angle)

def finish_path(out,log,angle,threshold=1e-9,preliminary=False):
    case=out.name;save=out/'tmp/CrI3.save'
    x,data=xml_data(save,threshold if preliminary else None)
    assert data['scf_error_Ry']<=threshold and tuple(data['grid'])==(162,162,225)
    assert int(x.findtext('.//output/band_structure/nks'))==36
    assert 'CRI3_CR_ONLY_ATOMIC_CANARY: zero-target atom types are unconstrained' in log
    m=np.array(data['moments_muB']); assert m.shape==(8,3)
    mag=3.0021777499999995
    h=math.radians(angle); angles=np.array([h/2]*4+[-h/2]*4)
    target=mag*np.column_stack((np.sin(angles)/math.sqrt(2),np.sin(angles)/math.sqrt(2),np.cos(angles)))
    lam=4*data['dv']; penalty=lam*float(np.sum((m-target)**2))
    printed=float(re.findall(r'constraint energy \(Ry\)\s*=\s*([-+\d.eEdD]+)',log)[-1].replace('D','e'))
    assert abs(printed-penalty/data['dv'])<=1e-8
    tau=2*lam*np.cross(target,m)@np.array([-1.,1.,0.])/math.sqrt(2)
    actual=np.arctan2((m[:,0]+m[:,1])/math.sqrt(2),m[:,2])
    achieved=float(actual[:4].mean()-actual[4:].mean())
    assert abs(achieved-h)<math.radians(5), 'Wrong achieved branch'
    from decimal import Decimal
    edft=2*Decimal(x.findtext('.//output/total_energy/etot'))
    write(out/'RESULT.json',{'protocol':'same finite vector lambda=4 and Cr target magnitude as h1p5/h3; iodine unconstrained','angle_target_degrees':angle,'angle_actual_degrees':math.degrees(achieved),'E_DFT_Ry':str(edft),'P_eff_Ry':penalty,'F_Ry':str(edft+Decimal(str(penalty))),'tau_target_Ry':float((tau[:4].sum()-tau[4:].sum())/2),'global_torque_Ry':float(tau.sum()),'Cr_magnitudes_muB':np.linalg.norm(m,axis=1).tolist(),**data})
    print('PATH_COMPLETE',case,flush=True)

def check():
    inp=(R/'optical_template.in').read_text()
    changed=replace(inp,'angle1(1)','45.0')
    assert changed.split('ATOMIC_SPECIES')[1]==inp.split('ATOMIC_SPECIES')[1]
    assert "lambda=4.0" in changed and '3.0021777499999995' in changed
    h=.04; step=1e-5; lam=.2; m=np.array([.01,.02,2.98]); axis=np.array([-1.,1.,0.])/math.sqrt(2)
    def target(a): return 3*np.array([math.sin(a)/math.sqrt(2),math.sin(a)/math.sqrt(2),math.cos(a)])
    derivative=(lam*np.sum((m-target(h+step))**2)-lam*np.sum((m-target(h-step))**2))/(2*step)
    torque=2*lam*np.dot(np.cross(target(h),m),axis)
    assert abs(derivative+torque)<1e-9
    print('PASS: real template preserves geometry and vector constraint; penalty derivative has correct torque sign')

if __name__=='__main__':
    if sys.argv[1]=='check': check()
    elif sys.argv[1]=='sd': sd(sys.argv[2])
    elif sys.argv[1]=='sd-retry': sd(sys.argv[2],retry=True)
    elif sys.argv[1]=='blind': sd(sys.argv[2],blind=True)
    elif sys.argv[1]=='path': path(sys.argv[2])
    elif sys.argv[1]=='path-retry': path(sys.argv[2],retry=True)
    else: raise ValueError(sys.argv[1])
