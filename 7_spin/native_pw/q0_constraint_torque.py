#!/usr/bin/env python3
"""Finite-penalty torque diagnostic from existing high-precision XML moments."""
import json,math,re,sys
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from parameters import load
PARAM = load()
import numpy as np
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent

def torque(m,target,axis,lam):
    # At variational stationarity: grad_m E_DFT=-2 lambda (m-target).
    # tau_axis=-grad_m E_DFT dot (axis cross m).
    return 2*lam*np.sum((m-target)*np.cross(axis,m),axis=1)

def self_check():
    m=np.array([[0.,0.,3.]]);target=np.array([[0.,0.,3.]])
    assert torque(m,target,np.array([0.,1.,0.]),4)[0]==0
    eps=1e-4;target=np.array([[3*math.sin(eps),0,3*math.cos(eps)]])
    assert abs(torque(m,target,np.array([0.,1.,0.]),4)[0]+72*math.sin(eps))<1e-14
    print('Torque sign and aligned-zero checks PASS')

def run():
    rows={}
    for name,rel in {'zero':'tight_q0/zero','plus_0p5':'tight_q0/plus','minus_0p5':'tight_q0/minus','plus_1':'angle1_q0/plus','minus_1':'angle1_q0/minus'}.items():
        p=ROOT/rel;assert json.loads((p/'SCF_RESULT.json').read_text())['verdict']=='TIGHT_SCF_ACCEPTED'
        text=(p/'scf.in').read_text()
        def value(key):
            matches=re.findall(r'(?im)^\s*'+re.escape(key)+r'\s*=\s*([-+\d.eEdD]+)',text)
            assert len(matches)==1
            return float(matches[0].replace('d','e').replace('D','e'))
        angle=math.radians(value('angle1(1)'));phase=math.radians(value('angle2(1)'))
        magnitude=value('starting_magnetization(1)');lam=value('lambda')
        direction=np.array([math.sin(angle)*math.cos(phase),math.sin(angle)*math.sin(phase),math.cos(angle)])
        xml=ET.parse(p/'tmp/CrI3.save/data-file-schema.xml')
        mags=[np.fromstring(x.text,sep=' ') for x in xml.iter() if x.tag.split('}')[-1]=='SiteMagnetization']
        assert len(mags)==8
        m=np.array(mags[:2]);target=np.tile(magnitude*direction,(2,1));axis=np.array([0.,1.,0.])
        tau=torque(m,target,axis,lam)
        rows[name]=dict(torque_about_global_y_Ry=tau.tolist(),sum_torque_Ry=float(sum(tau)),
            penalty_reconstructed_Ry=float(lam*np.sum((m-target)**2)),moments=m.tolist())
    result=dict(classification='DIAGNOSTIC',verdict='Q0_CONSTRAINT_TORQUE_NOISE_OBSERVED',rows=rows,
       interpretation='Stationary-state constraint-balance torque estimate. Nonzero common-rotation torque diagnoses numerical/source-state limitations; not validated finite-q reference or formal Phi2/Phi4.')
    with (ROOT/'Q0_CONSTRAINT_TORQUE.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    if sys.argv[1:]==['self-check']:self_check()
    elif not sys.argv[1:]:run()
    else:raise ValueError(sys.argv[1:])
