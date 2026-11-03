"""Compare candidate and held-out fields in the same pre-existing physical-G space."""
import importlib.util,json,math
from pathlib import Path
import numpy as np
import fit14 as f
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('resample',R.parent/'sd_fft_repair_20260921_r01/resample.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

def main():
 result={'physical_release':False,'diagnostic_only':True,'contract':'Use the held-out charge-density physical-G set already used by finish_sd; no fitted mask or scale','fields':{}}
 previous=json.loads((R/'BLIND_FIELD_RESULT.json').read_text())
 for kind in ('potential','density'):
  k=np.load(R/'fit14'/(kind+'_kernel.npy'),mmap_mode='r');suffix='FROZEN_HXC.bin' if kind=='potential' else 'RAW_DENSITY.bin'
  total=np.zeros(5);rows=[]
  for site in (0,1):
   ds=[R/'sd'/('blind'+str(site)+'_'+side) for side in ('plus','minus')]
   mill,_=s.read_millers(ds[0]/'tmp/CrI3.save/charge-density.dat');s.validate_millers(mill,s.COARSE,s.FINE)
   mask=np.zeros(s.COARSE,dtype=bool);mask[s.indices(mill,s.COARSE)]=True
   a,b=[f.g.read(d/'physical_samples'/suffix)[0] for d in ds]
   de=f.state('blind'+str(site)+'_plus')-f.state('blind'+str(site)+'_minus')
   stats=np.zeros(5)
   for comp in (0,1):
    pred=np.empty(len(a))
    for start in range(0,f.g.N,32768):
     ids=np.arange(start,min(start+32768,f.g.N))
     for tile in f.g.CELLS['sd'][1]:pred[f.g.index('sd',tile,ids)]=k[ids]@de[f.c.site_map('sd',tile),comp]
    obs=a[:,comp+1]-b[:,comp+1]
    p=np.fft.fftn(pred.reshape(s.COARSE,order='F'))/len(pred)
    y=np.fft.fftn(obs.reshape(s.COARSE,order='F'))/len(obs)
    values=np.array([np.sum(abs(p-y)**2),np.sum(abs(y)**2),np.sum(abs(p[mask]-y[mask])**2),np.sum(abs(p[~mask])**2),np.sum(abs(y[~mask])**2)])
    assert abs(values[0]-np.mean((pred-obs)**2))<max(values[0],1e-100)*1e-10
    stats+=values
   total+=stats
   rows.append({'site':site,'raw_relative_L2':float(math.sqrt(stats[0]/stats[1])),'same_G_relative_L2':float(math.sqrt(stats[2]/stats[1])),'prediction_outside_G_over_observed_norm':float(math.sqrt(stats[3]/stats[1])),'observed_outside_G_over_norm':float(math.sqrt(stats[4]/stats[1]))})
  raw=math.sqrt(total[0]/total[1]);assert abs(raw-previous['metrics'][kind]['relative_response_L2'])<1e-10
  result['fields'][kind]={'raw_relative_L2':raw,'same_G_relative_L2':float(math.sqrt(total[2]/total[1])),'error_power_removed_by_matching_G':float(1-total[2]/total[0]),'sites':rows}
 with (R/'PROJECTION_CONSISTENCY.json').open('x') as out:json.dump(result,out,indent=2)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
