"""POST HOC – not preregistered. Real-data gates; stop on first failure."""
from pathlib import Path
import json,subprocess,time,hashlib
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score
from sp27_plan_guard import guarded_input,file_sha256
from sp27_core import fit_leace64,transform_leace64,logistic64
from leace_cc2 import fit_cc2,transform_cc2
from sp27_posthoc_v2_erasers import fit_splince,transform_splince,fit_weber,transform_weber,covariance
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/case_studies/candid_ptx/sp27_posthoc_v2'; OLD=ROOT/'results/case_studies/candid_ptx/sp27'; TITLE='POST HOC – not preregistered'
def main():
 if (OUT/'E2_STOP.json').exists():raise RuntimeError('PREVIOUS_E2_STOP_REQUIRES_REVIEW')
 assert json.loads((OUT/'E2_UNIT_COMPLETE.json').read_text())['passed']
 assert file_sha256(OUT/'PREDICTIONS_V2.md')=='960c6b479cf30ac492c68b15e3da813d1cd052c1f66c798cee3848754d5a12c3'
 torch.set_num_threads(1);rows=[];head='anonymous-release'
 plan=ROOT/'ANALYSIS_PLAN_SP27.md';config=ROOT/'SP27_FROZEN_CONFIG.json'
 manifest=guarded_input(plan,config,'candid','manifest');m=pd.read_csv(manifest);z=m.tube.to_numpy();w=m.PTX.to_numpy()
 if len(m)!=19237:raise RuntimeError('RELEASE_SIZE_MISMATCH')
 for model in ('rad-dino','dinov2','biomedclip'):
  source=guarded_input(plan,config,'candid',model);x=np.asarray(np.load(source,mmap_mode='r'),np.float64)
  for fold in range(5):
   split=OLD/f'main_{model}'/f'fold{fold:02d}_split.npz'
   with np.load(split,allow_pickle=False) as s:at=s['A_train'];bt=s['B_train']
   mean=x[at].mean(0);sd=x[at].std(0).clip(1e-8);base=(x-mean)/sd
   for scenario,methods in [('S-out',('splince','weber')),('S-rel',('leace','leace_cc2','splince','weber'))]:
    ix=at if scenario=='S-out' else np.arange(len(x))
    for method in methods:
     folder=OUT/'E2'/'cache'/model/scenario/f'fold{fold:02d}'/method;folder.mkdir(parents=True,exist_ok=True)
     if (folder/'COMPLETE.json').exists():
      done=json.loads((folder/'COMPLETE.json').read_text())
      if done['cache_sha256']!=file_sha256(folder/'TRANSFORM.npz'):raise RuntimeError('CACHE_HASH_MISMATCH')
      rows.extend(done['checks']);continue
     fit={'leace':fit_leace64,'leace_cc2':fit_cc2,'splince':fit_splince,'weber':fit_weber}[method](base[ix],z[ix],w[ix]) if method in ('leace_cc2','splince') else {'leace':fit_leace64,'weber':fit_weber}[method](base[ix],z[ix])
     if method=='leace':xx=transform_leace64(base,fit);rank=np.linalg.matrix_rank(np.eye(x.shape[1])-fit['matrix'])
     elif method=='leace_cc2':xx=transform_cc2(base,fit);rank=fit['rank']
     elif method=='splince':xx=transform_splince(base,fit);rank=fit['projection_rank']
     else:xx=transform_weber(base,z,fit);rank=None
     checks=[]
     def check(name,value,low,high):
      row=dict(analysis_title=TITLE,model=model,scenario=scenario,fold=fold,method=method,check=name,value=float(value),lower=low,upper=high,passed=bool(low<=value<=high),git_head=head,split_sha256=file_sha256(split));rows.append(row);checks.append(row);pd.DataFrame(rows).to_csv(OUT/'GUARD_CHECKS.csv',index=False);print(json.dumps(row),flush=True)
      if not row['passed']:raise RuntimeError('REAL_DATA_GUARD_FAILED:'+name)
     for group in ((0,1) if method=='leace_cc2' else (None,)):
      gi=ix if group is None else ix[w[ix]==group];before=covariance(base[gi],z[gi]);after=covariance(xx[gi],z[gi]);check(f'fit_cov_Z_W{group}',np.max(abs(after))/np.max(abs(before)),0,1e-10)
     if method=='splince':check('fit_cov_W_preserved',np.max(abs(covariance(xx[ix],w[ix])-covariance(base[ix],w[ix])))/np.max(abs(covariance(base[ix],w[ix]))),0,1e-10)
     if rank is not None:expected=x.shape[1]-1 if method=='splince' else 2 if method=='leace_cc2' else 1;check('rank',rank,expected,expected)
     if scenario=='S-rel' and fold==0 and method in ('leace','leace_cc2'):
      f=np.column_stack((xx,w)) if method=='leace_cc2' else xx;f=(f-f[bt].mean(0))/f[bt].std(0).clip(1e-8);ft=torch.as_tensor(f,dtype=torch.float64);c,b=logistic64(ft,torch.as_tensor(z,dtype=torch.float64));scores=(ft@c+b).numpy()
      for group in ((0,1) if method=='leace_cc2' else (None,)):
       gi=np.arange(len(x)) if group is None else np.flatnonzero(w==group);check(f'release_insample_AUROC_W{group}',roc_auc_score(z[gi],scores[gi]),.48,.52)
      checks[-1]['coefficient_norm']=float(torch.linalg.vector_norm(c));pd.DataFrame(rows).to_csv(OUT/'GUARD_CHECKS.csv',index=False)
     arrays={k:np.asarray(v) for k,v in fit.items() if isinstance(v,(np.ndarray,int,float,bool,str))};arrays.update(base_mean=mean,base_sd=sd);np.savez_compressed(folder/'TRANSFORM.npz',**arrays)
     (folder/'COMPLETE.json').write_text(json.dumps(dict(analysis_title=TITLE,git_head=head,checks=checks,cache_sha256=file_sha256(folder/'TRANSFORM.npz'),split_sha256=file_sha256(split),feature_sha256=file_sha256(source)),indent=2)+'\n')
 (OUT/'E2_PREFLIGHT_COMPLETE.json').write_text(json.dumps(dict(analysis_title=TITLE,passed=True,n_checks=len(rows),git_head=head),indent=2)+'\n')
if __name__=='__main__':
 try:main()
 except Exception as e:
  (OUT/'E2_STOP.json').write_text(json.dumps(dict(analysis_title=TITLE,stage='real-data preflight',error=repr(e),time=time.time(),no_solver_adjustment=True),indent=2)+'\n');raise
