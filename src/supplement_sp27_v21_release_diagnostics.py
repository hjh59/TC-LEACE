"""POST HOC – not preregistered. Analytic W-only comparator; no training."""
from pathlib import Path
import numpy as np,pandas as pd,subprocess
from sp27_plan_guard import guarded_input,file_sha256
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/case_studies/candid_ptx/sp27_posthoc_v2';OLD=ROOT/'results/case_studies/candid_ptx/sp27'
def main():
 frame=pd.read_csv(guarded_input(ROOT/'ANALYSIS_PLAN_SP27.md',ROOT/'SP27_FROZEN_CONFIG.json','candid','manifest'));z=frame.tube.to_numpy();w=frame.PTX.to_numpy();p=z.mean();n=len(z);sw=np.where(z==1,1/(2*p),1/(2*(1-p)));prob=np.zeros(n)
 for g in (0,1):
  ix=w==g;q=z[ix].mean();prob[ix]=(q/p)/(q/p+(1-q)/(1-p))
 residual=(prob-z)*sw;d=pd.read_csv(OUT/'RELEASE_INSAMPLE_DIAGNOSTICS_V21.csv');rows=[]
 for _,row in d.iterrows():
  split=OLD/f'main_{row.model}'/f'fold{int(row.fold):02d}_split.npz'
  with np.load(split,allow_pickle=False) as s:bt=s['B_train']
  fw=(w-w[bt].mean())/max(w[bt].std(),1e-8);r=row.to_dict();r['analytic_W_only_full_comparator_W_gradient']=float(np.mean(fw*residual));r['analytic_W_only_intercept_gradient']=float(residual.mean());r['analytic_W_only_comparator_definition']='unpenalized intercept and W; frozen class-balanced logistic loss; no fit';r['source_diagnostics_sha256']=file_sha256(OUT/'RELEASE_INSAMPLE_DIAGNOSTICS_V21.csv');r['git_head']='anonymous-release';rows.append(r)
 pd.DataFrame(rows).to_csv(OUT/'RELEASE_INSAMPLE_DIAGNOSTICS_WITH_COMPARATOR_V21.csv',index=False)
 print('POST HOC – not preregistered: analytical comparator supplemented for',len(rows),'diagnostic rows; no training')
if __name__=='__main__':main()
