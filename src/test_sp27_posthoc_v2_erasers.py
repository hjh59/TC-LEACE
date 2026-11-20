"""POST HOC – not preregistered. Stop on the FIRST failed E2 synthetic guard."""
from pathlib import Path
import json,hashlib,subprocess,time
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score
from sp27_posthoc_v2_erasers import fit_splince,transform_splince,covariance,fit_weber,transform_weber
from sp27_core import fit_leace64,transform_leace64,logistic64
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/case_studies/candid_ptx/sp27_posthoc_v2';TITLE='POST HOC – not preregistered';PRED='960c6b479cf30ac492c68b15e3da813d1cd052c1f66c798cee3848754d5a12c3'

def main():
 rows=[];out=OUT/'E2_UNIT_TESTS.csv'
 if hashlib.sha256((OUT/'PREDICTIONS_V2.md').read_bytes()).hexdigest()!=PRED:raise RuntimeError('PREDICTIONS_HASH_CHANGED')
 (OUT/'E2_STARTED.json').write_text(json.dumps(dict(analysis_title=TITLE,predictions_sha256=PRED,E2_started_unix=time.time(),first_stage='synthetic guard tests',git_head='anonymous-release'),indent=2,ensure_ascii=False)+'\n')
 def check(name,value,bound,relation='<='):
  ok=bool(value<=bound) if relation=='<=' else bool(value==bound);rows.append(dict(analysis_title=TITLE,test=name,value=float(value),bound=float(bound),relation=relation,passed=ok));pd.DataFrame(rows).to_csv(out,index=False);print(TITLE,name,value,'PASS' if ok else 'FAIL',flush=True)
  if not ok:raise RuntimeError('E2_UNIT_TEST_FAILED:'+name)
 rng=np.random.default_rng(20261103);n=8000;d=16;z=rng.binomial(1,.35,n);w=rng.binomial(1,1/(1+np.exp(-(.4*z-.2))),n);a=rng.normal(size=d);a/=np.linalg.norm(a);b=rng.normal(size=d);b/=np.linalg.norm(b);x=rng.normal(size=(n,d))+1.4*z[:,None]*a+1.1*w[:,None]*b;at=np.arange(5000);te=np.arange(5000,n);fit=fit_splince(x[at],z[at],w[at]);P=fit['P'];cz=covariance(x[at],z[at]);cy=covariance(x[at],w[at]);leace=fit_leace64(x[at],z[at]);PL=leace['matrix'].T
 check('P_Cov_xZ_zero',np.max(abs(P@cz)),1e-8);check('P_Cov_xW_preserved',np.max(abs(P@cy-cy)),1e-8);check('projection_idempotence',np.max(abs(P@P-P)),1e-8);check('projection_rank',np.linalg.matrix_rank(P),d-1,'==');check('same_kernel_as_LEACE',np.max(abs(P@(np.eye(d)-np.linalg.pinv(PL)@PL))),1e-8)
 torch.set_num_threads(1);y=torch.as_tensor(w,dtype=torch.float64);scores=[]
 for xx in (transform_leace64(x,leace),transform_splince(x,fit)):
  data=torch.as_tensor(xx,dtype=torch.float64);c,intercept=logistic64(data[at],y[at],C=1e8);score=(data[te]@c+intercept).cpu().numpy();scores.append(roc_auc_score(w[te],score))
 check('C1e8_retrained_W_AUROC_difference',abs(scores[0]-scores[1]),1e-4)
 wf=fit_weber(x[at],z[at]);wx=transform_weber(x[at],z[at],wf);check('Weber_fit_covariance_relative',np.max(abs(covariance(wx,z[at])))/np.max(abs(cz)),1e-10)
 (OUT/'E2_UNIT_COMPLETE.json').write_text(json.dumps(dict(analysis_title=TITLE,passed=True,n_tests=len(rows),W_AUROCs=scores,predictions_sha256=PRED),indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':
 try:main()
 except Exception as error:
  (OUT/'E2_STOP.json').write_text(json.dumps(dict(analysis_title=TITLE,stage='synthetic guards',error=repr(error),no_solver_adjustment=True),indent=2,ensure_ascii=False)+'\n');raise
