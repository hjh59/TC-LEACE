import tempfile
from pathlib import Path
import numpy as np
from sp27_core import *

def run():
 rng=np.random.default_rng(11);x=rng.normal(size=(500,8));z=(x[:,0]+rng.normal(size=500)>.1).astype(int);w=(x[:,1]>.2).astype(int)
 fit=fit_leace64(x,z);out=transform_leace64(x,fit);assert np.linalg.norm((out-out.mean(0)).T@(z-z.mean())/len(z))<1e-8
 cc=fit_cc2(x,z,w);out=transform_cc2(x,cc)
 for v in (0,1):
  m=w==v;assert np.linalg.norm((out[m]-out[m].mean(0)).T@(z[m]-z[m].mean())/m.sum())<1e-8
 # Identical centered groups make conditional directions exactly collinear.
 xx=np.vstack([x,x]);zz=np.r_[z,z];ww=np.r_[np.zeros(len(x)),np.ones(len(x))];assert fit_cc2(xx,zz,ww)['rank']==1
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'bad.csv';p.write_bytes(b'\0'*100);calls=[]
  def build():calls.append(1);p.write_text('x\n1\n')
  assert resume_or_recompute(p,build,expected_rows=1)=='recomputed' and len(calls)==1
 # Twenty observations, two identical strata: P scores [0,1,2,3,4], N [0,0,0,0,0].
 yy=np.tile([1]*5+[0]*5,2);ww=np.repeat([0,1],10);ss=np.tile([0,1,2,3,4]+[0]*5,2);r=conditional_auc(yy,ww,ss,min_class=1);assert abs(r['auc']-.9)<1e-15
 a=decompose_fold(w,z,np.round(x[:,1],1),np.round(x[:,2],1));b=decompose_fold(w,z,np.zeros(500),np.ones(500));aggregate_decomposition([a,b])
 groups=np.repeat(np.arange(100),2);tr=np.arange(160);te=np.arange(160,200);parts,record=identity_split(tr,te,groups,0)
 try:validate_split(np.arange(112),np.arange(112,160),tr,te,groups,'A80B20')
 except ValueError:pass
 else:raise AssertionError('70/30 accepted')
 print('PASS: LEACE; cc2+collinear; fold+aggregate T5; manual20 conditional; 70/30 guard; NUL recompute')
if __name__=='__main__':run()
