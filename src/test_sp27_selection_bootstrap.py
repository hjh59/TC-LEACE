"""Selection must depend on correct validation estimand; methods share resamples."""
import numpy as np,pandas as pd
from sp27_attack_selection import select_attack
from sp27_core import validation_auc,ProtocolError
from sp27_bootstrap import paired_bootstrap,eligible_five_fold_pairs

def main():
 # Held-out score alignment: full-vector labels/scores must be subset together.
 yy=np.array([0,1,0,1,0,1,0,1]);ss=np.array([0.,1.,9.,0.,9.,0.,9.,0.]);iv=np.array([0,1]);assert validation_auc(yy,ss,iv)==1.
 try:validation_auc(yy[iv],ss,iv)
 except ProtocolError:pass
 else:raise AssertionError('mismatched validation vectors accepted')
 z=np.r_[np.zeros(30),np.ones(10),np.zeros(10),np.ones(30)];w=np.r_[np.zeros(40),np.ones(40)];linear=z-100*w;mlp=w
 assert select_attack(z,w,linear,mlp,True)['chosen']=='linear'
 assert select_attack(z,w,linear,mlp,False)['chosen']=='MLP'
 good=select_attack(z,w,linear,mlp,True);assert not good['fallback'] and good['selection_metric']=='conditional_AUROC'
 ix=np.r_[np.arange(39),np.arange(40,80)];bad=select_attack(z[ix],w[ix],linear[ix],mlp[ix],True);assert bad['fallback'] and bad['fallback_reason']=='B_val_WxZ_cell_lt10'
 folds=[]
 for f in range(2):
  ww=np.tile([0,0,1,1],100);zz=np.tile([0,1,0,1],100);s=zz*.7+ww*.1;u=ww*.6+zz*.1
  folds.append(dict(W=ww,Z=zz,identity_id=np.arange(400)//2+1000*f,raw_utility=u,methods={m:dict(utility=u,uncond=s,cond=s) for m in ('a','b')}))
 r=paired_bootstrap(folds,20,min_cond_class=1);assert all(row['a']==row['b'] and row['a']['valid'] for row in r)
 assert all(abs(row['a']['within']+row['a']['cross']-row['a']['total'])<1e-12 for row in r)
 table=pd.DataFrame(dict(pair_id=[0]*5+[1]*4,fold=list(range(5))+list(range(4)),fold_eligible=[True]*9));assert eligible_five_fold_pairs(table)==[0]
 print('PASS: separate selection, 10-cell boundary/fallback, paired within-fold bootstrap, T5 identity, five-fold inclusion')
if __name__=='__main__':main()
