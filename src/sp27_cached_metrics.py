"""Tie-exact weighted AUROCs with fixed score ranks; no refitting or pooling."""
import numpy as np

class RankedScore:
 def __init__(self,score):
  score=np.asarray(score,dtype=np.float64)
  if not np.isfinite(score).all():raise ValueError('NONFINITE_SCORE')
  levels,self.codes=np.unique(score,return_inverse=True);self.nlevels=len(levels)
 def auc(self,positive,negative,weights):
  p=np.bincount(self.codes,weights=weights*positive,minlength=self.nlevels);n=np.bincount(self.codes,weights=weights*negative,minlength=self.nlevels);den=p.sum()*n.sum()
  return float(p@(np.cumsum(n)-.5*n)/den) if den>0 else np.nan
 def utility(self,w,z,weights,min_cell=20):
  counts=np.array([[weights[(w==a)&(z==b)].sum() for b in (0,1)] for a in (0,1)]);den=counts[1].sum()*counts[0].sum()
  if counts.min()<min_cell or den<=0:return None
  values={};within=cross=0.
  for i,j in ((0,0),(1,1),(0,1),(1,0)):
   a=self.auc((w==1)&(z==i),(w==0)&(z==j),weights);weight=counts[1,i]*counts[0,j]/den;values['AUC_'+str(i)+str(j)]=a;values['w_'+str(i)+str(j)]=float(weight)
   if i==j:within+=weight*a
   else:cross+=weight*a
  pooled=self.auc(w==1,w==0,weights)
  if abs(within+cross-pooled)>=1e-12:raise RuntimeError('WEIGHTED_AUC_IDENTITY')
  return dict(**values,within=float(within),cross=float(cross),auc=pooled,pairs=float(den))
 def attack(self,w,z,weights,min_cell=20):
  counts=np.array([[weights[(w==a)&(z==b)].sum() for b in (0,1)] for a in (0,1)]);uncond=self.auc(z==1,z==0,weights);pairs=counts.sum(axis=0).prod();cp=counts.prod(axis=1)
  conditional=np.nan
  if counts.min()>=min_cell:conditional=float(np.average([self.auc((z==1)&(w==a),(z==0)&(w==a),weights) for a in (0,1)],weights=cp))
  return dict(uncond=uncond,pairs=float(pairs),cond=conditional,cond_pairs=float(cp.sum()) if np.isfinite(conditional) else 0.)

def weighted(values,weights):
 values=np.asarray(values,float);weights=np.asarray(weights,float);ok=np.isfinite(values)&(weights>0)
 return float(np.average(values[ok],weights=weights[ok])) if ok.any() else np.nan
