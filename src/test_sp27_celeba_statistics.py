"""Real estimands, tied scores, fold support and common cluster draws."""
import numpy as np
from pairwise_decomposition import compare_scores,_weighted_auc
from sp27_celeba_statistics import FoldScores,estimate,complete_average,resample_weights,METHODS,FAMILIES,SEED,struct_pred

def toy(n=1600):
 rng=np.random.default_rng(19);w=np.tile([0,0,1,1],n//4);z=np.tile([0,1,0,1],n//4);d=dict(W=w,Z=z,sample_id=np.arange(n).astype(str),identity_id=np.arange(n)//2,partition=np.full(n,'test'));raw=w+.4*z+rng.integers(0,7,n)/7;erased=raw-.25*z
 for method in METHODS:
  for probe in ('linear','MLP'):d[f'{method}__utility_{probe}']=raw if method=='raw' else erased
  for prefix in ('conditional','unconditional'):
   for family in FAMILIES:d[f'{method}__{prefix}_{family}']=np.round(z*.4+w*.2+rng.normal(0,1,n),1) if family!='suite' else d[f'{method}__{prefix}_linear']
 return d

def main():
 d=toy();fold=FoldScores(d);weights=np.random.default_rng(3).integers(0,4,len(fold.w)).astype(float);out=estimate([fold],[weights]);dec=compare_scores(fold.w,fold.z,d['raw__utility_linear'],d['leace__utility_linear'],weights)
 for a,b in [('within','within_term'),('cross','cross_term'),('total','total_damage')]:assert abs(out['leace']['linear_'+a]-dec[b])<1e-12
 score=d['leace__unconditional_linear'];assert abs(out['leace']['unconditional_linear']-_weighted_auc(fold.z,score,weights))<1e-12
 vals=[];pairs=[]
 for value in (0,1):
  ix=fold.w==value;vals.append(_weighted_auc(fold.z[ix],d['leace__conditional_linear'][ix],weights[ix]));pairs.append(weights[ix&(fold.z==0)].sum()*weights[ix&(fold.z==1)].sum())
 assert abs(out['leace']['conditional_linear']-np.average(vals,weights=pairs))<1e-12
 a=resample_weights([fold,fold],np.random.default_rng(SEED));b=resample_weights([fold,fold],np.random.default_rng(SEED));assert all(np.array_equal(x,y) for x,y in zip(a,b))
 assert np.isnan(complete_average([.6,np.nan],[10,20]))
 limited=np.ones(len(fold.w));limited[fold.cells[0,0]]=0;limited[np.flatnonzero(fold.cells[0,0])[:50]]=1
 bad=estimate([fold],[limited]);assert bad['leace']['T5_folds_used']==0 and np.isnan(bad['leace']['linear_cross']) and np.isfinite(bad['leace']['conditional_suite'])
 limited[fold.cells[0,0]]=0;limited[np.flatnonzero(fold.cells[0,0])[:29]]=1;assert np.isnan(estimate([fold],[limited])['leace']['conditional_suite'])
 a10,a01=struct_pred(.7,.7,.5,.5);assert abs(a10-.7)<1e-12 and abs(a01-.7)<1e-12
 print('PASS: exact weighted ties/T5; conditional pair weighting; common draws; 100/30 support gates; missing-fold NA; structural formula')
if __name__=='__main__':main()
