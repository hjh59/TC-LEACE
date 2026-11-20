"""POST HOC – not preregistered. Computability-only bootstrap estimands."""
import numpy as np
from pairwise_decomposition import compare_scores,_weighted_auc
from sp27_posthoc_v2_metrics import CandidFold,evaluate,weights_for

def main():
 rng=np.random.default_rng(11);w=np.tile([0,0,1,1],100);z=np.tile([0,1,0,1],100);u=np.round(w+.5*z+rng.normal(size=len(w)),1);s=np.round(z+.1*w+rng.normal(size=len(w)),1);methods={}
 for name in ('raw','leace','leace_cc2'):
  d=dict(W=w,Z=z,sample_id=np.arange(len(w)).astype(str),identity_id=np.arange(len(w))//2)
  for probe in ('linear','MLP'):d['utility_'+probe]=u if name=='raw' else u-.3*z
  for prefix in ('conditional','unconditional'):
   for family in ('linear','MLP','suite'):d[prefix+'_'+family]=s
  methods[name]=d
 fold=CandidFold(methods);weights=np.ones(len(w));weights[(w==0)&(z==0)]=0;weights[np.flatnonzero((w==0)&(z==0))[:2]]=1;r=evaluate([fold],[weights]);dec=compare_scores(w,z,u,u-.3*z,weights)
 assert np.isfinite(r['leace']['linear_cross']) and abs(r['leace']['linear_cross']-dec['cross_term'])<1e-12
 for i,j in ((0,0),(1,1),(1,0),(0,1)):
  mask=((w==1)&(z==i))|((w==0)&(z==j));assert abs(r['leace'][f'linear_AUC_{i}{j}']-_weighted_auc(w[mask],(u-.3*z)[mask],weights[mask]))<1e-12
 assert abs(r['leace']['conditional_suite']-sum(r['leace'][f'conditional_weight_W{k}']*r['leace'][f'conditional_suite_W{k}'] for k in (0,1)))<1e-12
 assert np.array_equal(weights_for([fold],np.random.default_rng(21161103))[0],weights_for([fold],np.random.default_rng(21161103))[0])
 print('POST HOC – not preregistered: PASS count>=1, four-cell ties/T5, conditional stratum weights, shared draws')
if __name__=='__main__':main()
