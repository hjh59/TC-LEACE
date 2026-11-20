import numpy as np
from sp27_cached_metrics import RankedScore
from pairwise_decomposition import one_score,_weighted_auc

def main():
 rng=np.random.default_rng(17);w=np.tile([0,0,1,1],100);z=np.tile([0,1,0,1],100);s=rng.integers(0,7,len(w))/7;weights=rng.integers(0,4,len(w)).astype(float);r=RankedScore(s);a=r.utility(w,z,weights);b=one_score(w,z,s,weights)
 assert abs(a['auc']-b['pooled_auc'])<1e-12
 for i,j in ((0,0),(1,1),(0,1),(1,0)):
  assert abs(a[f'AUC_{i}{j}']-b['cells'][i,j]['auc'])<1e-12
  assert abs(a[f'w_{i}{j}']-b['cells'][i,j]['weight'])<1e-12
 t=r.attack(w,z,weights);assert abs(t['uncond']-_weighted_auc(z,s,weights))<1e-12
 cc=[];pp=[]
 for v in (0,1):
  ix=w==v;cc.append(_weighted_auc(z[ix],s[ix],weights[ix]));pp.append(weights[ix&(z==0)].sum()*weights[ix&(z==1)].sum())
 assert abs(t['cond']-np.average(cc,weights=pp))<1e-12
 weights[(w==0)&(z==0)]=0;assert r.utility(w,z,weights) is None and np.isnan(r.attack(w,z,weights)['cond'])
 print('PASS: weighted ties, four cells, conditional pair weights and <20-cell NA')
if __name__=='__main__':main()
