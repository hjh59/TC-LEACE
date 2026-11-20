"""Read-only verification of released aggregate results. No model fitting."""
from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parent/'results'
def rows(path):return list(csv.DictReader((ROOT/path).open(encoding='utf-8')))
def check(label,actual,expected):
 assert abs(actual-expected)<=5e-4,(label,actual,expected)
 print(label,repr(actual),'PASS')
S=Path('case_studies/celeba/analysis/sp27_statistics')
a=rows(S/'AGGREGATE_CI.csv');models=['clip','dinov2','siglip','mae']
def point(m,k):return float(next(r['estimate'] for r in a if r['model']==m and r['beta_band']=='all' and r['metric']==k))
for m,e in zip(models,[.900135,.895219,.899308,.859500]):check(m+' LEACE conditional',point(m,'leace__conditional_suite'),e)
null=rows('case_studies/celeba/analysis/sp27_s3/CONDITIONAL_NULL.csv');delta=max(.02,np.quantile([float(r['suite_cond_AUROC']) for r in null],.975)-.5);check('CelebA delta',delta,.0233)
def p_one(v,theta,bound,greater):
 v=v[np.isfinite(v)];assert len(v)
 return (1+np.sum(v-theta>=theta-bound if greater else v-theta<=theta-bound))/(len(v)+1)
ps=[]
with np.load(ROOT/S/'diagnostics/PRIMARY_BOOTSTRAP_ARRAYS.npz',allow_pickle=False) as boot:
 for i,m in enumerate(models):
  assert str(boot['models'][i])==m
  ps.extend([(m,'H1',p_one(boot['H1'][i],point(m,'leace__conditional_suite'),.5+delta,True)),(m,'H2',p_one(boot['H2'][i],point(m,'leace__slope'),0,True)),(m,'H4',max(p_one(boot['H4_lin'][i],point(m,'cc2_minus_leace__linear_cross'),0,False),p_one(boot['H4_cond'][i],point(m,'cc2_minus_leace__conditional_suite'),delta,False)))])
order=np.argsort([p[2] for p in ps]);last=0
for rank,j in enumerate(order):
 m,h,p=ps[j];last=max(last,min(1,(len(ps)-rank)*p));expected=.011988 if h!='H4' else (.0560224 if m=='mae' else .109244);check(m+' '+h+' Holm',last,expected)
C=Path('case_studies/candid_ptx/sp27');c=rows(C/'statistics/CANDID_ESTIMATES_WITH_CI.csv')
def cp(method,metric):return float(next(r['estimate'] for r in c if r['model']=='rad-dino' and r['method']==method and r['metric']==metric))
for method,metric,e in [('leace','conditional_suite',.7706),('leace_cc2','conditional_suite',.7667),('leace','unconditional_suite',.804),('leace','linear_total',.176),('leace_cc2','linear_total',.042)]:check('CANDID '+method+' '+metric,cp(method,metric),e)
check('CANDID cross difference',cp('leace_cc2','linear_cross')-cp('leace','linear_cross'),-.105)
n=rows(C/'null_rad-dino/CONDITIONAL_NULL.csv');check('CANDID delta',max(.02,np.quantile([float(r['cond_auc']) for r in n],.975)-.5),.0785)
pb=rows('case_studies/candid_ptx/sp27_posthoc_v2_2/P_B_EXPLORATORY.csv')
for w,e in [(0,-.0614),(1,-.0472)]:
 r=next(r for r in pb if r['model']=='rad-dino' and r['metric']=='conditional_linear_W'+str(w));check('V2.2 P-b W='+str(w),float(r['estimate']),e)
print('ALL RELEASE NUMERIC CHECKS PASSED')
