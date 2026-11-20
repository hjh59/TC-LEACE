"""POST HOC – not preregistered. Saved-score statistics with computability-only gates."""
import numpy as np
from sp27_cached_metrics import RankedScore
# EXACT shared H3 definition used in the reported CelebA main analysis.
from sp27_celeba_statistics import struct_pred
TITLE='POST HOC – not preregistered'

def avg(a,w):
 a=np.asarray(a,float);w=np.asarray(w,float);ok=np.isfinite(a)&(w>0)
 return float(np.average(a[ok],weights=w[ok])) if ok.any() else np.nan

class CandidFold:
 def __init__(self,methods):
  first=next(iter(methods.values()));self.w=first['W'];self.z=first['Z'];self.ids=first['sample_id'];self.groups=first['identity_id'];self.methods=methods;self.ranks={};u,self.inverse=np.unique(self.groups,return_inverse=True);self.ng=len(u);self.cells={(w,z):(self.w==w)&(self.z==z) for w in (0,1) for z in (0,1)}
  for name,data in methods.items():
   for field in ('W','Z','sample_id','identity_id'):
    if not np.array_equal(data[field],first[field]):raise RuntimeError('V2_PAIRED_SAMPLE_ALIGNMENT')
   self.ranks[name]={k:RankedScore(v) for k,v in data.items() if k.startswith(('utility_','conditional_','unconditional_'))}
 def counts(self,weights):return np.array([[weights[self.cells[w,z]].sum() for z in (0,1)] for w in (0,1)])

def weights_for(folds,rng):return [np.bincount(rng.integers(0,f.ng,f.ng),minlength=f.ng)[f.inverse].astype(float) for f in folds]

def evaluate(folds,weights,raw_name='raw'):
 names=tuple(folds[0].methods);values={n:{} for n in names};kinds={};cweights=[];uweights=[];vweights=[];cellweights={ij:[] for ij in ((0,0),(1,1),(1,0),(0,1))};groupweights=[[],[]];groupweights_total=[[],[]];fold_estimable=[];allcounts=np.zeros((2,2))
 for fold,q in zip(folds,weights):
  c=fold.counts(q);fold_estimable.append(bool(c.min()>=1));allcounts+=c;cp=c.prod(axis=1);up=c.sum(axis=0).prod();vp=c.sum(axis=1).prod();uweights.append(up);vweights.append(vp);cweights.append(cp.sum());base={}
  for w in (0,1):groupweights[w].append(cp[w]);groupweights_total[w].append(float(c[w].sum()))
  for i,j in cellweights:cellweights[i,j].append(c[1,i]*c[0,j])
  def utility(name,probe):
   rank=fold.ranks[name]['utility_'+probe];aa={};ww={}
   for i,j in cellweights:
    aa[i,j]=rank.auc(fold.cells[1,i],fold.cells[0,j],q) if min(c[1,i],c[0,j])>=1 else np.nan;ww[i,j]=c[1,i]*c[0,j]/vp if vp>0 else np.nan
   pooled=rank.auc(fold.w==1,fold.w==0,q);inside=sum(ww[i,i]*aa[i,i] for i in (0,1));cross=sum(ww[i,j]*aa[i,j] for i,j in ((1,0),(0,1)))
   if c.min()>=1 and abs(inside+cross-pooled)>=1e-12:raise RuntimeError('V2_UTILITY_AUC_IDENTITY')
   return aa,ww,pooled,inside,cross
  for probe in ('linear','MLP'):base[probe]=utility(raw_name,probe)
  for name in names:
   def add(key,value,kind):values[name].setdefault(key,[]).append(value);kinds[key]=kind
   for family in ('linear','MLP','suite'):
    ur=fold.ranks[name]['unconditional_'+family];cr=fold.ranks[name]['conditional_'+family];add('unconditional_'+family,ur.auc(fold.z==1,fold.z==0,q),'u');ca=[]
    for w in (0,1):
     a=cr.auc(fold.cells[w,1],fold.cells[w,0],q) if c[w].min()>=1 else np.nan;ca.append(a);add(f'conditional_{family}_W{w}',a,'g'+str(w))
    cond=float(np.average(ca,weights=cp)) if c.min()>=1 else np.nan;add('conditional_'+family,cond,'c')
   beta=c[1,1]/c[:,1].sum()-c[1,0]/c[:,0].sum() if c.sum(axis=0).min()>0 else np.nan;add('impute_true',.5+beta/2,'u')
   for probe in ('linear','MLP'):
    a,ww,pooled,inside,cross=base[probe] if name==raw_name else utility(name,probe);ra,rw,rpooled,rinside,rcross=base[probe];within=rinside-inside;crossloss=rcross-cross;total=rpooled-pooled
    if c.min()>=1 and abs(within+crossloss-total)>=1e-12:raise RuntimeError('V2_T5_DAMAGE_IDENTITY')
    for key,value in [('within',within),('cross',crossloss),('total',total)]:add(probe+'_'+key,value if c.min()>=1 else np.nan,'v')
    for i,j in cellweights:add(f'{probe}_AUC_{i}{j}',a[i,j],'ij'+str(i)+str(j))
    add('impute_pred_'+probe,fold.ranks[name]['utility_'+probe].auc(fold.z==1,fold.z==0,q),'u')
    score=fold.methods[name]['utility_'+probe];variances=[]
    for w,z in ((0,0),(0,1),(1,0),(1,1)):
     mask=fold.cells[w,z];n=c[w,z];mean=np.dot(q[mask],score[mask])/n if n>0 else np.nan;variances.append(np.dot(q[mask],(score[mask]-mean)**2)/n if n>0 else np.nan)
    n0=c[:,0].sum();n1=c[:,1].sum();mu0=np.dot(q[fold.z==0],score[fold.z==0])/n0 if n0>0 else np.nan;mu1=np.dot(q[fold.z==1],score[fold.z==1])/n1 if n1>0 else np.nan;sd=np.sqrt(np.mean(variances));shift=(mu1-mu0)/sd if sd>0 else np.nan;add(probe+'_shift_sd',shift,'v')
    pi0=c[1,0]/n0 if n0>0 else np.nan;pi1=c[1,1]/n1 if n1>0 else np.nan
    for kind in ('guard','shift','advance'):
     target=ra if kind=='advance' else a
     if c.min()>=1 and (kind!='shift' or np.isfinite(shift)):
      p10,p01=struct_pred(target[0,0],target[1,1],pi0,pi1,shift if kind=='shift' else 0.);prediction=rcross-ww[1,0]*p10-ww[0,1]*p01
     else:prediction=np.nan
     add(probe+'_h3_'+kind+'_prediction',prediction,'v');add(probe+'_h3_'+kind+'_error',prediction-crossloss,'v')
 result={name:{} for name in names};weightmap={'u':uweights,'v':vweights,'c':cweights,'g0':groupweights[0],'g1':groupweights[1]};weightmap.update({'ij'+str(i)+str(j):w for (i,j),w in cellweights.items()})
 for name in names:
  for metric,a in values[name].items():result[name][metric]=avg(a,weightmap[kinds[metric]])
  for w in (0,1):result[name][f'conditional_weight_W{w}']=float(np.sum(groupweights[w])/np.sum(cweights))
  for family in ('linear','MLP','suite'):
   den=result[raw_name]['conditional_'+family]-.5 if raw_name in result and 'conditional_'+family in result[raw_name] else np.nan
   # retention is added after all methods are aggregated below.
   u=result[name]['unconditional_'+family];co=result[name]['conditional_'+family];result[name]['unconditional_minus_conditional_'+family]=u-co;result[name]['impute_share_'+family]=(u-co)/(u-.5) if u>.5 else np.nan
  for probe in ('linear','MLP'):result[name][probe+'_tube_shortcut_gain']=result[name][probe+'_AUC_10']-result[name][probe+'_AUC_00']
  result[name]['conditional_folds_used']=int(sum(fold_estimable));result[name]['T5_folds_used']=int(sum(fold_estimable))
  result[name]['beta']=float(allcounts[1,1]/allcounts[:,1].sum()-allcounts[1,0]/allcounts[:,0].sum())
 for name in names:
  for family in ('linear','MLP','suite'):
   den=result[raw_name]['conditional_'+family]-.5;result[name]['conditional_retention_'+family]=(result[name]['conditional_'+family]-.5)/den if den>0 else np.nan
 return result
