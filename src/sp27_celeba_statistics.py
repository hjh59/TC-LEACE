"""Frozen CelebA foldwise statistics; no training or historical result inputs."""
import hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import ndtr,ndtri
from sp27_cached_metrics import RankedScore
from sp27_plan_guard import file_sha256,require_frozen_plan,guarded_input
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/case_studies/celeba/analysis/sp27_confirmatory';OUT=ROOT/'results/case_studies/celeba/analysis/sp27_statistics';METHODS=('raw','leace','leace_cc2');MODELS=('clip','dinov2','siglip','mae');FAMILIES=('linear','MLP','suite');SEED=20261103+900000;BLOCK_SEED=SEED+1
_CANON=None;_AUDIT=None

def atomic_json(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,path)

def sha_config(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()

def pairs():
 require_frozen_plan(ROOT/'ANALYSIS_PLAN_SP27.md',ROOT/'SP27_FROZEN_CONFIG.json');meta=json.loads((BASE/'COUNT_ONLY_MANIFEST.json').read_text())
 if file_sha256(BASE/'QUALIFIED_PAIRS.csv')!=meta['qualified_pairs_sha256']:raise RuntimeError('QUALIFICATION_HASH')
 p=pd.read_csv(BASE/'QUALIFIED_PAIRS.csv').sort_values('pair_id').reset_index(drop=True)
 if len(p)!=85 or p.pair_id.duplicated().any():raise RuntimeError('QUALIFICATION_COVERAGE')
 return p

def canonical():
 global _CANON
 if _CANON is not None:return _CANON
 plan=ROOT/'ANALYSIS_PLAN_SP27.md';config=ROOT/'SP27_FROZEN_CONFIG.json';m=pd.read_csv(guarded_input(plan,config,'celeba','manifest'),sep='\t');m=m[m.split.eq('confirmatory')].reset_index(drop=True);ids=m.image_filename.to_numpy(dtype=str);g=m.identity_id.to_numpy();index={name:i for i,name in enumerate(ids)};y=np.full((len(m),40),-1,np.int8);found=0
 with guarded_input(plan,config,'celeba','attributes').open() as f:
  n=int(f.readline());names=f.readline().split()
  if len(names)!=40:raise RuntimeError('ATTRIBUTE_HEADER')
  for line in f:
   parts=line.split();i=index.get(parts[0])
   if i is not None:y[i]=(np.asarray(parts[1:],np.int8)==1);found+=1
 if found!=len(m) or (y<0).any():raise RuntimeError('ATTRIBUTE_COVERAGE')
 meta=json.loads((BASE/'COUNT_ONLY_MANIFEST.json').read_text());folds=[]
 for fold in range(5):
  path=BASE/f'fold{fold:02d}_A80B20.npz'
  if file_sha256(path)!=meta['folds'][fold]['split_sha256']:raise RuntimeError('FROZEN_SPLIT_HASH')
  with np.load(path,allow_pickle=False) as d:folds.append({k:d[k] for k in d.files})
 _CANON=dict(ids=ids,groups=g,labels=y,names=names,folds=folds);return _CANON

def audit_pair(task):
 model,pair_id=task;c=canonical();pair=pairs().set_index('pair_id').loc[pair_id];wi=c['names'].index(pair.W);zi=c['names'].index(pair.Z);records=[]
 for f in range(5):
  folder=BASE/model/f'fold{f:02d}'/f'pair{pair_id:03d}';marker=json.loads((folder/'COMPLETE.json').read_text());csv=folder/'METHOD_RESULTS.csv';npz=folder/'SCORES.npz';config=folder/'RUN_CONFIG.json';run=json.loads(config.read_text())
  if marker['config_hash']!=sha_config({k:v for k,v in run.items() if k!='git_head'}):raise RuntimeError('MAIN_CONFIG_HASH')
  if file_sha256(csv)!=marker['csv_sha256'] or file_sha256(npz)!=marker['npz_sha256']:raise RuntimeError('MAIN_ARTIFACT_HASH')
  table=pd.read_csv(csv)
  if len(table)!=3 or set(table.method)!=set(METHODS) or table.method.duplicated().any() or not table.pair_id.eq(pair_id).all() or not table.fold.eq(f).all():raise RuntimeError('MAIN_CSV_COVERAGE')
  parts=c['folds'][f];keep=np.sort(np.r_[parts['A_val'],parts['B_val'],parts['test']]);roles=np.full(len(c['ids']),'',dtype='<U8')
  for role,ix in parts.items():
   if role not in ('A','B'):roles[ix]=role
  expected=dict(sample_id=c['ids'][keep],identity_id=c['groups'][keep],W=c['labels'][keep,wi],Z=c['labels'][keep,zi],fold=np.full(len(keep),f),partition=roles[keep])
  with np.load(npz,allow_pickle=False) as data:
   arrays={k:data[k] for k in data.files}
  for k,v in arrays.items():
   if v.shape!=(len(keep),) or v.dtype.hasobject:raise RuntimeError('MAIN_ARRAY_SHAPE_OR_OBJECT')
   if np.issubdtype(v.dtype,np.number) and not np.isfinite(v).all():raise RuntimeError('MAIN_ARRAY_NONFINITE')
  for k,v in expected.items():
   if not np.array_equal(arrays[k],v):raise RuntimeError('RAW_METADATA_ALIGNMENT:'+k)
  for method in METHODS:
   row=table[table.method.eq(method)].iloc[0]
   if float(row.covariance_error)>=1e-8 or max(abs(row.linear_identity_error),abs(row.MLP_identity_error))>=1e-12:raise RuntimeError('SAVED_NUMERIC_GATE')
   for prefix in ('conditional','unconditional'):
    selected=row[prefix+'_selected'];key=f'{method}__{prefix}_'
    if not np.array_equal(arrays[key+'suite'],arrays[key+selected]):raise RuntimeError('SUITE_SELECTION_SCORE_MISMATCH')
   for key in ('A_train_linear_attack','utility_linear','utility_MLP')+tuple(prefix+'_'+family for prefix in ('conditional','unconditional') for family in FAMILIES):
    if arrays[f'{method}__{key}'].dtype!=np.float64:raise RuntimeError('SCORE_NOT_FLOAT64')
  records.append(dict(model=model,pair_id=int(pair_id),fold=f,score_path=str(npz.relative_to(ROOT)),score_sha256=marker['npz_sha256'],csv_sha256=marker['csv_sha256'],run_config_sha256=file_sha256(config),n_kept=len(keep),n_test=len(parts['test']),npz_all_arrays_readable=True,raw_metadata_alignment=True))
 return records

class FoldScores:
 def __init__(self,arrays):
  mask=arrays['partition']=='test';self.w=arrays['W'][mask];self.z=arrays['Z'][mask];self.ids=arrays['sample_id'][mask];self.groups=arrays['identity_id'][mask];u,self.inverse=np.unique(self.groups,return_inverse=True);self.n_clusters=len(u);self.cells={(a,b):(self.w==a)&(self.z==b) for a in (0,1) for b in (0,1)};self.scores={};self.ranks={}
  for method in METHODS:
   for probe in ('linear','MLP'):
    k=f'{method}__utility_{probe}';v=arrays[k][mask];self.scores[k]=v;self.ranks[k]=RankedScore(v)
   for prefix in ('conditional','unconditional'):
    for family in ('linear','MLP'):
     k=f'{method}__{prefix}_{family}';self.ranks[k]=RankedScore(arrays[k][mask])
    suite=arrays[f'{method}__{prefix}_suite'][mask]
    matches=[family for family in ('linear','MLP') if np.array_equal(suite,arrays[f'{method}__{prefix}_{family}'][mask])]
    if not matches:raise RuntimeError('SUITE_NOT_VALIDATION_CHOSEN_FAMILY')
    self.ranks[f'{method}__{prefix}_suite']=self.ranks[f'{method}__{prefix}_{matches[0]}']
 def counts(self,weights):return np.array([[weights[self.cells[a,b]].sum() for b in (0,1)] for a in (0,1)])
 def utility(self,method,probe,weights,counts):
  score=self.ranks[f'{method}__utility_{probe}'];den=counts[1].sum()*counts[0].sum()
  if counts.min()<20 or den<=0:return None
  aucs={};cw={};within=cross=0.
  for i,j in ((0,0),(1,1),(0,1),(1,0)):
   value=score.auc(self.cells[1,i],self.cells[0,j],weights);weight=counts[1,i]*counts[0,j]/den;aucs[i,j]=value;cw[i,j]=weight
   if i==j:within+=weight*value
   else:cross+=weight*value
  auc=score.auc(self.w==1,self.w==0,weights)
  if abs(within+cross-auc)>=1e-12:raise RuntimeError('FOLD_T5_WEIGHTED_IDENTITY')
  return dict(aucs=aucs,cell_weights=cw,within=within,cross=cross,auc=auc,pairs=den)

def complete_average(values,weights):
 values=np.asarray(values,float);weights=np.asarray(weights,float)
 if not np.isfinite(values).all() or not (weights>0).all():return np.nan
 return float(np.average(values,weights=weights))

def struct_pred(a00,a11,pi0,pi1,shift=0.):
 d0,d1=np.sqrt(2)*ndtri(np.clip([a00,a11],.001,.999));off=pi0*d0-pi1*d1+shift
 return float(ndtr((d1+off)/np.sqrt(2))),float(ndtr((d0-off)/np.sqrt(2)))

def score_shift(fold,method,probe,weights,counts):
 score=fold.scores[f'{method}__utility_{probe}'];means=[];variances=[]
 for z in (0,1):
  ix=fold.z==z;sw=weights[ix].sum();means.append(float(np.dot(weights[ix],score[ix])/sw) if sw>0 else np.nan)
 for w,z in ((0,0),(0,1),(1,0),(1,1)):
  ix=fold.cells[w,z];sw=counts[w,z]
  if sw<=0:return np.nan
  mean=float(np.dot(weights[ix],score[ix])/sw);variances.append(float(np.dot(weights[ix],(score[ix]-mean)**2)/sw))
 sd=np.sqrt(np.mean(variances));return (means[1]-means[0])/sd if sd>0 else np.nan

def estimate(folds,weights,include_structural=True):
 pooled=sum((f.counts(q) for f,q in zip(folds,weights)),np.zeros((2,2)));beta=pooled[1,1]/pooled[:,1].sum()-pooled[1,0]/pooled[:,0].sum();out={m:{} for m in METHODS};records={m:{} for m in METHODS};uw=[];cw=[];tw=[];t5ok=[];condok=[]
 for fold,weight in zip(folds,weights):
  counts=fold.counts(weight);up=counts.sum(axis=0).prod();tp=counts.sum(axis=1).prod();cp=counts.prod(axis=1);uw.append(up);cw.append(cp.sum());tw.append(tp);t5ok.append(counts.min()>=100);condok.append(counts.min()>=30);pi0=counts[1,0]/counts[:,0].sum();pi1=counts[1,1]/counts[:,1].sum();fbeta=pi1-pi0;raw={p:fold.utility('raw',p,weight,counts) for p in ('linear','MLP')}
  for method in METHODS:
   def add(k,v):records[method].setdefault(k,[]).append(v)
   for family in FAMILIES:
    r=fold.ranks[f'{method}__unconditional_{family}'];add('unconditional_'+family,r.auc(fold.z==1,fold.z==0,weight))
    r=fold.ranks[f'{method}__conditional_{family}'];a=[r.auc(fold.cells[w,1],fold.cells[w,0],weight) for w in (0,1)];add('conditional_'+family,float(np.average(a,weights=cp)) if condok[-1] else np.nan)
   add('impute_true',.5+fbeta/2)
   for probe in ('linear','MLP'):
    u=raw[probe] if method=='raw' else fold.utility(method,probe,weight,counts);r=raw[probe]
    within=r['within']-u['within'] if u is not None and r is not None else np.nan;cross=r['cross']-u['cross'] if u is not None and r is not None else np.nan;total=r['auc']-u['auc'] if u is not None and r is not None else np.nan
    if np.isfinite(total) and abs(within+cross-total)>=1e-12:raise RuntimeError('FOLD_DAMAGE_IDENTITY')
    for k,v in [('within',within),('cross',cross),('total',total)]:add(probe+'_'+k,v if t5ok[-1] else np.nan)
    add('impute_pred_'+probe,fold.ranks[f'{method}__utility_{probe}'].auc(fold.z==1,fold.z==0,weight))
    if include_structural:
     shift=score_shift(fold,method,probe,weight,counts);add(probe+'_shift_sd',shift)
     for kind in ('guard','shift','advance'):
      state=r if kind=='advance' else u
      if state is None or r is None or not np.isfinite(shift):pred=np.nan
      else:
       a10,a01=struct_pred(state['aucs'][0,0],state['aucs'][1,1],pi0,pi1,shift if kind=='shift' else 0.);pred=r['cross']-u['cell_weights'][1,0]*a10-u['cell_weights'][0,1]*a01
      add(probe+'_h3_'+kind+'_prediction',pred);add(probe+'_h3_'+kind+'_error',pred-cross)
     add(probe+'_h3_cross_obs',cross)
 for method in METHODS:
  for k,values in records[method].items():out[method][k]=complete_average(values,cw if k.startswith('conditional_') else tw if k.startswith(('linear_','MLP_')) else uw)
  for probe in ('linear','MLP'):
   r=out[method]
   if np.isfinite(r[probe+'_total']) and abs(r[probe+'_within']+r[probe+'_cross']-r[probe+'_total'])>=1e-12:raise RuntimeError('AGGREGATE_DAMAGE_IDENTITY')
  out[method].update(beta=float(beta),abs_beta=float(abs(beta)),conditional_folds_used=int(sum(condok)),T5_folds_used=int(sum(t5ok)))
 for method in METHODS:
  for family in FAMILIES:
   u=out[method]['unconditional_'+family];c=out[method]['conditional_'+family];den=out['raw']['conditional_'+family]-.5;out[method]['conditional_retention_'+family]=(c-.5)/den if den>0 else np.nan;out[method]['unconditional_minus_conditional_'+family]=u-c;out[method]['impute_share_'+family]=(u-c)/(u-.5) if u>.5 else np.nan
 return out

def load_pair(model,pair_id,audit):
 entries=audit['files'];lookup={(r['model'],r['pair_id'],r['fold']):r for r in entries};folds=[];tables=[]
 for f in range(5):
  item=lookup[model,pair_id,f];path=ROOT/item['score_path']
  if file_sha256(path)!=item['score_sha256']:raise RuntimeError('POST_AUDIT_SCORE_MODIFIED')
  with np.load(path,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
  folds.append(FoldScores(arrays));table=pd.read_csv(path.parent/'METHOD_RESULTS.csv');tables.append(table)
 table=pd.concat(tables,ignore_index=True);point=estimate(folds,[np.ones(len(f.w)) for f in folds])
 for method in METHODS:
  q=table[table.method.eq(method)].sort_values('fold');testcounts=np.array([[int(q.iloc[f][f'n_W{w}Z{z}']) for w in (0,1) for z in (0,1)] for f in range(5)]).reshape(5,2,2);up=testcounts.sum(axis=1).prod(axis=1);cp=testcounts.prod(axis=2).sum(axis=1)
  for family in FAMILIES:
   for prefix,weights in [('conditional',cp),('unconditional',up)]:
    reference=float(np.average(q[prefix+'_'+family+'_AUROC'],weights=weights))
    if abs(point[method][prefix+'_'+family]-reference)>=1e-12:raise RuntimeError('POINT_ATTACK_VS_TRAINING_ROWS')
  for probe in ('linear','MLP'):
   for metric in ('within','cross','total'):
    reference=float(np.average(q[probe+'_'+metric],weights=q.utility_pair_count))
    if abs(point[method][probe+'_'+metric]-reference)>=1e-12:raise RuntimeError('POINT_T5_VS_TRAINING_ROWS')
 return folds,point,table

def resample_weights(folds,rng):
 return [np.bincount(rng.integers(0,f.n_clusters,f.n_clusters),minlength=f.n_clusters)[f.inverse].astype(float) for f in folds]
