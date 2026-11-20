"""Frozen CANDID: 1000 paired within-fold patient resamples, no model fitting."""
import argparse,json,hashlib,os,subprocess,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from sp27_cached_metrics import RankedScore,weighted
from sp27_plan_guard import file_sha256,require_frozen_plan
from sp27_core import validate_file
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/case_studies/candid_ptx/sp27';METHODS=('raw','leace','leace_cc2');FAMILIES=('linear','MLP','suite');MODELS=('rad-dino','dinov2','biomedclip');SEED=20261103+900000

def atomic_json(p,o):
 q=p.with_suffix('.tmp.json');q.write_text(json.dumps(o,indent=2)+'\n');os.replace(q,p)

def load_model(model):
 folder=BASE/f'main_{model}';folds=[];repair_path=BASE/'score_metadata_repair_v1/COMPLETE.json'
 repair=json.loads(repair_path.read_text())
 if repair.get('status')!='COMPLETE' or repair.get('files_checked')!=145 or not repair.get('numeric_scores_and_labels_byte_identical') or repair.get('plan_sha256')!=file_sha256(ROOT/'ANALYSIS_PLAN_SP27.md'):raise RuntimeError('METADATA_REPAIR_NOT_VERIFIED')
 repaired={entry['source']:entry for entry in repair['files']};inputs={str(repair_path.relative_to(ROOT)):file_sha256(repair_path)};table=pd.read_csv(folder/'FOLD_RESULTS.csv')
 if len(table)!=15 or table.duplicated(['fold','method']).any():raise RuntimeError('MAIN_COVERAGE')
 for f in range(5):
  marker=json.loads((folder/f'fold{f}_COMPLETE.json').read_text());assert validate_file(folder/f'fold{f}_RESULTS.csv',expected_rows=3,checksum=marker['csv_sha256']);fold=dict(methods={})
  for method in METHODS:
   path=folder/f'fold{f}_{method}_SCORES.npz'
   if file_sha256(path)!=marker['scores'][method]:raise RuntimeError('SCORE_CHECKSUM')
   original=str(path.relative_to(ROOT));entry=repaired.get(original)
   if entry is None or entry['source_sha256']!=marker['scores'][method]:raise RuntimeError('REPAIR_SOURCE_CHECKSUM')
   inputs[original]=entry['source_sha256'];path=ROOT/entry['derived']
   if file_sha256(path)!=entry['derived_sha256']:raise RuntimeError('REPAIRED_SCORE_CHECKSUM')
   with np.load(path,allow_pickle=False) as data:
    labels={k:data['test_'+k] for k in ('W','Z','sample_id','identity_id','fold')};scores={k[5:]:data[k] for k in data.files if k.startswith('test_') and k[5:].startswith(('utility_','conditional_','unconditional_'))}
    # Keys above are converted once to their unprefixed names.
   if method=='raw':fold.update(labels)
   elif any(not np.array_equal(labels[k],fold[k]) for k in labels):raise RuntimeError('PAIRED_ALIGNMENT')
   if any(v.shape!=fold['W'].shape for v in scores.values()):raise RuntimeError('SCORE_SHAPE')
   fold['methods'][method]={k:RankedScore(v) for k,v in scores.items()};inputs[str(path.relative_to(ROOT))]=entry['derived_sha256']
  unique,inverse=np.unique(fold['identity_id'],return_inverse=True);fold['W_rank']=RankedScore(fold['W']);fold['n_clusters']=len(unique);fold['inverse']=inverse;folds.append(fold)
 return folds,inputs,table

def estimate(folds,allweights):
 out={};allw=np.concatenate([f['W'] for f in folds]);allz=np.concatenate([f['Z'] for f in folds]);aw=np.concatenate(allweights);beta=aw[(allw==1)&(allz==1)].sum()/aw[allz==1].sum()-aw[(allw==1)&(allz==0)].sum()/aw[allz==0].sum()
 for method in METHODS:
  result={};uw=[];cw=[];tw=[];vals={};eligible_cond=[];eligible_t5=[]
  for fold,weight in zip(folds,allweights):
   w,z=fold['W'],fold['Z'];s=fold['methods'][method];raw=fold['methods']['raw'];nW0=weight[w==0].sum();nW1=weight[w==1].sum();nZ0=weight[z==0].sum();nZ1=weight[z==1].sum();uw.append(float(nZ0*nZ1));tw.append(float(nW0*nW1));wc=np.array([[weight[(w==v)&(z==a)].sum() for a in (0,1)] for v in (0,1)]);cond_ok=wc.min()>=20;eligible_cond.append(cond_ok);cw.append(float(wc.prod(axis=1).sum()) if cond_ok else 0.);eligible_t5.append(cond_ok)
   def add(k,v):vals.setdefault(k,[]).append(v)
   for family in FAMILIES:
    ua=s['unconditional_'+family].attack(w,z,weight);ca=s['conditional_'+family].attack(w,z,weight);add('unconditional_'+family,ua['uncond']);add('conditional_'+family,ca['cond'])
   add('impute_true',fold['W_rank'].auc(z==1,z==0,weight))
   for probe in ('linear','MLP'):
    u=s['utility_'+probe].utility(w,z,weight);r=raw['utility_'+probe].utility(w,z,weight)
    if u is not None and r is not None:
     within=r['within']-u['within'];cross=r['cross']-u['cross'];total=r['auc']-u['auc']
     if abs(within+cross-total)>=1e-12:raise RuntimeError('BOOTSTRAP_T5_IDENTITY')
    else:within=cross=total=np.nan
    for key,value in [('within',within),('cross',cross),('total',total)]:add(probe+'_'+key,value)
    add('impute_pred_'+probe,s['utility_'+probe].auc(z==1,z==0,weight))
   
  for k,values in vals.items():result[k]=weighted(values,cw if k.startswith('conditional_') else tw if k.startswith(('linear_','MLP_')) else uw)
  for probe in ('linear','MLP'):
   if np.isfinite(result[probe+'_total']) and abs(result[probe+'_within']+result[probe+'_cross']-result[probe+'_total'])>=1e-12:raise RuntimeError('AGGREGATE_T5_IDENTITY')
  result.update(beta=float(beta),conditional_folds_used=int(sum(eligible_cond)),T5_folds_used=int(sum(eligible_t5)));out[method]=result
 for method in METHODS:
  for family in FAMILIES:
   den=out['raw']['conditional_'+family]-.5;out[method]['conditional_retention_'+family]=(out[method]['conditional_'+family]-.5)/den if den>0 else np.nan
   u=out[method]['unconditional_'+family];c=out[method]['conditional_'+family];out[method]['unconditional_minus_conditional_'+family]=u-c;out[method]['impute_share_'+family]=(u-c)/(u-.5) if u>.5 else np.nan
 return out

def model_worker(model):
 require_frozen_plan(ROOT/'ANALYSIS_PLAN_SP27.md',ROOT/'SP27_FROZEN_CONFIG.json');start=time.monotonic();folds,inputs,table=load_model(model);outdir=BASE/'statistics'/model;outdir.mkdir(parents=True,exist_ok=True)
 head='anonymous-release';config=dict(model=model,seed=SEED,replicates=1000,inputs=inputs,plan_sha256=file_sha256(ROOT/'ANALYSIS_PLAN_SP27.md'),threshold_sha256=file_sha256(BASE/'CANDID_DELTA_T.json'),sources={n:file_sha256(Path(__file__).with_name(n)) for n in ('finalize_sp27_candid.py','sp27_cached_metrics.py')});key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();atomic_json(outdir/'RUN_CONFIG.json',dict(**config,config_hash=key,git_head=head))
 point=estimate(folds,[np.ones(len(f['W'])) for f in folds])
 for method in METHODS:
  q=table[table.method==method].sort_values('fold')
  for probe in ('linear','MLP'):
   for short,col in [('within','within_term'),('cross','cross_term'),('total','total_damage')]:
    direct=float(np.average(q[probe+'_'+col],weights=q.utility_pairs))
    if abs(point[method][probe+'_'+short]-direct)>=1e-12:raise RuntimeError('POINT_ESTIMATE_VS_MAIN_T5')
  for family in FAMILIES:
   for prefix,weight in [('conditional','conditional_pairs'),('unconditional','target_pairs')]:
    direct=float(np.average(q[prefix+'_'+family+'_AUROC'],weights=q[weight]))
    if abs(point[method][prefix+'_'+family]-direct)>=1e-12:raise RuntimeError('POINT_ESTIMATE_VS_MAIN_ATTACK')
 atomic_json(outdir/'POINT_ESTIMATES.json',point);rng=np.random.default_rng(SEED);rows=[]
 for chunk in range(20):
  path=outdir/f'bootstrap_{chunk:02d}.csv';marker=path.with_suffix('.json');done=json.loads(marker.read_text()) if marker.exists() else {};restore=done.get('config_hash')==key and validate_file(path,expected_rows=150,checksum=done.get('sha256'));cr=[]
  for rep in range(chunk*50,(chunk+1)*50):
   weights=[]
   for fold in folds:
    count=np.bincount(rng.integers(0,fold['n_clusters'],fold['n_clusters']),minlength=fold['n_clusters']);weights.append(count[fold['inverse']].astype(float))
   if restore:continue
   result=estimate(folds,weights)
   for method,values in result.items():cr.append(dict(model=model,replicate=rep,method=method,**values))
  if restore:cr=pd.read_csv(path).to_dict('records')
  else:
   temp=path.with_suffix('.tmp.csv');pd.DataFrame(cr).to_csv(temp,index=False);os.replace(temp,path);atomic_json(marker,dict(config_hash=key,sha256=file_sha256(path)))
  rows.extend(cr);atomic_json(outdir/'PROGRESS.json',dict(completed_replicates=(chunk+1)*50,total_replicates=1000,seconds=time.monotonic()-start));print(model,(chunk+1)*50,'/1000',flush=True)
 df=pd.DataFrame(rows);df.to_csv(outdir/'BOOTSTRAP_REPLICATES.csv',index=False);summary=[]
 for method,values in point.items():
  q=df[df.method==method]
  for metric,value in values.items():
   vv=q[metric].to_numpy(float);vv=vv[np.isfinite(vv)];bounds=np.quantile(vv,[.025,.975,.05,.95]) if len(vv) else [np.nan]*4;summary.append(dict(model=model,method=method,metric=metric,estimate=value,ci_low=bounds[0],ci_high=bounds[1],one_sided_lower=bounds[2],one_sided_upper=bounds[3],valid_replicates=len(vv),requested_replicates=1000))
 pd.DataFrame(summary).to_csv(outdir/'ESTIMATES_WITH_CI.csv',index=False);atomic_json(outdir/'COMPLETE.json',dict(config_hash=key,git_head=head,seconds=time.monotonic()-start,replicates=1000,rows=len(df),csv_sha256=file_sha256(outdir/'BOOTSTRAP_REPLICATES.csv')));return model

def p_one(values,point,boundary,greater):
 a=np.asarray(values,float);a=a[np.isfinite(a)]
 if not len(a):return np.nan
 return float((1+np.sum(a-point>=point-boundary if greater else a-point<=point-boundary))/(len(a)+1))

def finish():
 tables=[];tests=[];delta=json.loads((BASE/'CANDID_DELTA_T.json').read_text())['delta_T']
 for model in MODELS:
  folder=BASE/'statistics'/model;tables.append(pd.read_csv(folder/'ESTIMATES_WITH_CI.csv'));df=pd.read_csv(folder/'BOOTSTRAP_REPLICATES.csv');point=json.loads((folder/'POINT_ESTIMATES.json').read_text());wide={m:df[df.method==m].sort_values('replicate').reset_index(drop=True) for m in METHODS};leace=wide['leace'];cc2=wide['leace_cc2'];p=point['leace'];c=point['leace_cc2'];h1=p_one(leace.conditional_suite,p['conditional_suite'],.5+delta,True);h2=p_one(leace.unconditional_minus_conditional_suite,p['unconditional_minus_conditional_suite'],0.,True);cross=(cc2.linear_cross-leace.linear_cross).to_numpy();cond=(cc2.conditional_suite-leace.conditional_suite).to_numpy();cp=c['linear_cross']-p['linear_cross'];cq=c['conditional_suite']-p['conditional_suite'];pc=p_one(cross,cp,0.,False);pq=p_one(cond,cq,delta,False)
  for name,values,value,bound,pv in [('H1',leace.conditional_suite.to_numpy(),p['conditional_suite'],.5+delta,h1),('H2prime',leace.unconditional_minus_conditional_suite.to_numpy(),p['unconditional_minus_conditional_suite'],0.,h2),('H4_cross',cross,cp,0.,pc),('H4_conditional',cond,cq,delta,pq)]:
   v=np.asarray(values);v=v[np.isfinite(v)];bounds=np.quantile(v,[.025,.975,.05,.95]);tests.append(dict(model=model,hypothesis=name,estimate=value,boundary=bound,p_value=pv,ci_low=bounds[0],ci_high=bounds[1],lower95=bounds[2],upper95=bounds[3],valid_replicates=len(v),primary=model=='rad-dino'))
  tests.append(dict(model=model,hypothesis='H4',p_value=max(pc,pq),primary=model=='rad-dino'))
 combined=pd.concat(tables,ignore_index=True);combined.to_csv(BASE/'statistics/CANDID_ESTIMATES_WITH_CI.csv',index=False);t=pd.DataFrame(tests);ix=t.index[(t.model=='rad-dino')&t.hypothesis.isin(['H1','H2prime','H4'])];order=sorted(ix,key=lambda i:t.loc[i,'p_value']);last=0.
 for rank,i in enumerate(order):
  if not np.isfinite(t.loc[i,'p_value']):raise RuntimeError('INCOMPLETE_HOLM_FAMILY')
  last=max(last,min(1.,(3-rank)*t.loc[i,'p_value']));t.loc[i,'p_holm']=last
 for i in t.index:
  h=t.loc[i,'hypothesis'];m=t.loc[i,'model'];boundpass=bool(t.loc[i,'lower95']>t.loc[i,'boundary']) if h in ('H1','H2prime') else bool(t.loc[i,'upper95']<t.loc[i,'boundary']) if h in ('H4_cross','H4_conditional') else bool((t[(t.model==m)&t.hypothesis.isin(['H4_cross','H4_conditional'])].upper95<t[(t.model==m)&t.hypothesis.isin(['H4_cross','H4_conditional'])].boundary).all());t.loc[i,'bounds_pass']=boundpass
  if m=='rad-dino' and h in ('H1','H2prime','H4'):t.loc[i,'formal_pass']=bool(boundpass and t.loc[i,'p_holm']<.05)
 t.to_csv(BASE/'statistics/CANDID_HYPOTHESES.csv',index=False)
 lines=['# SP27 CANDID results','',f'Frozen-plan analysis; delta_T,CANDID={delta:.8f}. 1,000 within-fold patient bootstrap replicates; all methods share draws. No model refitting. Primary RAD-DINO Holm family: H1,H2prime,H4.','', '|Model|LEACE conditional suite AUROC|95% CI|H1 interval classification|','|---|---:|---|---|']
 for model in MODELS:
  q=combined[(combined.model==model)&(combined.method=='leace')&(combined.metric=='conditional_suite')].iloc[0];label='still leaking' if q.one_sided_lower>.5+delta else 'negligible' if q.one_sided_lower>=.5-delta and q.one_sided_upper<=.5+delta else 'inconclusive';lines.append(f'|{model}|{q.estimate:.6f}|[{q.ci_low:.6f},{q.ci_high:.6f}]|{label}|')
 lines+=['','## Primary RAD-DINO decisions','']
 for _,row in t.loc[ix].iterrows():lines.append(f"- {row.hypothesis}: Holm p={row.p_holm:.6g}; formal_pass={row.formal_pass}.")
 lines+=['','Full data: CANDID_ESTIMATES_WITH_CI.csv, CANDID_HYPOTHESES.csv and per-model BOOTSTRAP_REPLICATES.csv. Undefined replicate counts and contributing-fold counts are explicit. CelebA main analysis is not included in this report.']
 (BASE/'statistics/RESULTS_SP27_CANDID.md').write_text('\n'.join(lines)+'\n');atomic_json(BASE/'statistics/COMPLETE.json',dict(models=list(MODELS),replicates_per_model=1000,finished_unix=time.time()))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=(*MODELS,'all'),default='all');a=ap.parse_args()
 if a.model=='all':
  reference=None
  for model in MODELS:
   folds,_,_=load_model(model);ids=[{k:f[k] for k in ('sample_id','identity_id','W','Z','fold')} for f in folds]
   if reference is None:reference=ids
   elif any(not np.array_equal(one[k],two[k]) for one,two in zip(reference,ids) for k in one):raise RuntimeError('CROSS_MODEL_BOOTSTRAP_INDEX_ALIGNMENT')
  with ProcessPoolExecutor(max_workers=3) as pool:list(pool.map(model_worker,MODELS))
  finish()
 else:model_worker(a.model)
if __name__=='__main__':main()
