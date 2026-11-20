"""SP27 CelebA: audit -> timed blocks -> resumable bootstrap -> frozen inference."""
import argparse,json,os,time,subprocess,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from pathlib import Path
import numpy as np,pandas as pd
import sp27_celeba_statistics as S
from sp27_celeba_statistics import ROOT,BASE,OUT,METHODS,MODELS,FAMILIES,SEED,BLOCK_SEED,atomic_json,sha_config,pairs,canonical,audit_pair,load_pair,estimate,resample_weights,struct_pred
from sp27_plan_guard import file_sha256,require_frozen_plan

def source_hashes():return {name:file_sha256(Path(__file__).with_name(name)) for name in ('finalize_sp27_celeba.py','sp27_celeba_statistics.py','sp27_cached_metrics.py','sp27_plan_guard.py')}

def pool_map_failfast(function,tasks,workers,label):
 pool=ProcessPoolExecutor(max_workers=workers,mp_context=get_context('fork'));futures={pool.submit(function,t):t for t in tasks};results=[]
 try:
  for i,future in enumerate(as_completed(futures)):
   results.append(future.result())
   if (i+1)%5==0 or i+1==len(tasks):print(label,i+1,'/',len(tasks),flush=True)
 except BaseException as e:
  atomic_json(OUT/'STOP.json',dict(stage=label,error=repr(e),traceback=traceback.format_exc(),time_unix=time.time()))
  for proc in list((pool._processes or {}).values()):proc.terminate()
  pool.shutdown(wait=False,cancel_futures=True);raise
 else:pool.shutdown(wait=True)
 return results

def audit_all(workers):
 require_frozen_plan(ROOT/'ANALYSIS_PLAN_SP27.md',ROOT/'SP27_FROZEN_CONFIG.json');p=pairs();canonical();tasks=[(m,int(pair)) for m in MODELS for pair in p.pair_id];start=time.monotonic();records=[]
 for values in pool_map_failfast(audit_pair,tasks,min(workers,8),'audit'):records.extend(values)
 records.sort(key=lambda r:(r['model'],r['pair_id'],r['fold']))
 if len(records)!=1700:raise RuntimeError('AUDIT_EXPECTED_1700_SCORE_FILES')
 report=dict(status='COMPLETE',score_files=1700,method_rows=5100,files=records,plan_sha256=file_sha256(ROOT/'ANALYSIS_PLAN_SP27.md'),qualification_sha256=file_sha256(BASE/'QUALIFIED_PAIRS.csv'),git_head='anonymous-release',elapsed_seconds=time.monotonic()-start)
 path=OUT/'INPUT_AUDIT.json'
 if path.exists():
  previous=json.loads(path.read_text())
  if previous['files']!=records or previous['plan_sha256']!=report['plan_sha256']:raise RuntimeError('AUDITED_MAIN_INPUT_CHANGED')
 else:atomic_json(path,report)
 pd.DataFrame(records).to_csv(OUT/'INPUT_AUDIT.csv',index=False)
 tables=[pd.read_csv(BASE/m/f'fold{f:02d}/FOLD_SUMMARY.csv') for m in MODELS for f in range(5)];allrows=pd.concat(tables,ignore_index=True)
 if len(allrows)!=5100 or allrows.duplicated(['model','pair_id','fold','method']).any():raise RuntimeError('FOLD_MATRIX_COVERAGE')
 allrows.to_csv(OUT/'METHOD_FOLD_RESULTS.csv',index=False);allrows.to_csv(OUT/'PAIRWISE_DECOMPOSITION.csv',index=False);print('PASS input audit:1700 score files,5100 rows',flush=True)
 return json.loads(path.read_text())

def structural_rows(model,pair_id,folds,point,table):
 rows=[];pair=pairs().set_index('pair_id').loc[pair_id]
 for f,fold in enumerate(folds):
  weight=np.ones(len(fold.w));counts=fold.counts(weight);pi0=counts[1,0]/counts[:,0].sum();pi1=counts[1,1]/counts[:,1].sum()
  for method in METHODS:
   for probe in ('linear','MLP'):
    raw=fold.utility('raw',probe,weight,counts);erased=fold.utility(method,probe,weight,counts);score=fold.scores[f'{method}__utility_{probe}'];row=dict(model=model,pair_id=pair_id,W=pair.W,Z=pair.Z,fold=f,method=method,probe=probe,pi0=pi0,pi1=pi1,beta=pi1-pi0,shift_sd=S.score_shift(fold,method,probe,weight,counts),cross_obs=raw['cross']-erased['cross'],utility_pairs=raw['pairs'])
    for i,j in ((0,0),(1,1),(0,1),(1,0)):
     row[f'auc{i}{j}_obs']=erased['aucs'][i,j];row[f'w_{i}{j}']=erased['cell_weights'][i,j]
    for w,z in ((0,0),(0,1),(1,0),(1,1)):row[f'variance_W{w}Z{z}']=float(np.var(score[fold.cells[w,z]]))
    for kind in ('guard','shift','advance'):
     state=raw if kind=='advance' else erased;a10,a01=struct_pred(state['aucs'][0,0],state['aucs'][1,1],pi0,pi1,row['shift_sd'] if kind=='shift' else 0.)
     row[kind+'_auc10_pred']=a10;row[kind+'_auc01_pred']=a01;row[kind+'_auc10_error']=a10-erased['aucs'][1,0];row[kind+'_auc01_error']=a01-erased['aucs'][0,1];row[kind+'_cross_pred']=raw['cross']-erased['cell_weights'][1,0]*a10-erased['cell_weights'][0,1]*a01;row[kind+'_cross_error']=row[kind+'_cross_pred']-row['cross_obs']
    rows.append(row)
 return rows

def chunk_valid(path,marker,key,shape,names,begin):
 try:
  d=json.loads(marker.read_text())
  if d['config_hash']!=key or d['sha256']!=file_sha256(path):return False
  with np.load(path,allow_pickle=False) as z:
   v=z['values'];mask=z['finite_mask'];ids=z['replicate'];metrics=z['metrics']
   if v.shape!=shape or not np.array_equal(mask,np.isfinite(v)) or np.isinf(v).any() or not np.array_equal(ids,np.arange(begin,begin+shape[0])) or not np.array_equal(metrics,np.asarray(names,dtype=str)):return False
  return True
 except Exception:return False

def pair_worker(task):
 model,pair_id=task
 if (OUT/'STOP.json').exists():raise RuntimeError('PIPELINE_STOP_FLAG')
 audit=json.loads((OUT/'INPUT_AUDIT.json').read_text());start=time.monotonic();folds,point,table=load_pair(model,pair_id,audit);folder=OUT/model/f'pair{pair_id:03d}';folder.mkdir(parents=True,exist_ok=True);names=list(point['raw']);config=dict(model=model,pair_id=pair_id,runtime_manifest_sha256=os.environ.get('SP27_RUNTIME_MANIFEST_SHA256'),seed=SEED,block_seed=BLOCK_SEED,replicates=1000,split_config='A80B20',input_audit_sha256=file_sha256(OUT/'INPUT_AUDIT.json'),plan_sha256=file_sha256(ROOT/'ANALYSIS_PLAN_SP27.md'),sources=source_hashes(),chunk_size=100,bootstrap_T5_minimum_cell=100,bootstrap_conditional_minimum_cell=30,missing_fold_policy='NA, never drop a fold in CelebA',metric_names=names);key=sha_config(config);head='anonymous-release';atomic_json(folder/'RUN_CONFIG.json',dict(**config,config_hash=key,git_head=head));atomic_json(folder/'POINT_ESTIMATES.json',point);pd.DataFrame(structural_rows(model,pair_id,folds,point,table)).to_csv(folder/'STRUCTURAL_FOLD.csv',index=False);rng=np.random.default_rng(SEED);chunks=[]
 for chunk in range(10):
  if (OUT/'STOP.json').exists():raise RuntimeError('PIPELINE_STOP_FLAG')
  path=folder/f'bootstrap_{chunk:02d}.npz';marker=path.with_suffix('.json');restored=chunk_valid(path,marker,key,(100,3,len(names)),names,chunk*100) if path.exists() and marker.exists() else False;values=[]
  for rep in range(chunk*100,(chunk+1)*100):
   weights=resample_weights(folds,rng)
   if restored:continue
   result=estimate(folds,weights);values.append([[result[m][n] for n in names] for m in METHODS])
  if restored:
   with np.load(path,allow_pickle=False) as z:v=z['values']
  else:
   v=np.asarray(values,np.float64);tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,values=v,finite_mask=np.isfinite(v),replicate=np.arange(chunk*100,(chunk+1)*100),metrics=np.asarray(names,dtype=str));os.replace(tmp,path);atomic_json(marker,dict(config_hash=key,sha256=file_sha256(path)))
   if not chunk_valid(path,marker,key,(100,3,len(names)),names,chunk*100):raise RuntimeError('WRITTEN_BOOTSTRAP_CHUNK_INVALID')
  chunks.append(v);atomic_json(folder/'PROGRESS.json',dict(model=model,pair_id=pair_id,completed_replicates=(chunk+1)*100,total_replicates=1000,elapsed_seconds=time.monotonic()-start));print('bootstrap',model,pair_id,(chunk+1)*100,'/1000',flush=True)
 v=np.concatenate(chunks);summary=[]
 for mi,m in enumerate(METHODS):
  for ki,k in enumerate(names):
   a=v[:,mi,ki];a=a[np.isfinite(a)];q=np.quantile(a,[.025,.975,.05,.95]) if len(a) else [np.nan]*4;summary.append(dict(model=model,pair_id=pair_id,method=m,metric=k,estimate=point[m][k],ci_low=q[0],ci_high=q[1],lower95=q[2],upper95=q[3],valid_replicates=len(a),requested_replicates=1000))
 pd.DataFrame(summary).to_csv(folder/'ESTIMATES_WITH_CI.csv',index=False);atomic_json(folder/'COMPLETE.json',dict(model=model,pair_id=pair_id,config_hash=key,git_head=head,replicates=1000,seconds=time.monotonic()-start,summary_sha256=file_sha256(folder/'ESTIMATES_WITH_CI.csv')));return dict(model=model,pair_id=pair_id,seconds=time.monotonic()-start)

def benchmark(audit,workers):
 p=pairs().sort_values(['abs_beta','pair_id']);tasks=[(m,int(pair)) for m in ('clip','mae') for pair in (p.iloc[0].pair_id,p.iloc[-1].pair_id)];rows=[]
 for model,pair in tasks:
  start=time.monotonic();folds,point,table=load_pair(model,pair,audit);loaded=time.monotonic()-start;rng=np.random.default_rng(SEED);start=time.monotonic()
  for rep in range(5):estimate(folds,resample_weights(folds,rng))
  seconds=(time.monotonic()-start)/5;rows.append(dict(model=model,pair_id=pair,replicates=5,seconds_per_replicate=seconds,load_and_point_validation_seconds=loaded));print('benchmark',model,pair,seconds,'seconds/rep',flush=True)
 report=dict(benchmark_only=True,protocol_unchanged=True,workers=workers,rows=rows,projected_compute_hours=float(np.mean([r['seconds_per_replicate'] for r in rows])*1000*340/workers/3600),source_hashes=source_hashes());atomic_json(OUT/'BENCHMARK.json',report);return report

def p_one(values,point,boundary,greater):
 a=np.asarray(values,float);a=a[np.isfinite(a)]
 if not len(a):return np.nan
 return float((1+np.sum(a-point>=point-boundary if greater else a-point<=point-boundary))/(len(a)+1))

def slope(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float)
 if len(x)<2 or not np.isfinite(x).all() or not np.isfinite(y).all() or np.var(x)<=np.finfo(float).eps:return np.nan
 return float(np.dot(x-x.mean(),y-y.mean())/np.sum((x-x.mean())**2))

def leave_pair_out(x,y,ids):
 x=np.asarray(x,float);y=np.asarray(y,float);ids=np.asarray(ids);unique,inverse=np.unique(ids,return_inverse=True);n=len(x);cnt=np.bincount(inverse);sx=np.bincount(inverse,weights=x);sy=np.bincount(inverse,weights=y);sxx=np.bincount(inverse,weights=x*x);sxy=np.bincount(inverse,weights=x*y);nn=n-cnt;xx=x.sum()-sx;yy=y.sum()-sy;xxx=np.dot(x,x)-sxx;xy=np.dot(x,y)-sxy;den=xxx-xx*xx/np.maximum(nn,1)
 if (nn<2).any() or (den<=np.finfo(float).eps).any() or not np.isfinite(x).all() or not np.isfinite(y).all():return np.full(n,np.nan)
 b=(xy-xx*yy/nn)/den;a=yy/nn-b*xx/nn;return a[inverse]+b[inverse]*x

def summary_metrics(data,names,indices,pair_ids):
 # data shape(pair,method,metric); NA is propagated for selected fixed pairs.
 v=data[indices];ids=pair_ids[indices];lookup={n:i for i,n in enumerate(names)};out={};x=v[:,0,lookup['abs_beta']]
 for mi,m in enumerate(METHODS):
  for k,ki in lookup.items():
   a=v[:,mi,ki];out[m+'__'+k]=float(np.mean(a)) if np.isfinite(a).all() and len(a) else np.nan
  out[m+'__slope']=slope(x,v[:,mi,lookup['unconditional_minus_conditional_suite']])
  for probe in ('linear','MLP'):
   obs=v[:,mi,lookup[probe+'_h3_cross_obs']];pred=v[:,mi,lookup[probe+'_h3_guard_prediction']];cv=leave_pair_out(x,obs,ids);den=np.sum((obs-obs.mean())**2) if len(obs) else 0
   for label,prediction in [('guard',pred),('zero',np.zeros(len(obs))),('beta',cv),('shift',v[:,mi,lookup[probe+'_h3_shift_prediction']]),('advance',v[:,mi,lookup[probe+'_h3_advance_prediction']])]:
    good=np.isfinite(obs).all() and np.isfinite(prediction).all() and len(obs)>0;out[f'{m}__{probe}_h3_{label}_MAE']=float(np.mean(abs(prediction-obs))) if good else np.nan;out[f'{m}__{probe}_h3_{label}_R2']=float(1-np.sum((prediction-obs)**2)/den) if good and den>0 else np.nan
   for baseline in ('zero','beta'):out[f'{m}__{probe}_h3_guard_vs_{baseline}_MAE_improvement']=out[f'{m}__{probe}_h3_{baseline}_MAE']-out[f'{m}__{probe}_h3_guard_MAE']
 for probe in ('linear','MLP'):
  for metric in ('within','cross','total'):out[f'cc2_minus_leace__{probe}_{metric}']=out['leace_cc2__'+probe+'_'+metric]-out['leace__'+probe+'_'+metric]
 for family in FAMILIES:
  for prefix in ('conditional','unconditional'):out[f'cc2_minus_leace__{prefix}_{family}']=out['leace_cc2__'+prefix+'_'+family]-out['leace__'+prefix+'_'+family]
 return out

def summarize_model(model,p):
 ids=p.pair_id.to_numpy();names=None;points=[];boot=[];pairtables=[];struct=[]
 for pair in ids:
  folder=OUT/model/f'pair{pair:03d}';done=json.loads((folder/'COMPLETE.json').read_text());run=json.loads((folder/'RUN_CONFIG.json').read_text());point=json.loads((folder/'POINT_ESTIMATES.json').read_text());nn=run['metric_names']
  if names is None:names=nn
  elif names!=nn:raise RuntimeError('METRIC_NAMES_CHANGED')
  if file_sha256(folder/'ESTIMATES_WITH_CI.csv')!=done['summary_sha256']:raise RuntimeError('SUMMARY_CHECKSUM')
  if run['sources']!=source_hashes():raise RuntimeError('BOOTSTRAP_SOURCE_CHANGED')
  chunks=[]
  for c in range(10):
   path=folder/f'bootstrap_{c:02d}.npz'
   if not chunk_valid(path,path.with_suffix('.json'),done['config_hash'],(100,3,len(names)),names,c*100):raise RuntimeError('SUMMARY_CHUNK_INVALID')
   with np.load(path,allow_pickle=False) as d:chunks.append(d['values'])
  boot.append(np.concatenate(chunks));points.append([[point[m][k] for k in names] for m in METHODS]);pairtables.append(pd.read_csv(folder/'ESTIMATES_WITH_CI.csv'));struct.append(pd.read_csv(folder/'STRUCTURAL_FOLD.csv'))
 point=np.asarray(points);boot=np.stack(boot,axis=1);target=p.Z.to_numpy();blocks=np.unique(target);rng=np.random.default_rng(BLOCK_SEED);group_order=np.argsort(np.rec.fromarrays([p.abs_beta.to_numpy(),ids]),kind='stable');bands={g:np.sort(ix) for g,ix in zip(('low','middle','high'),np.array_split(group_order,3))};bands['all']=np.arange(len(p));pointout={g:summary_metrics(point,names,ix,ids) for g,ix in bands.items()};repout={g:[] for g in bands}
 for rep in range(1000):
  draw=rng.choice(blocks,len(blocks),replace=True);selected=np.concatenate([np.flatnonzero(target==z) for z in draw])
  for group,allowed in bands.items():repout[group].append(summary_metrics(boot[rep],names,selected[np.isin(selected,allowed)],ids))
 rows=[];rep_rows=[]
 for group,metrics in pointout.items():
  for k,value in metrics.items():
   values=np.array([b[k] for b in repout[group]],float);good=values[np.isfinite(values)];q=np.quantile(good,[.025,.975,.05,.95]) if len(good) else [np.nan]*4;rows.append(dict(model=model,beta_band=group,metric=k,estimate=value,ci_low=q[0],ci_high=q[1],lower95=q[2],upper95=q[3],valid_replicates=len(good),requested_replicates=1000,n_pairs=len(bands[group])))
  for rep,values in enumerate(repout[group]):rep_rows.append(dict(model=model,beta_band=group,replicate=rep,**values))
 folder=OUT/model;pd.DataFrame(rows).to_csv(folder/'AGGREGATE_CI.csv',index=False);pd.DataFrame(rep_rows).to_csv(folder/'AGGREGATE_REPLICATES.csv',index=False);pd.concat(pairtables).to_csv(folder/'PAIR_ESTIMATES_WITH_CI.csv',index=False);pd.concat(struct).to_csv(folder/'STRUCTURAL_PREDICTION.csv',index=False);pd.DataFrame([dict(pair_id=int(ids[i]),W=p.iloc[i].W,Z=p.iloc[i].Z,beta_band=g) for g,indices in bands.items() if g!='all' for i in indices]).to_csv(folder/'BETA_BANDS.csv',index=False)
 return model

def final_report():
 p=pairs();summary=pd.concat([pd.read_csv(OUT/m/'AGGREGATE_CI.csv') for m in MODELS],ignore_index=True);pair=pd.concat([pd.read_csv(OUT/m/'PAIR_ESTIMATES_WITH_CI.csv') for m in MODELS],ignore_index=True);struct=pd.concat([pd.read_csv(OUT/m/'STRUCTURAL_PREDICTION.csv') for m in MODELS],ignore_index=True);summary.to_csv(OUT/'AGGREGATE_CI.csv',index=False);pair.to_csv(OUT/'PAIR_ESTIMATES_WITH_CI.csv',index=False);struct.to_csv(OUT/'STRUCTURAL_PREDICTION.csv',index=False);delta=json.loads((ROOT/'SP27_FROZEN_CONFIG.json').read_text())['delta_T_CelebA'];tests=[]
 for model in MODELS:
  point=summary[(summary.model==model)&summary.beta_band.eq('all')].set_index('metric');rep=pd.read_csv(OUT/model/'AGGREGATE_REPLICATES.csv');rep=rep[rep.beta_band.eq('all')]
  for h,metric,bound,greater in [('H1','leace__conditional_suite',.5+delta,True),('H2','leace__slope',0.,True),('H4_cross','cc2_minus_leace__linear_cross',0.,False),('H4_conditional','cc2_minus_leace__conditional_suite',delta,False)]:
   row=point.loc[metric];pv=p_one(rep[metric],row.estimate,bound,greater);tests.append(dict(model=model,hypothesis=h,estimate=row.estimate,boundary=bound,ci_low=row.ci_low,ci_high=row.ci_high,lower95=row.lower95,upper95=row.upper95,valid_replicates=row.valid_replicates,p_value=pv,bounds_pass=bool(row.lower95>bound if greater else row.upper95<bound)))
  two=tests[-2:];tests.append(dict(model=model,hypothesis='H4',p_value=max(t['p_value'] for t in two) if all(np.isfinite(t['p_value']) for t in two) else np.nan,bounds_pass=all(t['bounds_pass'] for t in two)))
 t=pd.DataFrame(tests);ix=t.index[t.hypothesis.isin(['H1','H2','H4'])];order=sorted(ix,key=lambda i:t.loc[i,'p_value']);family_complete=bool(len(ix)==12 and np.isfinite(t.loc[ix,'p_value']).all());last=0.
 if family_complete:
  for rank,i in enumerate(order):last=max(last,min(1.,(12-rank)*t.loc[i,'p_value']));t.loc[i,'p_holm']=last;t.loc[i,'formal_pass']=bool(t.loc[i,'bounds_pass'] and last<.05)
 else:t['family_status']='incomplete; no complete confirmatory family claim'
 t.to_csv(OUT/'CELEBA_HYPOTHESES.csv',index=False)
 # Pairwise H1 interval labels are descriptive, with no pairwise Holm claim.
 cls=[]
 for (model,pair_id),q in pair[(pair.method=='leace')&pair.metric.eq('conditional_suite')].groupby(['model','pair_id']):
  row=q.iloc[0];label='still leaking' if row.lower95>.5+delta else 'negligible' if row.lower95>=.5-delta and row.upper95<=.5+delta else 'inconclusive';cls.append(dict(model=model,pair_id=pair_id,label=label))
 pd.DataFrame(cls).to_csv(OUT/'PAIR_H1_DESCRIPTIVE.csv',index=False);make_figures(summary,pair,t);lines=['# SP27 results','',f'CelebA frozen main analysis:4models×85pairs×5folds×3methods. delta_T={delta:.8f}. 1,000 within-fold identity replicates plus common Z-block resampling for cross-pair summaries. Undefined replicate counts are explicit. Primary Holm family=12 tests; complete={family_complete}.','', '|Model|H1 Holm p/pass|H2 Holm p/pass|H4 Holm p/pass|','|---|---|---|---|']
 for model in MODELS:
  q=t[t.model.eq(model)].set_index('hypothesis');cells=[f"{q.loc[h].get('p_holm',np.nan):.6g}/{q.loc[h].get('formal_pass',False)}" for h in ('H1','H2','H4')];lines.append('|'+model+'|'+'|'.join(cells)+'|')
 lines+=['','CANDID primary RAD-DINO results are recorded separately in results/case_studies/candid_ptx/sp27/statistics/RESULTS_SP27_CANDID.md; that family is not combined with CelebA Holm12.','', 'Artifacts: CELEBA_HYPOTHESES.csv, AGGREGATE_CI.csv, PAIR_ESTIMATES_WITH_CI.csv, STRUCTURAL_PREDICTION.csv, PAIRWISE_DECOMPOSITION.csv, F1–F4 CSV and PNG, TABLE2_STRUCTURAL.csv.','', 'Inference is conditional on the already fitted OOF models; no eraser/probe refit was performed. Bootstrap T5 support is100 per W×Z cell and conditional support30; any missing required CelebA fold yields NA rather than dropping the fold. Z blocks are resampled without silently dropping undefined selected pairs. The optional MANCE++ arm is not part of this completed three-method analysis.']
 mainrows=['','## CelebA primary estimates (two-sided95% intervals)','','|Model|LEACE conditional suite AUROC|H2 slope|cc2 minus LEACE linear cross loss|','|---|---|---|---|']
 for model in MODELS:
  q=summary[(summary.model==model)&summary.beta_band.eq('all')].set_index('metric');values=[]
  for metric in ('leace__conditional_suite','leace__slope','cc2_minus_leace__linear_cross'):
   row=q.loc[metric];values.append(f'{row.estimate:.6f} [{row.ci_low:.6f},{row.ci_high:.6f}]')
  mainrows.append('|'+model+'|'+'|'.join(values)+'|')
 lines+=mainrows
 candid=ROOT/'results/case_studies/candid_ptx/sp27/statistics/CANDID_ESTIMATES_WITH_CI.csv'
 if candid.exists():pd.read_csv(candid).to_csv(OUT/'TABLE1_CANDID.csv',index=False)
 (OUT/'RESULTS_SP27.md').write_text('\n'.join(lines)+'\n');atomic_json(OUT/'COMPLETE.json',dict(status='COMPLETE',models=list(MODELS),pairs=85,replicates=1000,Holm_family_complete=family_complete,plan_sha256=file_sha256(ROOT/'ANALYSIS_PLAN_SP27.md'),finished_unix=time.time()))

def make_figures(summary,pair,tests):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors=dict(zip(MODELS,('#4477aa','#ee6677','#228833','#aa3377')))
 # F1 keeps the A_train in-sample result descriptive; its uncertainty is not a test-fold bootstrap.
 main=pd.read_csv(OUT/'METHOD_FOLD_RESULTS.csv');elig=pd.read_csv(BASE/'ELIGIBILITY_TABLE.csv');a=elig[elig.partition.eq('A_train')];f1=[]
 for model in MODELS:
  values=[]
  for pair_id,q in main[(main.model==model)&main.method.eq('leace')].groupby('pair_id'):
   weights=[]
   for f in range(5):
    z=a[(a.pair_id==pair_id)&a.fold.eq(f)].groupby('Z_value').n_images.sum();weights.append(float(z[0]*z[1]))
   values.append(np.average(q.sort_values('fold').A_insample_linear_AUROC,weights=weights))
  f1.append(dict(model=model,stage='A_train_linear',estimate=float(np.mean(values)),ci_low=np.nan,ci_high=np.nan))
  for stage,metric in [('test_linear','leace__unconditional_linear'),('test_MLP','leace__unconditional_MLP')]:
   q=summary[(summary.model==model)&summary.beta_band.eq('all')&summary.metric.eq(metric)].iloc[0];f1.append(dict(model=model,stage=stage,estimate=q.estimate,ci_low=q.ci_low,ci_high=q.ci_high))
 f1=pd.DataFrame(f1);f1.to_csv(OUT/'F1_THREE_LAYERS.csv',index=False);fig,ax=plt.subplots(figsize=(7,4))
 for model in MODELS:
  q=f1[f1.model==model];ax.plot(range(3),q.estimate,'o-',label=model,color=colors[model]);ax.fill_between(range(3),q.ci_low,q.ci_high,color=colors[model],alpha=.12)
 ax.set_xticks(range(3),['A train linear','Test linear','Test MLP']);ax.set_ylabel('Target AUROC (LEACE)');ax.axhline(.5,color='gray',ls=':');ax.legend();fig.tight_layout();fig.savefig(OUT/'F1_THREE_LAYERS.png',dpi=180);plt.close(fig)
 f2=summary[summary.metric.str.contains('__(unconditional_|conditional_|impute_)',regex=True)].copy();f2.to_csv(OUT/'F2_CONDITIONAL_VS_UNCONDITIONAL.csv',index=False);p0=pair[(pair.method=='leace')&pair.metric.isin(['abs_beta','unconditional_minus_conditional_suite'])].pivot(index=['model','pair_id'],columns='metric',values='estimate').reset_index();fig,ax=plt.subplots(figsize=(6,4))
 for model in MODELS:
  q=p0[p0.model==model];ax.scatter(q.abs_beta,q.unconditional_minus_conditional_suite,s=13,alpha=.6,label=model,color=colors[model]);reg=summary[(summary.model==model)&summary.beta_band.eq('all')&summary.metric.eq('leace__slope')].iloc[0];xx=np.array([0.,q.abs_beta.max()]);intercept=q.unconditional_minus_conditional_suite.mean()-reg.estimate*q.abs_beta.mean();ax.plot(xx,intercept+reg.estimate*xx,color=colors[model]);ax.text(.02,.98-.065*list(MODELS).index(model),f'{model}: slope={reg.estimate:.3f} [{reg.ci_low:.3f},{reg.ci_high:.3f}]',transform=ax.transAxes,va='top',fontsize=8,color=colors[model])
 ax.set_xlabel('|beta|');ax.set_ylabel('Unconditional minus conditional AUROC');ax.legend();fig.tight_layout();fig.savefig(OUT/'F2_IMPUTATION_SCATTER.png',dpi=180);plt.close(fig);p0.to_csv(OUT/'F2_IMPUTATION_SCATTER.csv',index=False)
 f3=summary[summary.metric.str.match(r'(raw|leace|leace_cc2)__(linear|MLP)_(within|cross|total)$')];f3.to_csv(OUT/'F3_DAMAGE.csv',index=False);fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,probe in zip(axes,('linear','MLP')):
  for model in MODELS:
   q=f3[(f3.model==model)&f3.metric.eq('leace__'+probe+'_cross')&~f3.beta_band.eq('all')].set_index('beta_band').reindex(['low','middle','high']);ax.plot(range(3),q.estimate,'o-',label=model,color=colors[model]);ax.fill_between(range(3),q.ci_low,q.ci_high,color=colors[model],alpha=.1)
  ax.set_xticks(range(3),['low','middle','high']);ax.set_title(probe+' utility');ax.set_xlabel('|beta| band');ax.set_ylabel('LEACE cross loss')
 axes[0].legend();fig.tight_layout();fig.savefig(OUT/'F3_DAMAGE.png',dpi=180);plt.close(fig)
 f4=summary[summary.metric.str.startswith('cc2_minus_leace__')];f4.to_csv(OUT/'F4_CC2_MINUS_LEACE.csv',index=False);fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,metric in zip(axes,('linear_cross','conditional_suite')):
  for model in MODELS:
   q=f4[(f4.model==model)&f4.metric.eq('cc2_minus_leace__'+metric)&~f4.beta_band.eq('all')].set_index('beta_band').reindex(['low','middle','high']);ax.plot(range(3),q.estimate,'o-',label=model,color=colors[model]);ax.fill_between(range(3),q.ci_low,q.ci_high,color=colors[model],alpha=.1)
  ax.axhline(0,color='gray',ls=':');ax.set_xticks(range(3),['low','middle','high']);ax.set_title(metric);ax.set_xlabel('|beta| band')
 axes[0].legend();fig.tight_layout();fig.savefig(OUT/'F4_CC2_MINUS_LEACE.png',dpi=180);plt.close(fig)
 summary[summary.metric.str.contains('_h3_')].to_csv(OUT/'TABLE2_STRUCTURAL.csv',index=False)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('all','audit','benchmark','run','summarize'),default='all');ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 if a.workers<1 or a.workers>int(os.environ.get('SLURM_CPUS_PER_TASK',a.workers)):raise RuntimeError('WORKERS_EXCEED_EXPLICIT_CPU_ALLOCATION')
 if (OUT/'STOP.json').exists():raise RuntimeError('PRIOR_FAILURE_REQUIRES_REVIEW')
 require_frozen_plan(ROOT/'ANALYSIS_PLAN_SP27.md',ROOT/'SP27_FROZEN_CONFIG.json');start=time.monotonic();atomic_json(OUT/'PIPELINE_STATUS.json',dict(stage=a.stage,status='running',workers=a.workers,time_unix=time.time()))
 if a.stage in ('all','audit'):audit=audit_all(a.workers)
 else:audit=json.loads((OUT/'INPUT_AUDIT.json').read_text())
 if a.stage in ('all','benchmark'):benchmark(audit,a.workers)
 if a.stage in ('all','run'):
  p=pairs();pool_map_failfast(pair_worker,[(m,int(pair)) for m in MODELS for pair in p.pair_id],a.workers,'bootstrap_pairs')
 if a.stage in ('all','summarize'):
  p=pairs()
  for model in MODELS:summarize_model(model,p)
  final_report()
 atomic_json(OUT/'PIPELINE_STATUS.json',dict(stage=a.stage,status='complete',workers=a.workers,elapsed_seconds=time.monotonic()-start,time_unix=time.time()))
if __name__=='__main__':
 try:main()
 except Exception as e:
  OUT.mkdir(parents=True,exist_ok=True);atomic_json(OUT/'STOP.json',dict(error=repr(e),traceback=traceback.format_exc(),time_unix=time.time()));raise
