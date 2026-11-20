"""POST HOC – not preregistered. E1 saved-score analyses; NO model fitting."""
import argparse,json,os,time,hashlib,subprocess,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
from scipy.special import ndtr
from sp27_posthoc_v2_metrics import CandidFold,evaluate,weights_for,TITLE
from sp27_cached_metrics import RankedScore
from sp27_plan_guard import require_frozen_plan,guarded_input,file_sha256
from sp27_celeba_statistics import FoldScores
ROOT=Path(__file__).resolve().parents[1];COLD=ROOT/'results/case_studies/candid_ptx/sp27';SOLD=ROOT/'results/case_studies/celeba/analysis/sp27_statistics';MAIN=ROOT/'results/case_studies/celeba/analysis/sp27_confirmatory';COUT=ROOT/'results/case_studies/candid_ptx/sp27_posthoc_v2';SOUT=ROOT/'results/celeba/sp27_posthoc_v2';SEED=21161103;MODELS=('clip','dinov2','siglip','mae');CMODELS=('rad-dino','dinov2','biomedclip');METHODS=('raw','leace','leace_cc2')

def write_csv(rows,path):
 d=pd.DataFrame(rows);d.insert(0,'analysis_title',TITLE);d.to_csv(path,index=False)

def atomic_json(path,obj):
 temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(dict(obj,analysis_title=TITLE),indent=2,ensure_ascii=False)+'\n');os.replace(temp,path)

def config_for(stage):
 return dict(stage=stage,seed=SEED,replicates=1000,analysis_title=TITLE,plan_sha256=file_sha256(ROOT/'ANALYSIS_PLAN_SP27.md'),git_head='anonymous-release',sources={n:file_sha256(Path(__file__).with_name(n)) for n in ('run_sp27_posthoc_v2_e1.py','sp27_posthoc_v2_metrics.py','sp27_cached_metrics.py')})

def load_candid(model):
 repair=json.loads((COLD/'score_metadata_repair_v1/COMPLETE.json').read_text());lookup={x['source']:x for x in repair['files']};folds=[]
 for f in range(5):
  methods={}
  for method in METHODS:
   source=COLD/f'main_{model}/fold{f}_{method}_SCORES.npz';record=lookup[str(source.relative_to(ROOT))];path=ROOT/record['derived']
   if file_sha256(path)!=record['derived_sha256']:raise RuntimeError('E1_SCORE_HASH')
   with np.load(path,allow_pickle=False) as d:methods[method]={k[5:]:d[k] for k in d.files if k.startswith('test_')}
  folds.append(CandidFold(methods))
 return folds

def flatten(result):
 out={m+'__'+k:v for m,d in result.items() for k,v in d.items()}
 for family in ('linear','MLP','suite'):
  for w in (0,1):k=f'conditional_{family}_W{w}';out['leace_cc2_minus_leace__'+k]=result['leace_cc2'][k]-result['leace'][k]
 for probe in ('linear','MLP'):
  for ij in ('00','11','10','01'):
   k=probe+'_AUC_'+ij
   for a,b in [('leace_cc2','leace'),('leace','raw'),('leace_cc2','raw')]:out[a+'_minus_'+b+'__'+k]=result[a][k]-result[b][k]
 return out

def cirow(model,label,estimate,values):
 a=np.asarray(values,float);v=a[np.isfinite(a)];q=np.quantile(v,[.025,.975]) if len(v) else [np.nan,np.nan]
 return dict(model=model,quantity=label,estimate=float(estimate),ci95_low=q[0],ci95_high=q[1],n_boot_total=len(a),n_boot_valid=len(v),n_boot_undefined=len(a)-len(v))

def candid_worker(model):
 folder=COUT/'E1'/model;folder.mkdir(parents=True,exist_ok=True);cfg=config_for('E1_CANDID_saved_scores');atomic_json(folder/'RUN_CONFIG.json',dict(**cfg,model=model));folds=load_candid(model);point=flatten(evaluate(folds,[np.ones(len(f.w)) for f in folds]));reported=pd.read_csv(COLD/'statistics/CANDID_ESTIMATES_WITH_CI.csv');maximum=0.
 for method in METHODS:
  for _,r in reported[(reported.model==model)&reported.method.eq(method)].iterrows():
   delta=abs(point[method+'__'+r.metric]-r.estimate);maximum=max(maximum,delta)
   if delta>=1e-12:raise RuntimeError('E1_CANDID_POINT_GUARD:'+method+':'+r.metric)
 names=list(point);rng=np.random.default_rng(SEED);mat=[];start=time.monotonic()
 for b in range(1000):
  r=flatten(evaluate(folds,weights_for(folds,rng)));mat.append([r[k] for k in names])
  if (b+1)%100==0:atomic_json(folder/'PROGRESS.json',dict(model=model,completed_replicates=b+1,total_replicates=1000,seconds=time.monotonic()-start));print('E1 candid',model,b+1,flush=True)
 mat=np.asarray(mat);np.savez_compressed(folder/'BOOTSTRAP.npz',analysis_title=np.asarray(TITLE),metrics=np.asarray(names,dtype=str),replicate=np.arange(1000),values=mat);rows=[cirow(model,k,point[k],mat[:,i]) for i,k in enumerate(names)];write_csv(rows,folder/'ESTIMATES_WITH_CI.csv');atomic_json(folder/'COMPLETE.json',dict(model=model,replicates=1000,maximum_point_difference=maximum,seconds=time.monotonic()-start));return model

def finish_candid():
 allrows=pd.concat([pd.read_csv(COUT/'E1'/m/'ESTIMATES_WITH_CI.csv') for m in CMODELS],ignore_index=True);allrows.to_csv(COUT/'E1_ESTIMATES_WITH_CI.csv',index=False)
 strata=allrows[allrows.quantity.str.contains(r'__conditional_(linear|MLP|suite)_W[01]$|__conditional_weight_W[01]$',regex=True)];strata.to_csv(COUT/'E1_1_CONDITIONAL_STRATA.csv',index=False)
 cells=allrows[allrows.quantity.str.contains(r'__(linear|MLP)_(AUC_(00|11|10|01)|tube_shortcut_gain)$',regex=True)];cells.to_csv(COUT/'E1_2_UTILITY_FOUR_CELL.csv',index=False)
 structural=allrows[allrows.quantity.str.contains(r'^(leace|leace_cc2)__linear_(h3_|cross$|shift_sd$)',regex=True)];structural.to_csv(COUT/'E1_3_STRUCTURAL_ERRORS.csv',index=False)
 frame=pd.read_csv(guarded_input(ROOT/'ANALYSIS_PLAN_SP27.md',ROOT/'SP27_FROZEN_CONFIG.json','candid','manifest'));z=frame.tube.to_numpy();w=frame.PTX.to_numpy();refs=[]
 for model in CMODELS:
  folds=load_candid(model)
  for f,fold in enumerate(folds):
   with np.load(COLD/f'main_{model}/fold{f:02d}_split.npz',allow_pickle=False) as d:at=d['A_train']
   for method,group in [('leace','all'),('leace_cc2',0),('leace_cc2',1)]:
    ix=at if group=='all' else at[w[at]==group];n=len(ix);p=float(z[ix].mean());R=float(ndtr(np.sqrt(767/(2*n*p*(1-p)))));s=fold.methods[method]['unconditional_linear' if group=='all' else 'conditional_linear'];mask=np.ones(len(fold.w),bool) if group=='all' else fold.w==group;observed=RankedScore(s[mask]).auc(fold.z[mask]==1,fold.z[mask]==0,np.ones(mask.sum()));refs.append(dict(dataset='CANDID',model=model,fold=f,method=method,W_stratum=group,d=768,n_fit=n,p_fit=p,R_optimal_direction_reference_upper_bound_not_prediction=R,observed_linear_AUROC=observed,attacker='attack_uncond_logistic' if group=='all' else 'attack_cond_logistic'))
 write_csv(refs,COUT/'E1_4_FINITE_SAMPLE_REFERENCE.csv');(COUT/'E1_REPORT.md').write_text('# '+TITLE+'\n\nE1 completed from saved scores only. No models trained. Bootstrap uses the recorded within-fold patient draws with seed21161103 and checks only class counts>=1; pair-screening thresholds are not reapplied.\n\nE1.1 reports both W strata, conditional pair weights and cc2−LEACE differences. E1.2 cell AUCs are weighted across folds by their own positive×negative cell pair counts; contrasts and tube shortcut gains use paired draws. E1.3 uses the exact same struct_pred code path as CelebA H3. E1.4 R is an optimal-direction reference upper bound, not a prediction, and does not account for finite-sample attacker fitting.\n\nNo p-values are reported for E1.1–E1.4. All outputs are post hoc and do not replace SP27 primary results.\n')
 atomic_json(COUT/'E1_COMPLETE.json',dict(models=list(CMODELS),replicates=1000,training=False))

def celeba_worker(task):
 model,pair_id=task;folder=SOUT/'E1_5'/model/f'pair{pair_id:03d}';folder.mkdir(parents=True,exist_ok=True);cfg=config_for('E1_5_CelebA_posthoc_stratified_H4');atomic_json(folder/'RUN_CONFIG.json',dict(**cfg,model=model,pair_id=pair_id));audit=json.loads((SOLD/'INPUT_AUDIT.json').read_text());lookup={(d['model'],d['pair_id'],d['fold']):d for d in audit['files']};folds=[];references=[];elig=pd.read_csv(MAIN/'ELIGIBILITY_TABLE.csv');source_table=pd.read_csv(SOLD/'METHOD_FOLD_RESULTS.csv');qsource=source_table[(source_table.model==model)&source_table.pair_id.eq(pair_id)]
 for f in range(5):
  entry=lookup[model,pair_id,f];path=ROOT/entry['score_path']
  if file_sha256(path)!=entry['score_sha256']:raise RuntimeError('E1_5_SCORE_HASH')
  with np.load(path,allow_pickle=False) as d:
   mask=d['partition']=='test';w=d['W'][mask];z=d['Z'][mask];groups=d['identity_id'][mask];u,inverse=np.unique(groups,return_inverse=True);ranks={m:RankedScore(d[m+'__utility_linear'][mask]) for m in ('leace','leace_cc2')};attack=RankedScore(d['leace_cc2__conditional_linear'][mask])
  cells={(a,b):(w==a)&(z==b) for a in (0,1) for b in (0,1)};folds.append(dict(w=w,z=z,ng=len(u),inverse=inverse,ranks=ranks,cells=cells))
  counts=elig[(elig.pair_id==pair_id)&elig.fold.eq(f)&elig.partition.eq('A_train')].pivot(index='W_value',columns='Z_value',values='n_images').reindex(index=[0,1],columns=[0,1]).to_numpy();base=qsource[(qsource.fold==f)&qsource.method.eq('leace')].iloc[0]
  for method,group in [('leace','all'),('leace_cc2',0),('leace_cc2',1)]:
   cc=counts.sum(axis=0) if group=='all' else counts[group];n=int(cc.sum());p=float(cc[1]/n);R=float(ndtr(np.sqrt(767/(2*n*p*(1-p)))))
   observed=float(base.unconditional_linear_AUROC) if group=='all' else attack.auc(cells[group,1],cells[group,0],np.ones(len(w)))
   references.append(dict(model=model,pair_id=pair_id,fold=f,method=method,W_stratum=group,d=768,n_fit=n,p_fit=p,R=R,observed_linear_AUROC=observed))
 def evaluate_delta(weights):
  ds=[];ws=[];ok100=True;ok1=True
  for fold,q in zip(folds,weights):
   c=np.array([[q[fold['cells'][a,b]].sum() for b in (0,1)] for a in (0,1)]);den=c.sum(axis=1).prod();ok100&=c.min()>=100;ok1&=c.min()>=1;delta=0.
   if c.min()>=1:
    for i,j in ((1,0),(0,1)):
     a=fold['ranks']['leace'].auc(fold['cells'][1,i],fold['cells'][0,j],q);b=fold['ranks']['leace_cc2'].auc(fold['cells'][1,i],fold['cells'][0,j],q);delta+=c[1,i]*c[0,j]/den*(a-b)
   else:delta=np.nan
   ds.append(delta);ws.append(den)
  value=float(np.average(ds,weights=ws)) if ok1 else np.nan
  return value,value if ok100 else np.nan
 point=evaluate_delta([np.ones(len(f['w'])) for f in folds]);rng=np.random.default_rng(SEED);values=[]
 for b in range(1000):
  weights=[np.bincount(rng.integers(0,f['ng'],f['ng']),minlength=f['ng'])[f['inverse']].astype(float) for f in folds];values.append(evaluate_delta(weights))
  if (b+1)%100==0:atomic_json(folder/'PROGRESS.json',dict(completed_replicates=b+1,total_replicates=1000))
 values=np.asarray(values);old=SOLD/model/f'pair{pair_id:03d}';run=json.loads((old/'RUN_CONFIG.json').read_text());idx=run['metric_names'].index('linear_cross');original=[]
 for c in range(10):
  path=old/f'bootstrap_{c:02d}.npz';marker=json.loads(path.with_suffix('.json').read_text())
  if file_sha256(path)!=marker['sha256']:raise RuntimeError('E1_5_ORIGINAL_BOOTSTRAP_HASH')
  with np.load(path,allow_pickle=False) as d:original.append(d['values'][:,2,idx]-d['values'][:,1,idx])
 original=np.concatenate(original)
 if not np.array_equal(np.isfinite(original),np.isfinite(values[:,1])) or (np.isfinite(original).any() and np.max(abs(original[np.isfinite(original)]-values[np.isfinite(original),1]))>=1e-12):raise RuntimeError('E1_5_STRICT_REPLAY_MISMATCH')
 np.savez_compressed(folder/'BOOTSTRAP.npz',analysis_title=np.asarray(TITLE),versions=np.asarray(['computable_ge1','original_ge100']),replicate=np.arange(1000),values=values,point=np.asarray(point));write_csv(references,folder/'FINITE_SAMPLE_REFERENCE.csv');atomic_json(folder/'COMPLETE.json',dict(model=model,pair_id=pair_id,replicates=1000,strict_original_replay_pass=True));return model,pair_id

def p_less(a,point):
 a=np.asarray(a,float);a=a[np.isfinite(a)];return float((1+np.sum(a-point<=point))/(len(a)+1)) if len(a) else np.nan

def finish_celeba():
 pairtable=pd.read_csv(MAIN/'QUALIFIED_PAIRS.csv').sort_values('pair_id').reset_index(drop=True);ids=pairtable.pair_id.to_numpy();targets=pairtable.Z.to_numpy();blocks=np.unique(targets);bands=pd.read_csv(SOLD/'clip/BETA_BANDS.csv');members={g:np.flatnonzero(np.isin(ids,bands[bands.beta_band.eq(g)].pair_id)) for g in ('low','middle','high')};rows=[];ref=[];allboot=[]
 for model in MODELS:
  values=[];points=[]
  for pair in ids:
   folder=SOUT/'E1_5'/model/f'pair{pair:03d}'
   with np.load(folder/'BOOTSTRAP.npz',allow_pickle=False) as d:values.append(d['values']);points.append(d['point'])
   ref.append(pd.read_csv(folder/'FINITE_SAMPLE_REFERENCE.csv'))
  values=np.stack(values,axis=1);points=np.asarray(points);rng=np.random.default_rng(SEED+1);replicates={g:[] for g in members}
  for b in range(1000):
   draw=rng.choice(blocks,len(blocks),replace=True);selected=np.concatenate([np.flatnonzero(targets==z) for z in draw])
   for group,allowed in members.items():
    take=selected[np.isin(selected,allowed)];replicates[group].append(np.mean(values[b,take],axis=0) if len(take) else np.full(2,np.nan))
  old=pd.read_csv(SOLD/model/'AGGREGATE_REPLICATES.csv')
  for group,indices in members.items():
   m=np.asarray(replicates[group]);point=points[indices].mean(axis=0);oldq=old[old.beta_band.eq(group)].sort_values('replicate')['cc2_minus_leace__linear_cross'].to_numpy()
   if not np.array_equal(np.isfinite(oldq),np.isfinite(m[:,1])) or (np.isfinite(oldq).any() and np.max(abs(oldq[np.isfinite(oldq)]-m[np.isfinite(oldq),1]))>=1e-12):raise RuntimeError('E1_5_AGGREGATE_STRICT_REPLAY_MISMATCH')
   for vi,version in enumerate(('computable_ge1','original_ge100')):
    rr=cirow(model,'cc2_minus_LEACE_linear_cross_loss',point[vi],m[:,vi]);rr.update(beta_band=group,version=version,raw_p=p_less(m[:,vi],point[vi]),holm_p=np.nan,n_pairs=len(indices));rows.append(rr)
   allboot.append((model,group,m))
 for version in ('computable_ge1','original_ge100'):
  selected=[r for r in rows if r['version']==version and r['beta_band']=='high'];last=0.
  for rank,r in enumerate(sorted(selected,key=lambda r:r['raw_p'])):last=max(last,min(1.,(4-rank)*r['raw_p']));r['holm_p']=last
 write_csv(rows,SOUT/'E1_5_STRATIFIED_H4.csv');np.savez_compressed(SOUT/'E1_5_BOOTSTRAP.npz',analysis_title=np.asarray(TITLE),models=np.asarray(MODELS),bands=np.asarray(['low','middle','high']),versions=np.asarray(['computable_ge1','original_ge100']),values=np.asarray([np.asarray([next(v for m,g,v in allboot if m==model and g==group) for group in ('low','middle','high')]) for model in MODELS]))
 detail=pd.concat(ref,ignore_index=True);detail.to_csv(SOUT/'E1_4_FINITE_SAMPLE_DETAIL.csv',index=False);merged=detail.merge(bands,on='pair_id',how='left',validate='many_to_one');summary=merged.groupby(['model','beta_band','method','W_stratum'],dropna=False).agg(median_R_optimal_direction_reference_upper_bound_not_prediction=('R','median'),median_observed_linear_AUROC=('observed_linear_AUROC','median'),n_pair_folds=('R','size')).reset_index();summary.insert(0,'analysis_title',TITLE);summary.to_csv(SOUT/'E1_4_FINITE_SAMPLE_TERTILES.csv',index=False)
 (SOUT/'E1_REPORT.md').write_text('# '+TITLE+'\n\n## Deviation from preregistered plan: stratified H4 sensitivity analysis\n\nE1.5 compares computability-only (>=1) bootstrap with the unchanged original>=100 rule, using the identical within-fold identity draws and Z-block draws. Every original-rule per-pair and per-band replicate matches the saved primary bootstrap within1e-12, including the NaN mask. No original file was overwritten.\n\nOnly high-|beta| p-values receive Holm4, separately for each version; low/middle p-values remain unadjusted. Actual centered one-sided p=(1+sum(theta_star-theta_hat<=theta_hat))/(n_valid+1). Intervals use finite draws only; missing required folds/pairs remain NA.\n\nE1.4 R medians are optimal-direction finite-sample reference upper bounds, not predictions. The observed values use saved LEACE attack_uncond-logistic and within-W cc2 attack_cond-logistic scores. No new model was fitted.\n')
 atomic_json(SOUT/'E1_COMPLETE.json',dict(models=list(MODELS),pairs=85,replicates=1000,original_replay_pass=True,training=False))

def run_pool(function,tasks,workers):
 pool=ProcessPoolExecutor(max_workers=workers);futures=[pool.submit(function,t) for t in tasks]
 try:
  for i,f in enumerate(as_completed(futures)):f.result();print('E1 completed tasks',i+1,'/',len(tasks),flush=True)
 except BaseException:
  for p in list((pool._processes or {}).values()):p.terminate()
  pool.shutdown(wait=False,cancel_futures=True);raise
 else:pool.shutdown()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('candid','celeba','all'),default='all');ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();require_frozen_plan(ROOT/'ANALYSIS_PLAN_SP27.md',ROOT/'SP27_FROZEN_CONFIG.json');COUT.mkdir(exist_ok=True);SOUT.mkdir(parents=True,exist_ok=True)
 if a.workers>int(os.environ.get('SLURM_CPUS_PER_TASK',a.workers)):raise RuntimeError('CPU_ALLOCATION_GUARD')
 if a.stage in ('candid','all'):run_pool(candid_worker,CMODELS,min(3,a.workers));finish_candid()
 if a.stage in ('celeba','all'):
  pairs=pd.read_csv(MAIN/'QUALIFIED_PAIRS.csv');run_pool(celeba_worker,[(m,int(p)) for m in MODELS for p in pairs.pair_id],a.workers);finish_celeba()
if __name__=='__main__':
 try:main()
 except Exception as error:
  for out in (COUT,SOUT):out.mkdir(parents=True,exist_ok=True);atomic_json(out/'E1_STOP.json',dict(error=repr(error),traceback=traceback.format_exc()))
  raise
