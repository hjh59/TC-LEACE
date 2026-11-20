"""POST HOC – not preregistered. V2.2 paired comparisons and top-five tables."""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from sp27_posthoc_v2_metrics import CandidFold,evaluate,weights_for,TITLE
from run_sp27_posthoc_v2_e2_statistics import load_folds as load_previous
from run_sp27_posthoc_v2_e1 import cirow,p_less,write_csv,atomic_json
from sp27_plan_guard import file_sha256
from sp27_v22_scope import PRED_SHA,BIASED
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/case_studies/candid_ptx/sp27_posthoc_v2_2';PREV=ROOT/'results/case_studies/candid_ptx/sp27_posthoc_v2';MODELS=('rad-dino','dinov2','biomedclip');SEED=21161103

def load_folds(model):
 previous,sources=load_previous(model);reader=OUT/'E2_v22/readers'/model
 if json.loads((reader/'COMPLETE.json').read_text())['rows']!=20:raise RuntimeError('READER_COVERAGE')
 folds=[]
 for fold,old in enumerate(previous):
  methods=dict(old.methods)
  for method in ('leace','leace_cc2','splince','weber'):
   path=reader/f'S-rel2_fold{fold}_{method}_SCORES.npz';meta=json.loads(path.with_suffix('.json').read_text())
   if file_sha256(path)!=meta['score_sha256']:raise RuntimeError('SCORE_HASH')
   with np.load(path,allow_pickle=False) as d:methods['S-rel2::'+method]={k[5:]:d[k] for k in d.files if k.startswith('test_')}
   sources.append(dict(path=str(path.relative_to(ROOT)),sha256=meta['score_sha256']))
  folds.append(CandidFold(methods))
 return folds,sources

def flatten(result):
 output={name+'@@'+k:v for name,d in result.items() for k,v in d.items()};pairs=[]
 for scenario in ('S-out','S-rel','S-rel2'):
  for method in ('leace_cc2','splince','weber'):pairs.append((scenario+'::'+method,scenario+'::leace'))
 for method in ('leace','leace_cc2','splince','weber'):
  for baseline in ('S-out','S-rel'):pairs.append(('S-rel2::'+method,baseline+'::'+method))
 for a,b in pairs:
  for k in result[a]:output[a+'_minus_'+b+'@@'+k]=result[a][k]-result[b][k]
 return output

def status(s):return BIASED if s=='S-rel' else 'unchanged S-out' if s=='S-out' else 'S-rel2 excludes B_train and B_val'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS,required=True);model=ap.parse_args().model
 if (OUT/'E2_V22_STOP.json').exists():raise RuntimeError('STOP_REQUIRES_REVIEW')
 if file_sha256(OUT/'PREDICTIONS_V2_2.md')!=PRED_SHA:raise RuntimeError('PREDICTIONS_CHANGED')
 folder=OUT/'E2_v22/statistics'/model;folder.mkdir(parents=True,exist_ok=True);head='anonymous-release';folds,sources=load_folds(model);point=flatten(evaluate(folds,[np.ones(len(f.w)) for f in folds],raw_name='S-out::raw'));names=list(point);rng=np.random.default_rng(SEED);values=[]
 atomic_json(folder/'RUN_CONFIG.json',dict(model=model,git_head=head,bootstrap_seed=SEED,n_boot_total=1000,sources=sources,amendment_sha256=file_sha256(OUT/'AMENDMENT_V2_2.md'),predictions_sha256=PRED_SHA,computability_only=True,source_sha256=file_sha256(Path(__file__))))
 start=time.monotonic()
 for rep in range(1000):
  result=flatten(evaluate(folds,weights_for(folds,rng),raw_name='S-out::raw'));values.append([result[n] for n in names])
  if (rep+1)%50==0:atomic_json(folder/'PROGRESS.json',dict(model=model,completed=rep+1,total=1000,seconds=time.monotonic()-start));print('V2.2 bootstrap',model,rep+1,flush=True)
 values=np.asarray(values)
 # Exact reuse audit: all original scenario estimates/draws remain unchanged.
 with np.load(PREV/'E2_v21/statistics'/model/'BOOTSTRAP.npz',allow_pickle=False) as old:
  old_names=old['metrics'].astype(str);new_index={n:i for i,n in enumerate(names)}
  for i,name in enumerate(old_names):
   if '_minus_' in name:continue
   j=new_index[name];a=old['values'][:,i];b=values[:,j]
   if not np.array_equal(np.isfinite(a),np.isfinite(b)) or (np.isfinite(a).any() and np.max(abs(a[np.isfinite(a)]-b[np.isfinite(a)]))>=1e-12):raise RuntimeError('PREVIOUS_SCENARIO_BOOTSTRAP_MISMATCH')
   if np.isfinite(old['point'][i]) and abs(old['point'][i]-point[name])>=1e-12:raise RuntimeError('PREVIOUS_SCENARIO_POINT_MISMATCH')
 np.savez_compressed(folder/'BOOTSTRAP.npz',analysis_title=TITLE,metrics=np.asarray(names,dtype=str),replicate=np.arange(1000),values=values,point=np.asarray([point[n] for n in names]));rows=[];comparisons=[];pb=[]
 for i,name in enumerate(names):
  label,metric=name.split('@@');r=cirow(model,name,point[name],values[:,i]);r['metric']=metric
  if '_minus_' in label:
   a,b=label.split('_minus_');sa,ma=a.split('::');sb,mb=b.split('::');r.update(scenario_a=sa,method_a=ma,scenario_b=sb,method_b=mb,design_status_a=status(sa),design_status_b=status(sb));comparisons.append(r)
   if label=='S-rel2::leace_cc2_minus_S-rel2::leace' and metric in ('conditional_linear_W0','conditional_linear_W1'):
    pb.append(dict(r,prediction='P-b',W_stratum=int(metric[-1]),p_formula='(1 + sum(theta_star - theta_hat <= theta_hat)) / (n_valid + 1)',raw_p=p_less(values[:,i],point[name]),multiplicity='none; exploratory'))
  else:
   scenario,method=label.split('::');r.update(scenario=scenario,method=method,design_status=status(scenario));rows.append(r)
 write_csv(rows,folder/'CANDID_ESTIMATES_WITH_CI.csv');write_csv(comparisons,folder/'COMPARISONS_WITH_CI.csv');write_csv(pb,folder/'P_B_EXPLORATORY.csv');d=pd.DataFrame(rows);cmp=pd.DataFrame(comparisons)
 for name,mask in [('STRATIFIED_CONDITIONAL_AUROC',d.metric.str.match(r'conditional_(linear|MLP|suite)_W[01]$|conditional_weight_W[01]$')),('UTILITY_FOUR_CELL',d.metric.str.match(r'(linear|MLP)_(AUC_(00|11|10|01)|tube_shortcut_gain)$')),('STRUCTURAL_ERRORS',d.method.isin(['leace','leace_cc2'])&d.metric.str.match(r'linear_(h3_|cross$|shift_sd$)'))]:write_csv(d[mask].to_dict('records'),folder/(name+'.csv'))
 top=[];linear_top=[]
 for scenario in ('S-out','S-rel','S-rel2'):
  q=cmp[(cmp.scenario_a==scenario)&cmp.method_a.eq('splince')&cmp.method_b.eq('leace')].copy();q['abs_difference']=q.estimate.abs();q=q[np.isfinite(q.estimate)].sort_values(['abs_difference','metric'],ascending=[False,True]);linear=q[q.metric.str.match(r'(conditional_linear(_W[01])?|unconditional_linear|linear_(within|cross|total|AUC_(00|01|10|11)))$')]
  for destination,selected,label in [(top,q.head(5),'all reported finite metrics'),(linear_top,linear.head(5),'linear readers only; P-a')]:
   for rank,(_,r) in enumerate(selected.iterrows(),1):
    a=d[(d.scenario==scenario)&d.method.eq('splince')&d.metric.eq(r.metric)].iloc[0];b=d[(d.scenario==scenario)&d.method.eq('leace')&d.metric.eq(r.metric)].iloc[0];row=r.to_dict();row.update(rank=rank,ranking_scope=label,scenario=scenario,splince_estimate=a.estimate,splince_ci95_low=a.ci95_low,splince_ci95_high=a.ci95_high,leace_estimate=b.estimate,leace_ci95_low=b.ci95_low,leace_ci95_high=b.ci95_high);destination.append(row)
 write_csv(top,folder/'SPLINCE_LEACE_TOP5.csv');write_csv(linear_top,folder/'SPLINCE_LEACE_TOP5_LINEAR.csv')
 atomic_json(folder/'COMPLETE.json',dict(model=model,replicates=1000,seconds=time.monotonic()-start,n_estimates=len(rows),n_comparisons=len(comparisons),top5_rows=len(top),original_scenarios_bootstrap_replay_pass=True,git_head=head))
 (folder/'REPORT.md').write_text('# '+TITLE+'\n\nV2.2: fit excludes B_train and B_val. Shared patient bootstrap indices (1000, seed21161103); no pair selection thresholds reapplied in bootstrap. All original S-out/S-rel point estimates and base-metric bootstrap arrays reproduced within1e-12, including NaN masks. S-rel is labelled: '+BIASED+'.\n\nOnly S-rel2 P-b has exploratory centered one-sided p-values, no Holm. TOP5 ranks all finite reported metrics by absolute point difference; TOP5_LINEAR separately ranks linear-reader metrics for P-a. Individual SPLINCE/LEACE intervals and paired difference intervals are all included.\n')
if __name__=='__main__':
 try:main()
 except Exception as error:
  (OUT/'E2_V22_STOP.json').write_text(json.dumps(dict(analysis_title=TITLE,stage='V2.2 statistics',error=repr(error),time=time.time()),indent=2)+'\n');raise
