"""POST HOC – not preregistered. Paired patient bootstrap for V2.1 E2."""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from sp27_posthoc_v2_metrics import CandidFold,evaluate,weights_for,TITLE
from run_sp27_posthoc_v2_e1 import load_candid,cirow,p_less,write_csv,atomic_json
from sp27_plan_guard import file_sha256
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/case_studies/candid_ptx/sp27_posthoc_v2';MODELS=('rad-dino','dinov2','biomedclip');SEED=21161103

def load_folds(model):
 old=load_candid(model);reader=OUT/'E2_v21/readers'/model
 complete=json.loads((reader/'COMPLETE.json').read_text())
 if complete['rows']!=30:raise RuntimeError('E2_READER_COVERAGE')
 folds=[];sources=[]
 for fold,base in enumerate(old):
  methods={'S-out::'+name:data for name,data in base.methods.items()}
  for scenario,names in [('S-out',('splince','weber')),('S-rel',('leace','leace_cc2','splince','weber'))]:
   for method in names:
    path=reader/f'{scenario}_fold{fold}_{method}_SCORES.npz';meta=json.loads(path.with_suffix('.json').read_text())
    if file_sha256(path)!=meta['score_sha256']:raise RuntimeError('E2_SCORE_HASH')
    with np.load(path,allow_pickle=False) as d:methods[scenario+'::'+method]={k[5:]:d[k] for k in d.files if k.startswith('test_')}
    sources.append(dict(path=str(path.relative_to(ROOT)),sha256=meta['score_sha256']))
  folds.append(CandidFold(methods))
 return folds,sources

def flatten(result):
 output={name+'@@'+k:v for name,d in result.items() for k,v in d.items()}
 comparisons=[]
 for scenario in ('S-out','S-rel'):
  for m in ('splince','weber','leace_cc2'):comparisons.append((scenario+'::'+m,scenario+'::leace'))
 for m in ('leace','leace_cc2'):comparisons.append(('S-rel::'+m,'S-out::'+m))
 for a,b in comparisons:
  for k in result[a]:output[a+'_minus_'+b+'@@'+k]=result[a][k]-result[b][k]
 return output

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS,required=True);args=ap.parse_args();model=args.model
 if (OUT/'E2_V21_STOP.json').exists():raise RuntimeError('E2_V21_STOP_REQUIRES_REVIEW')
 folder=OUT/'E2_v21/statistics'/model;folder.mkdir(parents=True,exist_ok=True);head='anonymous-release';folds,sources=load_folds(model);point=flatten(evaluate(folds,[np.ones(len(f.w)) for f in folds],raw_name='S-out::raw'));names=list(point);rng=np.random.default_rng(SEED);values=[]
 run=dict(analysis_title=TITLE,model=model,git_head=head,bootstrap_seed=SEED,n_boot_total=1000,sources=sources,amendment_sha256=file_sha256(OUT/'AMENDMENT_V2_1.md'),predictions_sha256=file_sha256(OUT/'PREDICTIONS_V2.md'),computability_only=True,source_sha256=file_sha256(Path(__file__)))
 atomic_json(folder/'RUN_CONFIG.json',run);start=time.monotonic()
 for rep in range(1000):
  result=flatten(evaluate(folds,weights_for(folds,rng),raw_name='S-out::raw'));values.append([result[n] for n in names])
  if (rep+1)%50==0:atomic_json(folder/'PROGRESS.json',dict(analysis_title=TITLE,model=model,completed=rep+1,total=1000,seconds=time.monotonic()-start));print('E2 bootstrap',model,rep+1,flush=True)
 values=np.asarray(values);np.savez_compressed(folder/'BOOTSTRAP.npz',analysis_title=TITLE,metrics=np.asarray(names,dtype=str),replicate=np.arange(1000),values=values,point=np.asarray([point[n] for n in names]));rows=[];comparisons=[];pb=[]
 for i,name in enumerate(names):
  label,metric=name.split('@@');r=cirow(model,name,point[name],values[:,i]);r['metric']=metric
  if '_minus_' in label:
   a,b=label.split('_minus_');sa,ma=a.split('::');sb,mb=b.split('::');r.update(scenario_a=sa,method_a=ma,scenario_b=sb,method_b=mb);comparisons.append(r)
   if label=='S-rel::leace_cc2_minus_S-rel::leace' and metric in ('conditional_linear_W0','conditional_linear_W1'):
    p=dict(r,prediction='P-b',W_stratum=int(metric[-1]),p_formula='(1 + sum(theta_star - theta_hat <= theta_hat)) / (n_valid + 1)',raw_p=p_less(values[:,i],point[name]),multiplicity='none; exploratory');pb.append(p)
  else:
   scenario,method=label.split('::');r.update(scenario=scenario,method=method);rows.append(r)
 write_csv(rows,folder/'CANDID_ESTIMATES_WITH_CI.csv');write_csv(comparisons,folder/'COMPARISONS_WITH_CI.csv');write_csv(pb,folder/'P_B_EXPLORATORY.csv')
 d=pd.read_csv(folder/'CANDID_ESTIMATES_WITH_CI.csv')
 d[d.metric.str.match(r'conditional_(linear|MLP|suite)_W[01]$|conditional_weight_W[01]$')].to_csv(folder/'STRATIFIED_CONDITIONAL_AUROC.csv',index=False)
 d[d.metric.str.match(r'(linear|MLP)_(AUC_(00|11|10|01)|tube_shortcut_gain)$')].to_csv(folder/'UTILITY_FOUR_CELL.csv',index=False)
 d[d.method.isin(['leace','leace_cc2']) & d.metric.str.match(r'linear_(h3_|cross$|shift_sd$)')].to_csv(folder/'STRUCTURAL_ERRORS.csv',index=False)
 atomic_json(folder/'COMPLETE.json',dict(analysis_title=TITLE,model=model,replicates=1000,seconds=time.monotonic()-start,n_rows=len(rows),n_comparisons=len(comparisons),git_head=head))
 (folder/'REPORT.md').write_text('# '+TITLE+'\n\nPaired within-fold patient bootstrap (1000, seed 21161103); all methods/scenarios share draws. Only class counts >=1 are required in bootstrap. Original SP27 score files supply S-out raw/LEACE/cc2. Other readers use frozen settings and amendment 2.1 cached erasers. All AUROCs and T5 terms are computed within folds and pair-weighted across folds. Only P-b has exploratory centered one-sided p-values, with no Holm; other comparisons have estimates, two-sided 95% percentile intervals and valid counts. Weber requires true Z on application.\n\nProvenance: amendment SHA-256 '+run['amendment_sha256']+'; unchanged prediction SHA-256 '+run['predictions_sha256']+'; code commit '+head+'.\n')
if __name__=='__main__':
 try:main()
 except Exception as e:
  (OUT/'E2_V21_STOP.json').write_text(json.dumps(dict(analysis_title=TITLE,stage='E2 statistics',error=repr(e),time=time.time()),indent=2)+'\n');raise
