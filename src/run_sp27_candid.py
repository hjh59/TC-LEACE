"""SP27 guarded CANDID null calibration and three-method main analysis.

No T13 results are inputs. Freeze verification precedes all real-label reads.
Main conditional metrics additionally require completed RAD-DINO null calibration.
"""
import argparse,json,hashlib,os,subprocess,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score
from sp27_plan_guard import guarded_input,require_frozen_plan,file_sha256
from sp27_core import identity_split,fit_leace64,transform_leace64,conditional_auc,decompose_fold,logistic64,validate_file
from sp27_mlp import fit_mlp,mlp_predict
from sp27_attack_selection import select_attack
from sp27_eligibility import eligibility
from leace_cc2 import fit_cc2,transform_cc2
from run_sp27_s3_null import permute_in_B
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/case_studies/candid_ptx/sp27';PLAN=ROOT/'ANALYSIS_PLAN_SP27.md';CONFIG=ROOT/'SP27_FROZEN_CONFIG.json';SEED=20261103
METHODS=('raw','leace','leace_cc2');MODELS=('rad-dino','dinov2','biomedclip');ACFG=dict(hidden=(256,64),drop=.2,lr=.001,wd=.0001)

def atomic_json(path,obj):
 tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,path)

def mlp_ensemble(ff,z,bt,bv,seed):
 yy=torch.as_tensor(z,device=ff.device,dtype=torch.float32);pred=[];epochs=[]
 for rep in range(3):
  model,_,_,ep=fit_mlp(ff[bt].float(),yy[bt],ff[bv].float(),yy[bv],ACFG['hidden'],ACFG['drop'],ACFG['lr'],ACFG['wd'],seed+rep,epochs=200,patience=15,return_best_epoch=True)
  pred.append(mlp_predict(model,ff.float()).astype(np.float64));epochs.append(ep);del model
 return np.mean(pred,axis=0),epochs

def scale(ff,train):return (ff-ff[train].mean(0))/ff[train].std(0,unbiased=False).clamp_min(1e-8)

def transformed(x,z,w,at,method):
 base=(x-x[at].mean(0))/x[at].std(0).clip(1e-8);rank=0
 if method=='raw':xx=base
 elif method=='leace':xx=transform_leace64(base,fit_leace64(base[at],z[at]));rank=1
 else:
  fit=fit_cc2(base[at],z[at],w[at]);xx=transform_cc2(base,fit);rank=fit['rank']
 checks=[];relative=[]
 for ix in ([at] if method=='leace' else [at[w[at]==v] for v in (0,1)] if method=='leace_cc2' else []):
  cv=lambda v:np.linalg.norm((v[ix]-v[ix].mean(0)).T@(z[ix]-z[ix].mean())/len(ix))
  checks.append(float(cv(xx)));relative.append(float(cv(xx)/max(cv(base),np.finfo(float).eps)))
 if max(checks,default=0.)>=1e-8:raise RuntimeError('COVARIANCE_GATE')
 return xx,rank,max(checks,default=0.),max(relative,default=0.)

def save_npz(path,arrays):
 tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);os.replace(tmp,path)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS,required=True);ap.add_argument('--mode',choices=('calibrate','main'),required=True);a=ap.parse_args()
 freeze=require_frozen_plan(PLAN,CONFIG)
 if a.mode=='calibrate' and a.model!='rad-dino':raise RuntimeError('ONLY_PRIMARY_NULL_CALIBRATION')
 if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
 torch.set_num_threads(4);dev=torch.device('cuda:0');OUT.mkdir(parents=True,exist_ok=True)
 threshold_path=OUT/'CANDID_DELTA_T.json'
 if a.mode=='main':
  if not threshold_path.is_file():raise PermissionError('CANDID_NULL_CALIBRATION_REQUIRED')
  threshold=json.loads(threshold_path.read_text());evidence=OUT/'null_rad-dino/CONDITIONAL_NULL.csv'
  if threshold['plan_sha256']!=freeze['analysis_plan_sha256'] or file_sha256(evidence)!=threshold['null_csv_sha256'] or threshold['permutations']!=20:raise PermissionError('CANDID_THRESHOLD_EVIDENCE_MISMATCH')
 # Guard first; no old result/config loaders or T13 imports.
 manifest=guarded_input(PLAN,CONFIG,'candid','manifest');feature=guarded_input(PLAN,CONFIG,'candid',a.model)
 m=pd.read_csv(manifest);required={'patient_group','fold','tube','PTX'}
 if not required.issubset(m):raise RuntimeError('MANIFEST_SCHEMA')
 g=m.patient_group.astype(str).to_numpy();outer=m.fold.to_numpy();z=m.tube.to_numpy();w=m.PTX.to_numpy();ids=m['dicom_id'].astype(str).to_numpy() if 'dicom_id' in m else np.asarray([f'candid_row{i}' for i in range(len(m))]);x=np.asarray(np.load(feature,mmap_mode='r'),np.float64)
 if len(x)!=len(m) or not np.isfinite(x).all() or set(np.unique(z))!={0,1} or set(np.unique(w))!={0,1} or set(np.unique(outer))!=set(range(5)):raise RuntimeError('INPUT_ALIGNMENT_OR_CLASSES')
 folder=OUT/(('null_' if a.mode=='calibrate' else 'main_')+a.model);folder.mkdir(exist_ok=True);head='anonymous-release'
 source_names=('run_sp27_candid.py','sp27_core.py','sp27_mlp.py','sp27_plan_guard.py','sp27_attack_selection.py','sp27_eligibility.py','leace_cc2.py','run_sp27_s3_null.py')
 run=dict(model=a.model,mode=a.mode,git_head=head,plan_sha256=freeze['analysis_plan_sha256'],frozen_config_sha256=file_sha256(CONFIG),manifest_sha256=file_sha256(manifest),feature_sha256=file_sha256(feature),split_config='A80B20',seed=SEED,target_Z='tube',retained_W='PTX',attacker_config=ACFG,source_hashes={n:file_sha256(Path(__file__).with_name(n)) for n in source_names})
 runhash=hashlib.sha256(json.dumps({k:v for k,v in run.items() if k!='git_head'},sort_keys=True).encode()).hexdigest();atomic_json(folder/'RUN_CONFIG.json',dict(**run,config_hash=runhash));allparts=[];allchecks=[]
 # Count-only preflight for ALL folds before any probe fit or metric evaluation.
 for f in range(5):
  tr=np.flatnonzero(outer!=f);te=np.flatnonzero(outer==f);parts,guard=identity_split(tr,te,g,f);sf=folder/f'fold{f:02d}_split.npz';save_npz(sf,parts);cells={k:np.array([[np.sum((w[ix]==wv)&(z[ix]==zv)) for zv in (0,1)] for wv in (0,1)]) for k,ix in parts.items() if k not in ('A','B')};support=eligibility(cells,'candid');allparts.append((parts,support,file_sha256(sf)))
  for role,counts in cells.items():
   ix=parts[role];row=dict(model=a.model,fold=f,partition=role,split_sha256=file_sha256(sf),split_config='A80B20',conditional_eligible=support['conditional_eligible'],T5_eligible=support['T5_eligible'],stop_required=support['stop_required'],reason=';'.join(support['partition_reasons'][role]))
   for wv in (0,1):
    for zv in (0,1):row[f'n_W{wv}Z{zv}']=int(counts[wv,zv]);row[f'identities_W{wv}Z{zv}']=len(np.unique(g[ix[(w[ix]==wv)&(z[ix]==zv)]]))
   allchecks.append(row)
  pd.DataFrame(allchecks).to_csv(folder/'ELIGIBILITY_TABLE.csv',index=False)
  if support['stop_required']:raise RuntimeError('CANDID_A_TRAIN_CELL_LT50')
  for role,label in [('A_train',w),('A_val',w),('B_train',z),('B_val',z)]:
   if len(np.unique(label[parts[role]]))!=2:raise RuntimeError('TRAIN_OR_VALIDATION_SINGLE_CLASS')
 rows=[]
 for f,(parts,support,splithash) in enumerate(allparts):
  at,av,bt,bv,te=[parts[k] for k in ('A_train','A_val','B_train','B_val','test')];raw_utility={}
  if a.mode=='calibrate':
   xx,rank,cov,rel=transformed(x,z,w,at,'leace');ff=scale(torch.as_tensor(np.column_stack((xx,w)),device=dev,dtype=torch.float64),bt)
   for rep in range(20):
    path=folder/f'fold{f}_perm{rep:02d}.npz';marker=path.with_suffix('.json');key=hashlib.sha256(f'{runhash}:{splithash}:{f}:{rep}'.encode()).hexdigest();shapes={f'{role}_{k}':(len(parts[role]),) for role in ('A_val','B_val','test') for k in ('sample_id','identity_id','W','Z','score','fold','partition')}
    done=json.loads(marker.read_text()) if marker.exists() else {}
    if done.get('key')!=key or not validate_file(path,shapes=shapes,checksum=done.get('npz_sha256')):
     seed=SEED+800000+100*f+rep;zp=permute_in_B(z,w,parts['B'],seed);yy=torch.as_tensor(zp,device=dev,dtype=torch.float64);c,b=logistic64(ff[bt],yy[bt]);linear=(ff@c+b).cpu().numpy();mlp,epochs=mlp_ensemble(ff,zp,bt,bv,seed+20);choice=select_attack(zp[bv],w[bv],linear[bv],mlp[bv],True);score=mlp if choice['chosen']=='MLP' else linear;stat=conditional_auc(z[te],w[te],score[te],min_class=20)
     result=dict(model=a.model,fold=f,permutation=rep,seed=seed,cond_auc=stat['auc'],conditional_pairs=stat['pair_count'],eligible=stat['eligible'],split_config='A80B20',split_sha256=splithash,config_hash=runhash,covariance_error=cov,selection=choice,epochs=epochs)
     arrays={}
     for role in ('A_val','B_val','test'):
      ix=parts[role]
      for k,v in [('sample_id',ids[ix]),('identity_id',g[ix]),('W',w[ix]),('Z',zp[ix] if role=='B_val' else z[ix]),('score',score[ix]),('fold',np.full(len(ix),f)),('partition',np.full(len(ix),role))]:arrays[f'{role}_{k}']=v
     save_npz(path,arrays);done=dict(key=key,npz_sha256=file_sha256(path),result=result);atomic_json(marker,done)
    rows.append(done['result']);pd.DataFrame(rows).to_csv(folder/'NULL_FOLD_RESULTS.csv',index=False);print('null',f,rep,'complete',flush=True)
  else:
   foldcsv=folder/f'fold{f}_RESULTS.csv';foldmarker=folder/f'fold{f}_COMPLETE.json';foldkey=hashlib.sha256(f'{runhash}:{splithash}:{f}'.encode()).hexdigest();done=json.loads(foldmarker.read_text()) if foldmarker.exists() else {};valid=done.get('key')==foldkey and validate_file(foldcsv,expected_rows=3,checksum=done.get('csv_sha256'))
   scorekeys=('sample_id','identity_id','W','Z','fold','partition','utility_linear','utility_MLP','conditional_linear','conditional_MLP','conditional_suite','unconditional_linear','unconditional_MLP','unconditional_suite')
   shapes={f'{role}_{k}':(len(parts[role]),) for role in ('A_val','B_val','test') for k in scorekeys}
   if valid:
    for method in METHODS:
     valid=valid and validate_file(folder/f'fold{f}_{method}_SCORES.npz',shapes=shapes,checksum=done.get('scores',{}).get(method))
   if valid:
    rows.extend(pd.read_csv(foldcsv).to_dict('records'));print('restored fold',f,flush=True);continue
   fold_start=len(rows)
   for mi,method in enumerate(METHODS):
    start=time.monotonic();seed=SEED+100000*MODELS.index(a.model)+1000*f+100*mi;xx,rank,cov,rel=transformed(x,z,w,at,method);ff=torch.as_tensor(xx,device=dev,dtype=torch.float64);wy=torch.as_tensor(w,device=dev,dtype=torch.float32);zy=torch.as_tensor(z,device=dev,dtype=torch.float64);u=scale(ff,at);scores={};row=dict(model=a.model,W='PTX',Z='tube',method=method,fold=f,split_config='A80B20',split_sha256=splithash,seed=seed,config_hash=runhash,git_head=head,erasure_rank=rank,covariance_error=cov,relative_covariance_error=rel,beta=float(w[te][z[te]==1].mean()-w[te][z[te]==0].mean()),n_test=len(te),target_pairs=int(z[te].sum())*int(len(te)-z[te].sum()))
    for wv in (0,1):
     for zv in (0,1):row[f'n_W{wv}Z{zv}']=int(np.sum((w[te]==wv)&(z[te]==zv)))
    best=None
    for C in (.1,1.,10.):
     c,b=logistic64(u[at],wy[at],C=C);pred=(u@c+b).cpu().numpy();quality=roc_auc_score(w[av],pred[av])
     if best is None or quality>best[0]:best=(quality,pred,C)
    scores['utility_linear']=best[1];row['utility_linear_C']=best[2];best=None
    for ci,(drop,lr) in enumerate(((0.,.001),(.1,.0003))):
     model,quality,_,ep=fit_mlp(u[at].float(),wy[at],u[av].float(),wy[av],(256,),drop,lr,.0001,seed+ci,epochs=120,patience=12,return_best_epoch=True);pred=mlp_predict(model,u.float()).astype(np.float64)
     if best is None or quality>best[0]:best=(quality,pred,ci,ep)
     del model
    scores['utility_MLP']=best[1];row['utility_MLP_config']=best[2];row['utility_MLP_epoch']=best[3];c,b=logistic64(u[at],zy[at]);row['A_insample_linear_AUROC']=float(roc_auc_score(z[at],(u[at]@c+b).cpu().numpy()))
    for conditional in (False,True):
     af=scale(torch.cat((ff,wy[:,None].double()),1) if conditional else ff,bt);c,b=logistic64(af[bt],zy[bt]);linear=(af@c+b).cpu().numpy();mlp,epochs=mlp_ensemble(af,z,bt,bv,seed+(30 if conditional else 20));selection=select_attack(z[bv],w[bv],linear[bv],mlp[bv],conditional);prefix='conditional' if conditional else 'unconditional';row[prefix+'_selection']=json.dumps(selection);row[prefix+'_MLP_epochs']=json.dumps(epochs)
     for family,pred in [('linear',linear),('MLP',mlp),('suite',mlp if selection['chosen']=='MLP' else linear)]:
      scores[prefix+'_'+family]=pred
      if conditional:
       stat=conditional_auc(z[te],w[te],pred[te],min_class=20);row[prefix+'_'+family+'_AUROC']=stat['auc'];row['conditional_pairs']=stat['pair_count'];row['conditional_eligible']=stat['eligible']
      else:row[prefix+'_'+family+'_AUROC']=float(roc_auc_score(z[te],pred[te]))
    row['impute_true']=float(roc_auc_score(z[te],w[te]))
    for probe in ('linear','MLP'):
     pred=scores['utility_'+probe][te]
     if method=='raw':raw_utility[probe]=pred.copy()
     row['impute_pred_'+probe]=float(roc_auc_score(z[te],pred));row['T5_eligible']=support['T5_eligible']
     if support['T5_eligible']:
      dec=decompose_fold(w[te],z[te],raw_utility[probe],pred);row['utility_pairs']=dec['pair_count']
      for k in ('within_term','cross_term','total_damage','identity_error'):row[probe+'_'+k]=dec[k]
      for i,j in ((0,0),(1,1),(0,1),(1,0)):
       row[f'{probe}_w_{i}{j}']=dec['raw']['cells'][(i,j)]['weight']
       for typ in ('raw','erased'):row[f'{probe}_AUC_{i}{j}_{typ}']=dec[typ]['cells'][(i,j)]['auc']
    arrays={}
    for role in ('A_val','B_val','test'):
     ix=parts[role]
     for k,v in dict(sample_id=ids[ix],identity_id=g[ix],W=w[ix],Z=z[ix],fold=np.full(len(ix),f),partition=np.full(len(ix),role),**{k:v[ix] for k,v in scores.items()}).items():arrays[f'{role}_{k}']=v
    scorepath=folder/f'fold{f}_{method}_SCORES.npz';save_npz(scorepath,arrays);row['scores_sha256']=file_sha256(scorepath);row['seconds']=time.monotonic()-start;rows.append(row);pd.DataFrame(rows).to_csv(folder/'FOLD_RESULTS.csv',index=False);print('main',a.model,f,method,'complete',flush=True)
  if a.mode=='main':
   foldrows=rows[fold_start:];pd.DataFrame(foldrows).to_csv(foldcsv,index=False);atomic_json(foldmarker,dict(key=foldkey,csv_sha256=file_sha256(foldcsv),scores={method:file_sha256(folder/f'fold{f}_{method}_SCORES.npz') for method in METHODS}))
 if a.mode=='calibrate':
  df=pd.DataFrame(rows);summary=[]
  if len(df)!=100 or df.duplicated(['fold','permutation']).any():raise RuntimeError('NULL_COVERAGE')
  for rep,q in df.groupby('permutation'):
   eligible=q[q.eligible]
   if eligible.empty:raise RuntimeError('NO_ESTIMABLE_NULL_FOLDS')
   summary.append(dict(permutation=int(rep),cond_auc=float(np.average(eligible.cond_auc,weights=eligible.conditional_pairs)),folds_used=len(eligible)))
  path=folder/'CONDITIONAL_NULL.csv';pd.DataFrame(summary).to_csv(path,index=False);values=np.array([r['cond_auc'] for r in summary]);record=dict(plan_sha256=freeze['analysis_plan_sha256'],config_hash=runhash,permutations=20,median=float(np.median(values)),q975=float(np.quantile(values,.975)),delta_T=float(max(.02,np.quantile(values,.975)-.5)),null_csv_sha256=file_sha256(path))
  if threshold_path.exists() and json.loads(threshold_path.read_text())!=record:raise RuntimeError('CANDID_THRESHOLD_IMMUTABLE')
  atomic_json(threshold_path,record)
 if a.mode=='main':pd.DataFrame(rows).to_csv(folder/'FOLD_RESULTS.csv',index=False)
 atomic_json(folder/'COMPLETE.json',dict(config_hash=runhash,rows=len(rows),expected_rows=100 if a.mode=='calibrate' else 15,git_head=head,mode=a.mode))
if __name__=='__main__':
 try:main()
 except Exception as e:
  print('SP27_STOP',type(e).__name__,str(e),flush=True);raise
