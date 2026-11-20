"""SP27 frozen CelebA80 closed-form main runner. No legacy output inputs."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from run_celeba_preexperiment import BASE,DATA,MANIFEST,pre_labels
from sp27_core import fit_leace64,transform_leace64,conditional_auc,decompose_fold,logistic64,validate_file,validation_auc
from sp27_mlp import fit_mlp,mlp_predict
from sp27_attack_selection import select_attack
from leace_cc2 import fit_cc2,transform_cc2
from t20_split_guard import sha256,validate_split
from sp27_plan_guard import require_frozen_plan,guarded_input
from sp27_core import identity_split
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'ANALYSIS_PLAN_SP27.md';FROZEN=ROOT/'SP27_FROZEN_CONFIG.json'
METHODS=('raw','leace','leace_cc2');UTILITY_GRID=(((256,),0.,1e-3,1e-4),((256,),.1,3e-4,1e-4))

def auc(y,s):return float(roc_auc_score(y,s)) if np.unique(y).size==2 else np.nan

def frozen_attack_config(table,target,fold):
 q=table[(table.target_Z==target)&(table.outer_fold==fold)&(table.method=='raw')&(table.attacker=='MLP_tuned')]
 if len(q)!=3 or q.config_json.nunique()!=1:raise RuntimeError('T2_CONFIG_AMBIGUOUS')
 return json.loads(q.iloc[0].config_json)

def mlp3(train,y,val,yv,evaluate,cfg,seed):
 out=[];validation=[];epochs=[]
 for rep in range(3):
  fit,_,_,ep=fit_mlp(train.float(),y,val.float(),yv,tuple(cfg['hidden']),cfg['drop'],cfg['lr'],cfg['wd'],seed+rep,epochs=200,patience=15,return_best_epoch=True)
  out.append(mlp_predict(fit,evaluate.float()).astype(np.float64));validation.append(mlp_predict(fit,val.float()).astype(np.float64));epochs.append(ep);del fit
 return np.mean(out,axis=0),np.mean(validation,axis=0),epochs

def split_manifest_fields(guard):
 result={k:v for k,v in guard.items() if k not in ('split_config','seed')}
 result['split_seed']=guard['seed']
 return result

def main_payload(scores,keep):
 out={k:np.asarray(v)[keep] for k,v in scores.items()}
 if any(v.dtype.hasobject for v in out.values()):raise RuntimeError('OBJECT_METADATA_FORBIDDEN')
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=('clip','dinov2','siglip','mae'),required=True);ap.add_argument('--pair-ids',default='all');ap.add_argument('--fold',type=int,choices=(0,1,2,3,4),required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
 frozen=require_frozen_plan(PLAN,FROZEN)
 torch.set_num_threads(4);dev=torch.device('cuda:0');root=BASE/'analysis/sp27_confirmatory';split_file=root/f'fold{a.fold:02d}_A80B20.npz';outer=root/'COUNT_ONLY_MANIFEST.json';count_manifest=json.loads(outer.read_text())
 if count_manifest['plan_sha256']!=frozen['analysis_plan_sha256'] or count_manifest['eligibility_sha256']!=sha256(root/'ELIGIBILITY_TABLE.csv') or count_manifest['qualified_pairs_sha256']!=sha256(root/'QUALIFIED_PAIRS.csv'):raise RuntimeError('COUNT_ONLY_MANIFEST_HASH')
 pairfile=guarded_input(PLAN,FROZEN,'celeba','pairs');eligibility_root=root;eligibility_file=root/'PAIR_FOLD_ELIGIBILITY.csv';eligibility_table=pd.read_csv(eligibility_file);selected_pairs=pd.read_csv(root/'QUALIFIED_PAIRS.csv');requested=list(selected_pairs.pair_id.astype(int)) if a.pair_ids=='all' else [int(s) for s in a.pair_ids.split(',')]
 if not set(requested).issubset(set(selected_pairs.pair_id)):raise RuntimeError('PAIR_NOT_FIVE_FOLD_QUALIFIED')
 manifest=pd.read_csv(guarded_input(PLAN,FROZEN,'celeba','manifest'),sep='\t');guarded_input(PLAN,FROZEN,'celeba','attributes');pre,names,labels=pre_labels(manifest,split='confirmatory');groups=pre.identity_id.to_numpy();ids=pre.image_filename.to_numpy().astype(str);pairs=pd.read_csv(pairfile);t2file=guarded_input(PLAN,FROZEN,'celeba',a.model+'_T2');t2=pd.read_csv(t2file)
 with np.load(split_file,allow_pickle=False) as d:splits={k:d[k] for k in d.files}
 tr=np.sort(np.concatenate((splits['A'],splits['B'])));te=splits['test'];replayed,guard=identity_split(tr,te,groups,a.fold)
 if any(not np.array_equal(splits[k],replayed[k]) for k in splits):raise RuntimeError('SPLIT_REPLAY_MISMATCH')
 if sha256(split_file)!=count_manifest['folds'][a.fold]['split_sha256']:raise RuntimeError('SPLIT_CHECKSUM')
 AT,AV,BT,BV=[splits[k] for k in ('A_train','A_val','B_train','B_val')];keep=np.sort(np.concatenate((AV,BV,te)))
 for key in ('A_train','A_val','B_train','B_val','test'):
  for other in ('A_train','A_val','B_train','B_val','test'):
   if key!=other and np.intersect1d(groups[splits[key]],groups[splits[other]]).size:raise RuntimeError('IDENTITY_OVERLAP')
 role=np.full(len(pre),'',dtype='<U8')
 for key in ('A_train','A_val','B_train','B_val','test'):role[splits[key]]=key
 feature=guarded_input(PLAN,FROZEN,'celeba',a.model);mm=np.load(feature,mmap_mode='r');mapping={s:i for i,s in enumerate(manifest.image_filename)};x=np.asarray(mm[[mapping[s] for s in pre.image_filename]],np.float64)
 if not np.isfinite(x).all():raise RuntimeError('NONFINITE_FEATURES')
 head='anonymous-release';a.output.mkdir(parents=True,exist_ok=True);records=[]
 source_hashes={str(p.name):sha256(p) for p in (Path(__file__),Path(__file__).with_name('sp27_core.py'),Path(__file__).with_name('sp27_mlp.py'),Path(__file__).with_name('leace_cc2.py'),Path(__file__).with_name('sp27_attack_selection.py'),Path(__file__).with_name('sp27_plan_guard.py'))}
 for pair_id in requested:
  pair=pairs[pairs.pair_id==pair_id].iloc[0];zi=names.index(pair.target_Z);wi=names.index(pair.retained_W);z=labels[:,zi];w=labels[:,wi];beta=float(w[te][z[te]==1].mean()-w[te][z[te]==0].mean());seed=20261103+1000000*('clip','dinov2','siglip','mae').index(a.model)+10000*zi+100*a.fold
  config=dict(exploratory=False,analysis_plan_sha256=frozen['analysis_plan_sha256'],frozen_config_sha256=sha256(FROZEN),split_config='A80B20',split_version='identity_seed20261103_plus_fold',model=a.model,pair_id=pair_id,fold=a.fold,seed=seed,git_head=head,source_hashes=source_hashes,split_sha256=sha256(split_file),outer_sha256=sha256(outer),feature_sha256=frozen['inputs']['celeba'][a.model]['sha256'],T2_config_sha256=sha256(t2file),eligibility_sha256=sha256(eligibility_file),qualified_pairs_sha256=sha256(eligibility_root/'QUALIFIED_PAIRS.csv'),pairfile_sha256=sha256(pairfile),utility_grid=UTILITY_GRID,methods=METHODS,**split_manifest_fields(guard))
  eq=eligibility_table[(eligibility_table.pair_id==pair_id)&(eligibility_table.fold==a.fold)]
  if len(eq)!=1:raise RuntimeError('ELIGIBILITY_ROW_MISSING')
  if not bool(eq.iloc[0].pair_primary_eligible):raise RuntimeError('PAIR_PRIMARY_INELIGIBLE')
  if not bool(eq.iloc[0].fold_eligible):
   records.extend([dict(model=a.model,pair_id=pair_id,fold=a.fold,method=method,status='NA',reason=str(eq.iloc[0].reason),beta=beta,abs_beta=abs(beta)) for method in METHODS]);continue
  ch=hashlib.sha256(json.dumps({k:v for k,v in config.items() if k!='git_head'},sort_keys=True).encode()).hexdigest();folder=a.output/f'pair{pair_id:03d}';folder.mkdir(exist_ok=True);csv=folder/'METHOD_RESULTS.csv';npz=folder/'SCORES.npz';marker=folder/'COMPLETE.json'
  expected_shapes={k:(len(keep),) for k in ('sample_id','identity_id','partition','fold','W','Z')}
  for method in METHODS:
   expected_shapes[f'{method}__A_train_linear_attack']=(len(keep),)
   for probe in ('linear','MLP'):expected_shapes[f'{method}__utility_{probe}']=(len(keep),)
   for condition in ('conditional','unconditional'):
    for family in ('linear','MLP','suite'):expected_shapes[f'{method}__{condition}_{family}']=(len(keep),)
  if marker.exists():
   done=json.loads(marker.read_text())
   if done.get('config_hash')==ch and validate_file(csv,expected_rows=3,checksum=done.get('csv_sha256')) and validate_file(npz,shapes=expected_shapes,checksum=done.get('npz_sha256')):
    records.extend(pd.read_csv(csv).to_dict('records'));continue
  base_mu=x[AT].mean(0);base_sd=x[AT].std(0).clip(1e-8);base=(x-base_mu)/base_sd;cfg=frozen_attack_config(t2,pair.target_Z,a.fold);scores=dict(sample_id=ids,identity_id=groups,partition=role,fold=np.full(len(pre),a.fold),W=w,Z=z);method_rows=[];raw_scores={}
  for mi,method in enumerate(METHODS):
   start=time.monotonic();cov_error=0.;rank=0
   if method=='raw':xx=base.copy()
   elif method=='leace':fit=fit_leace64(base[AT],z[AT]);xx=transform_leace64(base,fit);rank=1
   else:fit=fit_cc2(base[AT],z[AT],w[AT]);xx=transform_cc2(base,fit);rank=fit['rank']
   if method=='leace':cov_error=float(np.linalg.norm((xx[AT]-xx[AT].mean(0)).T@(z[AT]-z[AT].mean())/len(AT)))
   if method=='leace_cc2':
    for value in (0,1):
     ix=AT[w[AT]==value];cov_error=max(cov_error,float(np.linalg.norm((xx[ix]-xx[ix].mean(0)).T@(z[ix]-z[ix].mean())/len(ix))))
   if cov_error>=1e-8:raise RuntimeError(f'COVARIANCE_GATE {method} {cov_error}')
   features=torch.as_tensor(xx,device=dev,dtype=torch.float64);wy=torch.as_tensor(w,dtype=torch.float32,device=dev);zy=torch.as_tensor(z,dtype=torch.float32,device=dev);row=dict(model=a.model,pair_id=pair_id,W=pair.retained_W,Z=pair.target_Z,fold=a.fold,method=method,beta=beta,abs_beta=abs(beta),erasure_rank=rank,covariance_error=cov_error,split_check=True,config_hash=ch,exploratory=False,split_config='A80B20')
   for wv in (0,1):
    for zv in (0,1):row[f'n_W{wv}Z{zv}']=int(np.sum((w[te]==wv)&(z[te]==zv)))
   # Utility on A_train, configurations selected on A_val; no all-A refit.
   mu=features[AT].mean(0);sd=features[AT].std(0,unbiased=False).clamp_min(1e-8);scaled=(features-mu)/sd;best=None
   for C in (.1,1.,10.):
    c,b=logistic64(scaled[AT],wy[AT],C=C);pred=(scaled@c+b).detach().cpu().numpy();quality=validation_auc(w,pred,AV)
    if best is None or quality>best[0]:best=(quality,pred,C)
   scores[f'{method}__utility_linear']=best[1];row['utility_linear_C']=best[2];best_mlp=None
   for ci,settings in enumerate(UTILITY_GRID):
    fit,quality,_,ep=fit_mlp(scaled[AT].float(),wy[AT],scaled[AV].float(),wy[AV],*settings,seed+500000+1000*wi+mi*10+ci,epochs=120,patience=12,return_best_epoch=True)
    pred=mlp_predict(fit,scaled.float()).astype(np.float64)
    if best_mlp is None or quality>best_mlp[0]:best_mlp=(quality,pred,ci,ep)
    del fit
   scores[f'{method}__utility_MLP']=best_mlp[1];row['utility_MLP_config']=best_mlp[2];row['utility_MLP_best_epoch']=best_mlp[3]
   # First layer: in-sample A_train linear target attack, separate from B attacker.
   c,b=logistic64(scaled[AT],zy[AT]);row['A_insample_linear_AUROC']=auc(z[AT],(scaled[AT]@c+b).detach().cpu().numpy());row['A_insample_metric_role']='F1_only';scores[f'{method}__A_train_linear_attack']= (scaled@c+b).detach().cpu().numpy().astype(np.float64)
   for condition in (False,True):
    ff=torch.cat((features,wy[:,None].double()),1) if condition else features;mu=ff[BT].mean(0);sd=ff[BT].std(0,unbiased=False).clamp_min(1e-8);ff=(ff-mu)/sd;c,b=logistic64(ff[BT],zy[BT]);linear=(ff@c+b).detach().cpu().numpy();attack_seed=seed+mi*10+(1000*wi+500000 if condition else 0);mlp,mv,epochs=mlp3(ff[BT],zy[BT],ff[BV],zy[BV],ff,cfg,attack_seed);selection=select_attack(z[BV],w[BV],linear[BV],mv,condition);chosen=selection['chosen'];prefix='conditional' if condition else 'unconditional';row[prefix+'_selected']=chosen;row[prefix+'_selection_metric']=selection['selection_metric'];row[prefix+'_selection_fallback']=selection['fallback'];row[prefix+'_fallback_reason']=selection['fallback_reason'];row[prefix+'_B_val_counts']=json.dumps(selection['B_val_counts']);row[prefix+'_validation_linear']=selection['validation_linear'];row[prefix+'_validation_MLP']=selection['validation_MLP'];row[prefix+'_MLP_seeds']=json.dumps([attack_seed+i for i in range(3)])
    for family,pred in [('linear',linear),('MLP',mlp),('suite',mlp if chosen=='MLP' else linear)]:
     scores[f'{method}__{prefix}_{family}']=np.asarray(pred,np.float64)
     if condition:
      stat=conditional_auc(z[te],w[te],pred[te]);row[f'conditional_{family}_AUROC']=stat['auc'];row['conditional_pair_count']=stat['pair_count'];row['conditional_eligible']=stat['eligible']
     else:row[f'unconditional_{family}_AUROC']=auc(z[te],pred[te])
   for probe in ('linear','MLP'):
    pred=scores[f'{method}__utility_{probe}'][te]
    if method=='raw':raw_scores[probe]=pred.copy()
    dec=decompose_fold(w[te],z[te],raw_scores[probe],pred);row[f'{probe}_within']=dec['within_term'];row[f'{probe}_cross']=dec['cross_term'];row[f'{probe}_total']=dec['total_damage'];row[f'{probe}_identity_error']=dec['identity_error'];row['utility_pair_count']=dec['pair_count']
    for i,j in ((0,0),(1,1),(0,1),(1,0)):
     row[f'{probe}_w_{i}{j}']=dec['raw']['cells'][(i,j)]['weight'];row[f'{probe}_AUC_{i}{j}_raw']=dec['raw']['cells'][(i,j)]['auc'];row[f'{probe}_AUC_{i}{j}_erased']=dec['erased']['cells'][(i,j)]['auc']
   row['seconds']=time.monotonic()-start;method_rows.append(row);print(a.model,pair_id,a.fold,method,'done',row['seconds'],flush=True)
  temp=npz.with_suffix('.tmp.npz');np.savez_compressed(temp,**main_payload(scores,keep));os.replace(temp,npz)
  if not validate_file(npz,shapes=expected_shapes):raise RuntimeError('WRITTEN_MAIN_NPZ_INVALID')
  temp=csv.with_suffix('.tmp.csv');pd.DataFrame(method_rows).to_csv(temp,index=False);os.replace(temp,csv);(folder/'RUN_CONFIG.json').write_text(json.dumps(config,indent=2)+'\n');done=dict(config_hash=ch,npz_sha256=sha256(npz),csv_sha256=sha256(csv));marker.write_text(json.dumps(done)+'\n');records.extend(method_rows)
 pd.DataFrame(records).to_csv(a.output/'FOLD_SUMMARY.csv',index=False)
 (a.output/'COMPLETE.json').write_text(json.dumps(dict(model=a.model,fold=a.fold,expected_rows=len(requested)*3,rows=len(records),git_head=head,plan_sha256=frozen['analysis_plan_sha256']),indent=2)+'\n')
if __name__=='__main__':main()
