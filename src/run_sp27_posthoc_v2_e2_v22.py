"""POST HOC – not preregistered. V2.2 E2 frozen readers, cached erasers."""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score
from sp27_v22_scope import PRED_SHA,fit_indices
from sp27_plan_guard import guarded_input,file_sha256
from sp27_core import logistic64,validate_file
from sp27_mlp import fit_mlp,mlp_predict
from sp27_attack_selection import select_attack
from sp27_posthoc_v2_erasers import transform_splince,transform_weber
from sp27_core import transform_leace64
from leace_cc2 import transform_cc2
from run_sp27_candid import mlp_ensemble,scale,save_npz,atomic_json,ACFG
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/case_studies/candid_ptx/sp27_posthoc_v2_2';OLD=ROOT/'results/case_studies/candid_ptx/sp27';TITLE='POST HOC – not preregistered';MODELS=('rad-dino','dinov2','biomedclip');SEED=20261103;METHOD_INDEX={'leace':1,'leace_cc2':2,'splince':3,'weber':4}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS,required=True);args=ap.parse_args();model=args.model
 if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
 if (OUT/'E2_V22_STOP.json').exists():raise RuntimeError('E2_V22_STOP_REQUIRES_REVIEW')
 preflight=json.loads((OUT/'E2_V22_PREFLIGHT_COMPLETE.json').read_text());amend=file_sha256(OUT/'AMENDMENT_V2_2.md')
 if not preflight['passed'] or preflight['amendment_sha256']!=amend:raise RuntimeError('PREFLIGHT_INVALID')
 if file_sha256(OUT/'PREDICTIONS_V2_2.md')!=PRED_SHA:raise RuntimeError('PREDICTIONS_CHANGED')
 torch.set_num_threads(4);device=torch.device('cuda:0');plan=ROOT/'ANALYSIS_PLAN_SP27.md';config=ROOT/'SP27_FROZEN_CONFIG.json'
 mp=guarded_input(plan,config,'candid','manifest');xp=guarded_input(plan,config,'candid',model);m=pd.read_csv(mp);x=np.asarray(np.load(xp,mmap_mode='r'),np.float64);z=m.tube.to_numpy();w=m.PTX.to_numpy();groups=m.patient_group.astype(str).to_numpy(dtype=str);ids=m.dicom_id.astype(str).to_numpy(dtype=str) if 'dicom_id' in m else np.asarray([f'candid_row{i}' for i in range(len(m))]);head='anonymous-release'
 folder=OUT/'E2_v22'/'readers'/model;folder.mkdir(parents=True,exist_ok=True)
 sources={n:file_sha256(Path(__file__).with_name(n)) for n in ('run_sp27_posthoc_v2_e2_v22.py','run_sp27_candid.py','sp27_core.py','sp27_mlp.py','sp27_attack_selection.py','sp27_posthoc_v2_erasers.py')}
 run=dict(analysis_title=TITLE,model=model,git_head=head,amendment_sha256=amend,plan_sha256=file_sha256(plan),prediction_sha256=file_sha256(OUT/'PREDICTIONS_V2_2.md'),attacker_config=ACFG,seed=SEED,sources=sources,split_config='A80B20',preflight_sha256=file_sha256(OUT/'E2_V22_PREFLIGHT_COMPLETE.json'))
 import hashlib
 run['config_hash']=hashlib.sha256(json.dumps(run,sort_keys=True).encode()).hexdigest();atomic_json(folder/'RUN_CONFIG.json',run);rows=[]
 for fold in range(5):
  split=OLD/f'main_{model}'/f'fold{fold:02d}_split.npz'
  with np.load(split,allow_pickle=False) as s:parts={k:s[k] for k in s.files}
  at,av,bt,bv,te=[parts[k] for k in ('A_train','A_val','B_train','B_val','test')];fit_ix=fit_indices(parts,len(x))
  for scenario,methods in [('S-rel2',('leace','leace_cc2','splince','weber'))]:
   for method in methods:
    if (OUT/'E2_V22_STOP.json').exists():raise RuntimeError('OTHER_E2_JOB_FAILED')
    path=folder/f'{scenario}_fold{fold}_{method}_SCORES.npz';marker=path.with_suffix('.json');cache=OUT/'E2_v22'/'cache'/model/scenario/f'fold{fold:02d}'/method;done=json.loads((cache/'COMPLETE.json').read_text());cache_sha=file_sha256(cache/'TRANSFORM.npz')
    if cache_sha!=done['cache_sha256'] or done['split_sha256']!=file_sha256(split) or done['amendment_sha256']!=amend:raise RuntimeError('TRANSFORM_CACHE_INVALID')
    membership=OUT/'FIT_MEMBERSHIP'/model/f'fold{fold:02d}.npz'
    if file_sha256(membership)!=done['fit_membership_sha256']:raise RuntimeError('FIT_MEMBERSHIP_HASH')
    with np.load(membership,allow_pickle=False) as membership_data:
     if not np.array_equal(membership_data['fit_indices'],fit_ix):raise RuntimeError('FIT_MEMBERSHIP_MISMATCH')
    shapes={f'{role}_{k}':(len(parts[role]),) for role in ('A_val','B_val','test') for k in ('sample_id','identity_id','W','Z','fold','partition','utility_linear','utility_MLP','conditional_linear','conditional_MLP','conditional_suite','unconditional_linear','unconditional_MLP','unconditional_suite')}
    if marker.exists():
     previous=json.loads(marker.read_text())
     if previous.get('config_hash')==run['config_hash'] and previous.get('cache_sha256')==cache_sha and validate_file(path,shapes=shapes,checksum=previous.get('score_sha256')):
      rows.append(previous);print('restored',model,scenario,fold,method,flush=True);continue
    start=time.monotonic()
    with np.load(cache/'TRANSFORM.npz',allow_pickle=False) as d:fit={k:d[k] for k in d.files}
    base=(x-fit['base_mean'])/fit['base_sd'];xx={'leace':transform_leace64,'leace_cc2':transform_cc2,'splince':transform_splince}[method](base,fit) if method!='weber' else transform_weber(base,z,fit)
    ff=torch.as_tensor(xx,device=device,dtype=torch.float64);wy=torch.as_tensor(w,device=device,dtype=torch.float32);zy=torch.as_tensor(z,device=device,dtype=torch.float64);u=scale(ff,at);seed=SEED+100000*MODELS.index(model)+1000*fold+100*METHOD_INDEX[method]+750000;scores={};row=dict(analysis_title=TITLE,model=model,scenario=scenario,fold=fold,method=method,seed=seed,git_head=head,config_hash=run['config_hash'],split_sha256=file_sha256(split),cache_sha256=cache_sha,oracle_true_Z_at_application=method=='weber',fit_membership_sha256=file_sha256(membership),n_eraser_fit=len(fit_ix),amendment_sha256=amend)
    best=None
    for C in (.1,1.,10.):
     c,b=logistic64(u[at],wy[at],C=C);pred=(u@c+b).cpu().numpy();quality=roc_auc_score(w[av],pred[av])
     if best is None or quality>best[0]:best=(quality,pred,C)
    scores['utility_linear']=best[1];row['utility_linear_C']=best[2];best=None
    for ci,(drop,lr) in enumerate(((0.,.001),(.1,.0003))):
     net,quality,_,ep=fit_mlp(u[at].float(),wy[at],u[av].float(),wy[av],(256,),drop,lr,.0001,seed+ci,epochs=120,patience=12,return_best_epoch=True);pred=mlp_predict(net,u.float()).astype(np.float64)
     if best is None or quality>best[0]:best=(quality,pred,ci,ep)
     del net
    scores['utility_MLP']=best[1];row['utility_MLP_config']=best[2];row['utility_MLP_epoch']=best[3]
    for conditional in (False,True):
     af=scale(torch.cat((ff,wy[:,None].double()),1) if conditional else ff,bt);c,b=logistic64(af[bt],zy[bt]);linear=(af@c+b).cpu().numpy();mlp,epochs=mlp_ensemble(af,z,bt,bv,seed+(30 if conditional else 20));selection=select_attack(z[bv],w[bv],linear[bv],mlp[bv],conditional);prefix='conditional' if conditional else 'unconditional';row[prefix+'_selection']=selection;row[prefix+'_MLP_epochs']=epochs
     for family,pred in [('linear',linear),('MLP',mlp),('suite',mlp if selection['chosen']=='MLP' else linear)]:scores[prefix+'_'+family]=pred
    arrays={}
    for role in ('A_val','B_val','test'):
     ix=parts[role]
     for key,value in dict(sample_id=ids[ix],identity_id=groups[ix],W=w[ix],Z=z[ix],fold=np.full(len(ix),fold),partition=np.full(len(ix),role),**{k:np.asarray(v[ix],np.float64) for k,v in scores.items()}).items():arrays[f'{role}_{key}']=value
    save_npz(path,arrays)
    if not validate_file(path,shapes=shapes):raise RuntimeError('NEW_SCORE_INTEGRITY')
    row.update(score_sha256=file_sha256(path),seconds=time.monotonic()-start);atomic_json(marker,row);rows.append(row);pd.DataFrame(rows).to_csv(folder/'PROGRESS.csv',index=False);print('complete',model,scenario,fold,method,'seconds',row['seconds'],flush=True);del ff,u;torch.cuda.empty_cache()
 if len(rows)!=20:raise RuntimeError('E2_COVERAGE')
 atomic_json(folder/'COMPLETE.json',dict(analysis_title=TITLE,model=model,rows=len(rows),git_head=head,amendment_sha256=amend));print('E2 model complete',model,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  (OUT/'E2_V22_STOP.json').write_text(json.dumps(dict(analysis_title=TITLE,stage='E2 reader training',error=repr(e),time=time.time()),indent=2)+'\n');raise
