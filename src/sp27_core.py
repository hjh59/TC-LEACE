"""SP27 closed-form sprint primitives; no INLP or legacy analysis imports."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from leace_cc2 import fit_cc2,transform_cc2
from pairwise_decomposition import compare_scores
from t20_split_guard import validate_split,sha256

class ProtocolError(RuntimeError):pass

def identity_split(train,test,groups,fold):
    train=np.asarray(train);g=np.asarray(groups);rng=np.random.default_rng(20261103+fold);ids=np.unique(g[train]);ids=rng.permutation(ids);nb=int(round(.2*len(ids)));B=train[np.isin(g[train],ids[:nb])];A=train[np.isin(g[train],ids[nb:])]
    record=validate_split(A,B,train,test,g,'A80B20')
    out=dict(A=A,B=B,test=np.asarray(test))
    for name,idx in [('A',A),('B',B)]:
        gids=rng.permutation(np.unique(g[idx]));nv=max(1,int(round(.1*len(gids))));mask=np.isin(g[idx],gids[:nv]);out[name+'_val']=idx[mask];out[name+'_train']=idx[~mask]
    for i,one in enumerate(('A_train','A_val','B_train','B_val','test')):
        for two in ('A_train','A_val','B_train','B_val','test')[i+1:]:
            if np.intersect1d(g[out[one]],g[out[two]]).size:raise ProtocolError('IDENTITY_OVERLAP')
    return out,dict(**record,seed=20261103+fold)

def fit_leace64(x,z):
    x=np.asarray(x,np.float64);z=np.asarray(z,np.float64);mu=x.mean(0);xc=x-mu;c=xc.T@(z-z.mean())/len(x);cov=xc.T@xc/len(x)+1e-6*np.eye(x.shape[1]);v=np.linalg.solve(cov,c);den=float(c@v)
    removal=np.outer(v,c)/den if den>np.finfo(float).eps else np.zeros_like(cov)
    # row-vector convention: X'=(X-mu)@(I-v c^T/(c^T v))+mu
    return dict(mean=mu,matrix=np.eye(x.shape[1])-removal)

def transform_leace64(x,fit):return (np.asarray(x,np.float64)-fit['mean'])@fit['matrix']+fit['mean']

def conditional_auc(z,w,score,min_class=30):
    z=np.asarray(z);w=np.asarray(w);score=np.asarray(score);vals=[];weights=[];cells=[]
    for value in (0,1):
        mask=w==value;n1=int(z[mask].sum());n0=int(mask.sum()-n1);cells.append((n0,n1))
        if min(n0,n1)<min_class:return dict(auc=np.nan,pair_count=0,eligible=False,cells=cells)
        vals.append(roc_auc_score(z[mask],score[mask]));weights.append(n0*n1)
    return dict(auc=float(np.average(vals,weights=weights)),pair_count=int(sum(weights)),eligible=True,cells=cells)

def decompose_fold(w,z,raw,erased):
    d=compare_scores(w,z,np.asarray(raw,np.float64),np.asarray(erased,np.float64));err=abs(d['within_term']+d['cross_term']-d['total_damage'])
    if err>=1e-12:raise ProtocolError('DECOMPOSITION_IDENTITY')
    d['pair_count']=int(np.sum(w))*int(len(w)-np.sum(w));d['identity_error']=err;return d

def aggregate_decomposition(rows):
    weights=np.array([r['pair_count'] for r in rows],float);out={k:float(np.average([r[k] for r in rows],weights=weights)) for k in ('within_term','cross_term','total_damage')}
    if abs(out['within_term']+out['cross_term']-out['total_damage'])>=1e-12:raise ProtocolError('AGGREGATE_IDENTITY')
    return out

def validate_file(path,expected_rows=None,shapes=None,checksum=None):
    path=Path(path)
    try:
        if not path.is_file() or path.stat().st_size==0:return False
        with path.open('rb') as f:
            if not any(f.read(4096)):return False
        if checksum is not None and sha256(path)!=checksum:return False
        if path.suffix=='.csv':
            d=pd.read_csv(path)
            if expected_rows is None or len(d)!=expected_rows:return False
        elif path.suffix=='.npz':
            with np.load(path,allow_pickle=False) as d:
                arrays={k:d[k] for k in d.files}
                if shapes is None:return False
                if any(k not in arrays or arrays[k].shape!=tuple(shape) for k,shape in shapes.items()):return False
                if any(np.issubdtype(v.dtype,np.number) and not np.isfinite(v).all() for v in arrays.values()):return False
        else:return False
        return True
    except Exception:return False

def resume_or_recompute(path,build,**checks):
    if validate_file(path,**checks):return 'restored'
    build()
    if not validate_file(path,**checks):raise ProtocolError('INVALID_RECOMPUTED_FILE')
    return 'recomputed'

def logistic64(x,y,C=1.,max_iter=1000):
    a=x.to(torch.float64);yy=y.to(device=x.device,dtype=torch.float64);n,d=a.shape
    coef=torch.zeros(d,device=x.device,dtype=a.dtype,requires_grad=True);bias=torch.zeros((),device=x.device,dtype=a.dtype,requires_grad=True)
    n1=yy.sum().clamp_min(1);n0=(n-yy.sum()).clamp_min(1);sw=torch.where(yy>.5,n/(2*n1),n/(2*n0))
    opt=torch.optim.LBFGS([coef,bias],max_iter=max_iter,tolerance_grad=1e-10,tolerance_change=1e-14,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=(torch.nn.functional.binary_cross_entropy_with_logits(a@coef+bias,yy,reduction='none')*sw).mean()+coef.square().sum()/(2*C*n);loss.backward();return loss
    opt.step(closure);return coef.detach(),bias.detach()

def validation_auc(labels_all,scores_all,validation_indices):
    """Index BOTH labels and predictions on the same validation row IDs."""
    labels=np.asarray(labels_all);scores=np.asarray(scores_all);ix=np.asarray(validation_indices,dtype=np.int64)
    if labels.ndim!=1 or scores.shape!=labels.shape:raise ProtocolError('VALIDATION_FULL_VECTOR_ALIGNMENT')
    if ix.ndim!=1 or len(ix)==0 or len(np.unique(ix))!=len(ix) or ix.min()<0 or ix.max()>=len(labels):raise ProtocolError('VALIDATION_INDEX_INVALID')
    return float(roc_auc_score(labels[ix],scores[ix]))
