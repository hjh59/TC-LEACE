"""Exact tie-aware AUROC decomposition by two binary labels.

For each cell (i,j), positives have W=1,Z=i and negatives W=0,Z=j.
The cell weight is P(W=1,Z=i) P(W=0,Z=j) / (P(W=1) P(W=0)).
Diagonal contributions form the exact within-cell part; off-diagonal
contributions form the exact cross-cell part. Their sum equals pooled AUROC.
"""
from __future__ import annotations
import numpy as np

CELL_ORDER=((0,0),(0,1),(1,0),(1,1))

def _corr(a,b,w):
    sw=w.sum()
    if sw<=0:return float('nan')
    ma=np.dot(w,a)/sw;mb=np.dot(w,b)/sw
    va=np.dot(w,(a-ma)**2);vb=np.dot(w,(b-mb)**2)
    if va<=0 or vb<=0:return float('nan')
    return float(np.dot(w,(a-ma)*(b-mb))/np.sqrt(va*vb))

def _weighted_auc(y,s,w):
    ok=np.isfinite(s)&(w>0);y=y[ok].astype(int);s=s[ok];w=w[ok]
    p=float(w[y==1].sum());n=float(w[y==0].sum())
    if p<=0 or n<=0:return float('nan')
    order=np.argsort(s,kind='mergesort');s=s[order];y=y[order];w=w[order]
    starts=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1]
    inv=np.repeat(np.arange(len(starts)),np.diff(np.r_[starts,len(s)]))
    wp=np.bincount(inv,weights=w*(y==1),minlength=len(starts))
    wn=np.bincount(inv,weights=w*(y==0),minlength=len(starts))
    return float(wp@(np.cumsum(wn)-.5*wn)/(p*n))

def one_score(y,z,s,w=None):
    y=np.asarray(y,dtype=np.int8);z=np.asarray(z,dtype=np.int8);s=np.asarray(s,dtype=np.float64)
    w=np.ones(len(y),dtype=np.float64) if w is None else np.asarray(w,dtype=np.float64)
    if not (len(y)==len(z)==len(s)==len(w)):raise ValueError('input lengths differ')
    npos=float(w[y==1].sum());nneg=float(w[y==0].sum())
    if npos<=0 or nneg<=0:raise ValueError('W must have both classes')
    cells={};pooled=_weighted_auc(y,s,w)
    for i,j in CELL_ORDER:
        pm=(y==1)&(z==i);nm=(y==0)&(z==j)
        np_=float(w[pm].sum());nn_=float(w[nm].sum());weight=np_*nn_/(npos*nneg)
        val=_weighted_auc(np.r_[np.ones(int(pm.sum()),dtype=np.int8),np.zeros(int(nm.sum()),dtype=np.int8)],
                          np.r_[s[pm],s[nm]],np.r_[w[pm],w[nm]]) if np_>0 and nn_>0 else float('nan')
        cells[(i,j)]={'weight':weight,'auc':val,'contribution':weight*val if np.isfinite(val) else 0.0}
    decomp=sum(q['contribution'] for q in cells.values())
    if abs(decomp-pooled)>1e-9:raise AssertionError(f'exact AUC identity failed: {decomp} vs {pooled}')
    return {'pooled_auc':pooled,'cells':cells,'decomposition_auc':decomp}

def compare_scores(y,z,raw,erased,w=None):
    y=np.asarray(y,dtype=np.int8);z=np.asarray(z,dtype=np.int8)
    w=np.ones(len(y),dtype=np.float64) if w is None else np.asarray(w,dtype=np.float64)
    r=one_score(y,z,raw,w);e=one_score(y,z,erased,w)
    wd=sum(r['cells'][(i,i)]['contribution']-e['cells'][(i,i)]['contribution'] for i in (0,1))
    cd=sum(r['cells'][(i,j)]['contribution']-e['cells'][(i,j)]['contribution'] for i,j in ((0,1),(1,0)))
    total=r['pooled_auc']-e['pooled_auc']
    if abs(wd+cd-total)>1e-9:raise AssertionError(f'damage identity failed: {wd}+{cd}!={total}')
    # Conventional CCD: stratum-size weighted within-Z AUROC drop.
    ccd=0.0;est=True
    for i in (0,1):
        wi=w[z==i];yi=y[z==i]
        if wi.sum()<=0 or wi[yi==1].sum()<=0 or wi[yi==0].sum()<=0:
            est=False;break
        ar=_weighted_auc(yi,np.asarray(raw)[z==i],wi);ae=_weighted_auc(yi,np.asarray(erased)[z==i],wi)
        ccd+=(wi.sum()/w.sum())*(ar-ae)
    phi=_corr(y.astype(float),z.astype(float),w)
    pz1=float(w[z==1].sum());pz0=float(w[z==0].sum())
    beta=(float(w[(y==1)&(z==1)].sum()/pz1)-float(w[(y==1)&(z==0)].sum()/pz0)) if pz1>0 and pz0>0 else float('nan')
    return {'raw':r,'erased':e,'within_term':float(wd),'cross_term':float(cd),'total_damage':float(total),
            'ccd':float(ccd) if est else float('nan'),'ccd_estimable':est,'phi':phi,'beta':beta,
            'raw_borrow':r['pooled_auc']-ccd if est else float('nan')}

def pair_bootstrap(y,z,raw,erased,groups,replicates=1000,seed=0):
    """Cluster bootstrap conditional on fixed predictions; returns T5 intervals."""
    y=np.asarray(y,dtype=np.int8);z=np.asarray(z,dtype=np.int8);groups=np.asarray(groups)
    raw=np.asarray(raw,dtype=float);erased=np.asarray(erased,dtype=float)
    unique,gi=np.unique(groups,return_inverse=True);rng=np.random.default_rng(seed);rows=[]
    rr,raw_codes=np.unique(raw,return_inverse=True);er,erased_codes=np.unique(erased,return_inverse=True)
    def score_state(codes,nlevels,w):
        pos=[np.bincount(codes,weights=w*((y==1)&(z==i)),minlength=nlevels) for i in (0,1)]
        neg=[np.bincount(codes,weights=w*((y==0)&(z==i)),minlength=nlevels) for i in (0,1)]
        pcounts=[float(q.sum()) for q in pos];ncounts=[float(q.sum()) for q in neg]
        npos=sum(pcounts);nneg=sum(ncounts)
        if npos<=0 or nneg<=0:return None
        cells={}
        for i,j in CELL_ORDER:
            weight=pcounts[i]*ncounts[j]/(npos*nneg)
            auc=float(pos[i]@(np.cumsum(neg[j])-.5*neg[j])/(pcounts[i]*ncounts[j])) if pcounts[i]>0 and ncounts[j]>0 else float('nan')
            cells[(i,j)]={'weight':weight,'auc':auc,'contribution':weight*auc if np.isfinite(auc) else 0.0}
        pooled=sum(v['contribution'] for v in cells.values())
        return cells,pooled,pcounts,ncounts
    for b in range(replicates):
        wc=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique)).astype(float)
        w=wc[gi];r=score_state(raw_codes,len(rr),w);e=score_state(erased_codes,len(er),w)
        if r is None or e is None:continue
        rc,ra,pc,nc=r;ec,ea,_,_=e
        within=sum(rc[(i,i)]['contribution']-ec[(i,i)]['contribution'] for i in (0,1))
        cross=sum(rc[(i,j)]['contribution']-ec[(i,j)]['contribution'] for i,j in ((0,1),(1,0)))
        total=ra-ea
        if abs(within+cross-total)>1e-9:raise AssertionError('bootstrap exact damage identity failed')
        pop=[float(w[z==i].sum()) for i in (0,1)];ccd=0.0;ok=True
        for i in (0,1):
            if pc[i]<=0 or nc[i]<=0 or pop[i]<=0:ok=False;break
            ar=float(rc[(i,i)]['auc']);ae=float(ec[(i,i)]['auc'])
            ccd+=pop[i]/float(w.sum())*(ar-ae)
        rows.append([within,cross,ccd if ok else np.nan,total])
    a=np.asarray(rows,dtype=float)
    if len(a)==0:raise RuntimeError('No valid cluster bootstrap replicates')
    return {'valid_replicates':len(a),'requested_replicates':replicates,'replicate_values':a,
            'metrics':{name:[float(v) for v in np.nanquantile(a[:,i],[.025,.975])] for i,name in enumerate(('within_term','cross_term','ccd','total_damage'))}}
