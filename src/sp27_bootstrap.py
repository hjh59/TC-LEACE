"""Paired within-test-fold cluster bootstrap with recomputed pair weights."""
import numpy as np
from pairwise_decomposition import _weighted_auc,compare_scores

def fold_cluster_weights(identity_ids,rng):
    unique,inverse=np.unique(identity_ids,return_inverse=True)
    multiplicity=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))
    return multiplicity[inverse].astype(np.float64)

def recompute_fold(w,z,raw_utility,utility,uncond,cond,weights,min_cond_class=30):
    w=np.asarray(w);z=np.asarray(z);weights=np.asarray(weights,dtype=float)
    wn=float(weights[w==0].sum());wp=float(weights[w==1].sum());zn=float(weights[z==0].sum());zp=float(weights[z==1].sum())
    if min(wn,wp,zn,zp)<=0:return dict(valid=False,reason='bootstrap_missing_class')
    dec=compare_scores(w,z,raw_utility,utility,weights)
    if abs(dec['within_term']+dec['cross_term']-dec['total_damage'])>=1e-12:raise RuntimeError('BOOTSTRAP_T5_IDENTITY')
    cells=[];aucs=[];pairs=[]
    for value in (0,1):
        m=w==value;n0=float(weights[m&(z==0)].sum());n1=float(weights[m&(z==1)].sum());cells.append((n0,n1))
        if min(n0,n1)>=min_cond_class:aucs.append(_weighted_auc(z[m],np.asarray(cond)[m],weights[m]));pairs.append(n0*n1)
    condok=len(aucs)==2
    return dict(valid=True,uncond_auc=_weighted_auc(z,np.asarray(uncond),weights),uncond_pairs=zn*zp,cond_auc=float(np.average(aucs,weights=pairs)) if condok else np.nan,cond_pairs=sum(pairs) if condok else 0.,within=dec['within_term'],cross=dec['cross_term'],total=dec['total_damage'],utility_pairs=wn*wp,conditional_eligible=condok)

def aggregate_folds(rows):
    good=[r for r in rows if r.get('valid',False)]
    if not good:return dict(valid=False,folds_used=0)
    out=dict(valid=True,folds_used=len(good))
    for name,weight in [('uncond_auc','uncond_pairs'),('within','utility_pairs'),('cross','utility_pairs'),('total','utility_pairs')]:out[name]=float(np.average([r[name] for r in good],weights=[r[weight] for r in good]))
    cond=[r for r in good if r['conditional_eligible']];out['conditional_folds_used']=len(cond);out['cond_auc']=float(np.average([r['cond_auc'] for r in cond],weights=[r['cond_pairs'] for r in cond])) if cond else np.nan
    if abs(out['within']+out['cross']-out['total'])>=1e-12:raise RuntimeError('AGGREGATE_BOOTSTRAP_T5_IDENTITY')
    return out

def paired_bootstrap(folds,replicates=1000,seed=20261103,min_cond_class=30,require_all_folds=True):
    """folds: [{W,Z,identity_id,raw_utility,methods:{name:{utility,uncond,cond}}}].

    All methods share identity multiplicities per replicate/fold. Scores and
    validation-selected attacker families stay fixed. Invalid replicates are
    retained with flags, never retrained or silently replaced.
    """
    rng=np.random.default_rng(seed);names=tuple(folds[0]['methods']);results=[]
    for b in range(replicates):
        per_method={name:[] for name in names}
        for fold in folds:
            weights=fold_cluster_weights(fold['identity_id'],rng)
            for method in names:
                s=fold['methods'][method];per_method[method].append(recompute_fold(fold['W'],fold['Z'],fold['raw_utility'],s['utility'],s['uncond'],s['cond'],weights,min_cond_class))
        aggregated={name:aggregate_folds(values) for name,values in per_method.items()}
        if require_all_folds:
            for value in aggregated.values():
                if value.get('folds_used',0)!=len(folds) or value.get('conditional_folds_used',0)!=len(folds):value.update(valid=False,reason='bootstrap_required_fold_not_estimable')
        results.append(aggregated)
    return results

def eligible_five_fold_pairs(table):
    """Cross-pair summaries require all five folds, with no missing flags."""
    return [pair for pair,q in table.groupby('pair_id') if len(q)==5 and set(q.fold)==set(range(5)) and q.fold_eligible.fillna(False).all()]

def draw_Z_blocks(pair_targets,rng):
    targets=np.asarray(pair_targets);blocks=np.unique(targets);draws=rng.choice(blocks,len(blocks),replace=True)
    return np.concatenate([np.flatnonzero(targets==z) for z in draws])
