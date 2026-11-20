"""POST HOC – not preregistered. V2.2 release fitting membership guard."""
import numpy as np
PRED_SHA='b2aee9067310a660c737a957a0fa0ff632d572898d7390cb0650f582043ce576'
BIASED='biased design (attacker auxiliary data included in eraser fit); superseded by S-rel2'
def fit_indices(parts,n):
 excluded=np.concatenate([parts['B_train'],parts['B_val']]);ix=np.setdiff1d(np.arange(n),excluded)
 expected=np.concatenate([parts['A_train'],parts['A_val'],parts['test']])
 if len(np.unique(expected))!=len(expected) or len(np.unique(excluded))!=len(excluded) or not np.array_equal(ix,np.sort(expected)):raise RuntimeError('S_REL2_FIT_MEMBERSHIP')
 if np.intersect1d(ix,excluded).size:raise RuntimeError('S_REL2_B_LEAKAGE')
 return ix
