import numpy as np

def permute_in_B(z,w,B,seed):
    out=np.asarray(z).copy();rng=np.random.default_rng(seed)
    for value in (0,1):
        ix=B[np.asarray(w)[B]==value];out[ix]=rng.permutation(out[ix])
    return out

