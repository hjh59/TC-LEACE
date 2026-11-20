"""Rank-at-most-two conditional erasure in the existing LEACE covariance metric.

With centered X, Sigma = X.T @ X / n + 1e-6 I (the T7 LEACE rule).
For c_w = Cov(X,Z|W=w), form U = orth([Sigma^-1/2 c_0,
Sigma^-1/2 c_1]). The applied map is X' = X - Sigma^1/2 U U.T
Sigma^-1/2 (X - mean(X)). A zero c_w contributes no column. If both
conditional covariances vanish, the map is identity (rank 0). If Z is
constant in an observed W stratum, its empirical conditional covariance is
exactly zero and that stratum is recorded as degenerate. An absent or
singleton W stratum remains unestimable and raises an error.
"""
import numpy as np


def leace_covariance(x, ridge=1e-6):
    """Match the float64 covariance and ridge in gpu_leace."""
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean(0)
    return xc.T @ xc / len(x) + ridge * np.eye(x.shape[1])


def fit_cc2(x, z, w, covariance=None, eigenvalue_cutoff=0.0,
            angle_tolerance=1e-6, residual_tolerance=1e-8):
    x = np.asarray(x, dtype=np.float64)
    z, w = np.asarray(z, dtype=np.float64), np.asarray(w)
    cov = leace_covariance(x) if covariance is None else np.asarray(covariance, dtype=np.float64)
    if x.ndim != 2 or z.shape != (len(x),) or w.shape != (len(x),):
        raise ValueError("incompatible x, z, w shapes")
    if cov.shape != (x.shape[1], x.shape[1]):
        raise ValueError("covariance shape mismatch")
    vals, vecs = np.linalg.eigh((cov + cov.T) / 2)
    keep = vals > eigenvalue_cutoff
    if not keep.any():
        raise ValueError("covariance has no retained directions")
    v, d = vecs[:, keep], vals[keep]
    wh = (v / np.sqrt(d)) @ v.T
    wh_plus = (v * np.sqrt(d)) @ v.T
    cs = []
    degenerate_groups = []
    for group in (0, 1):
        mask = w == group
        if mask.sum() < 2:
            raise ValueError(f"W={group} lacks target variation")
        xx, zz = x[mask], z[mask]
        if len(np.unique(zz)) < 2:
            cs.append(np.zeros(x.shape[1], dtype=np.float64))
            degenerate_groups.append(group)
        else:
            cs.append(((xx - xx.mean(0)).T @ (zz - zz.mean())) / len(zz))
    whitened = np.column_stack([wh @ c for c in cs])
    norms = np.linalg.norm(whitened, axis=0)
    active = norms > np.finfo(float).eps * max(1.0, np.linalg.norm(x))
    angle = float("nan")
    near_collinear = False
    if active.all():
        cosine = float(np.clip(abs(np.dot(whitened[:, 0], whitened[:, 1])) / np.prod(norms), 0, 1))
        angle = float(np.arccos(cosine))
        near_collinear = angle < angle_tolerance
    if not active.any():
        u = np.empty((x.shape[1], 0))
        rank = 0
    else:
        basis, singular, _ = np.linalg.svd(whitened[:, active], full_matrices=False)
        # Small covariance norms alone do not justify discarding a direction.
        # With two active directions rank 1 is a candidate only when their
        # angle satisfies the protocol; the residual gate can still require 2.
        candidates = [1] if active.sum() == 1 else ([1, 2] if near_collinear else [2])
        rank = 0
        for candidate in candidates:
            uc = basis[:, :candidate]
            rem = wh_plus @ uc @ uc.T @ wh
            worst = max(np.linalg.norm(c - rem @ c) for c in cs)
            if worst < residual_tolerance:
                rank = candidate
                break
        if rank == 0:
            raise ValueError("covariance directions outside retained whitening support")
        u = basis[:, :rank]
    removal = wh_plus @ u @ u.T @ wh
    residuals = [float(np.linalg.norm(c - removal @ c)) for c in cs]
    return {"mean": x.mean(0), "removal": removal, "rank": rank,
            "angle_radians": angle, "near_collinear": near_collinear,
            "degenerate_groups": degenerate_groups,
            "conditional_covariance_norms": norms.tolist(),
            "residual_norms": residuals, "cutoff": float(eigenvalue_cutoff)}


def transform_cc2(x, fitted):
    x = np.asarray(x, dtype=np.float64)
    return x - (x - fitted["mean"]) @ fitted["removal"].T
