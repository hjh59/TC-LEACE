"""Freeze saved T7 outer folds and validate actual identity-level A/B splits."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_split(a, b, outer_train, outer_test, groups, split_config, tolerance=.01):
    """Check image and identity partitions, then enforce the declared A ratio."""
    expected = {"A80B20": .8, "A20B80_actual": .2}[split_config]
    a, b, tr, te = [np.asarray(q, dtype=np.int64) for q in (a, b, outer_train, outer_test)]
    groups = np.asarray(groups)
    if any(q.ndim != 1 or len(q) == 0 for q in (a, b, tr, te)):
        raise ValueError("split indices must be nonempty 1-D arrays")
    if any(len(np.unique(q)) != len(q) for q in (a, b, tr, te)):
        raise ValueError("duplicate split index")
    if min(int(q.min()) for q in (a, b, tr, te)) < 0 or max(int(q.max()) for q in (a, b, tr, te)) >= len(groups):
        raise ValueError("split index out of bounds")
    if np.intersect1d(tr, te).size or not np.array_equal(np.sort(np.r_[tr, te]), np.arange(len(groups))):
        raise ValueError("outer train/test do not partition the pre-experiment rows")
    if np.intersect1d(groups[tr], groups[te]).size:
        raise ValueError("outer identity overlap")
    if np.intersect1d(a, b).size or not np.array_equal(np.sort(np.r_[a, b]), np.sort(tr)):
        raise ValueError("A/B do not partition outer train")
    ga, gb = np.unique(groups[a]), np.unique(groups[b])
    if np.intersect1d(ga, gb).size:
        raise ValueError("A/B identity overlap")
    image_fraction, identity_fraction = len(a) / len(tr), len(ga) / (len(ga) + len(gb))
    if abs(image_fraction - expected) > tolerance or abs(identity_fraction - expected) > tolerance:
        raise ValueError(f"{split_config}: image={image_fraction:.6f}, identity={identity_fraction:.6f}; expected {expected:.2f} ± {tolerance:.2f}")
    return {"split_config": split_config, "A_image_fraction": image_fraction,
            "A_identity_fraction": identity_fraction, "n_A_images": len(a), "n_B_images": len(b),
            "n_A_identities": len(ga), "n_B_identities": len(gb)}


def freeze_actual_t7(t7_dirs, output, sample_ids=None):
    """Cross-check 4 models × 34 targets × 5 saved folds; write immutable NPZ."""
    t7_dirs = [Path(p) for p in t7_dirs]
    if len(t7_dirs) != 4:
        raise ValueError("expected four T7 model directories")
    with np.load(t7_dirs[0] / "z00_OOF.npz", allow_pickle=False) as d:
        groups = d["groups"]
    n = len(groups)
    if sample_ids is not None and len(sample_ids) != n:
        raise ValueError("sample IDs and T7 row count differ")
    folds = []
    for f in range(5):
        with np.load(t7_dirs[0] / "checkpoints" / f"z00_f{f:02d}.npz", allow_pickle=False) as d:
            folds.append(d["test_idx"].astype(np.int64))
    if not np.array_equal(np.sort(np.concatenate(folds)), np.arange(n)):
        raise ValueError("outer folds do not partition all rows")
    for f, te in enumerate(folds):
        tr = np.setdiff1d(np.arange(n), te)
        if np.intersect1d(groups[tr], groups[te]).size:
            raise ValueError(f"outer identity overlap in fold {f}")
    for model_dir in t7_dirs:
        with np.load(model_dir / "z00_OOF.npz", allow_pickle=False) as d:
            if not np.array_equal(d["groups"], groups):
                raise ValueError(f"group row order differs: {model_dir}")
        for z in range(34):
            for f in range(5):
                with np.load(model_dir / "checkpoints" / f"z{z:02d}_f{f:02d}.npz", allow_pickle=False) as d:
                    if not np.array_equal(d["test_idx"], folds[f]):
                        raise ValueError(f"outer index differs: {model_dir}, z{z}, fold{f}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        with np.load(output, allow_pickle=False) as d:
            if not np.array_equal(d["identity_id"], groups) or any(not np.array_equal(d[f"test_idx_{f}"], folds[f]) for f in range(5)):
                raise ValueError("existing frozen split has different content")
    else:
        np.savez_compressed(output, identity_id=groups, **{f"test_idx_{f}": te for f, te in enumerate(folds)},
                            **({"sample_id": np.asarray(sample_ids, dtype=str)} if sample_ids is not None else {}))
    return {"path": str(output), "sha256": sha256(output), "n_rows": n,
            "n_identities": int(np.unique(groups).size), "fold_sizes": [len(q) for q in folds],
            "cross_checked_checkpoints": 4 * 34 * 5}


def load_outer_fold(path, fold):
    with np.load(path, allow_pickle=False) as d:
        groups = d["identity_id"]
        te = d[f"test_idx_{fold}"]
    return np.setdiff1d(np.arange(len(groups)), te), te, groups
