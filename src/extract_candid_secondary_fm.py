#!/usr/bin/env python3
"""Label-blind, resumable DINOv2/BiomedCLIP extraction for CANDID-PTX.

Reads only SOP, filename, and image_path columns from the prepared manifest.
BiomedCLIP uses the final pre-projection visual CLS feature, matching the
project's T2.2 representation definition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/case_studies/candid_ptx"
MANIFEST = BASE / "manifest.csv"
MODEL_ROOT = REPO / "results/stage2/model_cache"
OUTPUT_ROOT = BASE / "embeddings"
BATCH_SIZE = 32
CHUNK_SIZE = 128
QC_N = 500
QC_SEED = 20261026
SPECS = {
    "dinov2": {
        "repo": "facebook/dinov2-base",
        "revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        "path": MODEL_ROOT / "dinov2-base",
        "dimension": 768,
        "input_size": 224,
        "feature": "AutoImageProcessor; final last_hidden_state CLS (768-D)",
    },
    "biomedclip": {
        "repo": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "revision": "9f341de24bfb00180f1b847274256e9b65a3a32e",
        "path": MODEL_ROOT / "biomedclip",
        "dimension": 768,
        "input_size": 224,
        "feature": "OpenCLIP visual.trunk.forward_features[:,0,:], before learned image projection (768-D)",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> pd.DataFrame:
    # Explicit usecols prevents loading PTX, tube, split, or other label columns.
    manifest = pd.read_csv(MANIFEST, usecols=["SOPInstanceUID", "filename", "image_path"])
    manifest = manifest.sort_values("SOPInstanceUID", kind="stable").reset_index(drop=True)
    if manifest.SOPInstanceUID.isna().any() or manifest.SOPInstanceUID.duplicated().any():
        raise RuntimeError("CANDID manifest must have unique, non-null SOP IDs")
    if manifest.image_path.isna().any() or manifest.image_path.duplicated().any():
        raise RuntimeError("CANDID manifest has missing or duplicate image paths")
    missing = [str(p) for p in manifest.image_path if not Path(p).is_file()]
    if missing:
        raise RuntimeError(f"Missing source images: {len(missing)}; first={missing[0]}")
    return manifest


def load_encoder(name: str, device: torch.device):
    spec = SPECS[name]
    if name == "dinov2":
        processor = AutoImageProcessor.from_pretrained(spec["path"], local_files_only=True)
        model = AutoModel.from_pretrained(spec["path"], local_files_only=True).eval().to(device)

        def encode(images: list[Image.Image]) -> np.ndarray:
            pixels = processor(images=images, return_tensors="pt").pixel_values.to(device)
            with torch.inference_mode():
                return model(pixel_values=pixels).last_hidden_state[:, 0, :].float().cpu().numpy()

        return model, encode

    import open_clip

    model, preprocess = open_clip.create_model_from_pretrained(f"local-dir:{spec['path']}")
    model = model.eval().to(device)
    if not hasattr(model.visual, "trunk") or not hasattr(model.visual.trunk, "forward_features"):
        raise RuntimeError("Cannot verify BiomedCLIP pre-projection visual feature path")

    def encode(images: list[Image.Image]) -> np.ndarray:
        pixels = torch.stack([preprocess(im) for im in images]).to(device)
        with torch.inference_mode():
            tokens = model.visual.trunk.forward_features(pixels)
        if tokens.ndim != 3 or tokens.shape[-1] != spec["dimension"]:
            raise RuntimeError(f"Unexpected BiomedCLIP token shape: {tuple(tokens.shape)}")
        return tokens[:, 0, :].float().cpu().numpy()

    return model, encode


def extract(name: str, device_name: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CANDID secondary-FM extraction")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    spec = SPECS[name]
    manifest = load_manifest()
    out = OUTPUT_ROOT / name
    chunks = out / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    model, encode = load_encoder(name, device)
    n = len(manifest)
    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    started = time.time()

    for ci in range(n_chunks):
        lo = ci * CHUNK_SIZE
        hi = min(n, lo + CHUNK_SIZE)
        chunk_path = chunks / f"chunk_{ci:04d}.npy"
        if chunk_path.exists():
            arr = np.load(chunk_path, mmap_mode="r")
            if arr.shape == (hi - lo, spec["dimension"]) and arr.dtype == np.float32 and np.isfinite(arr).all():
                continue
            chunk_path.unlink()

        values: list[np.ndarray] = []
        for begin in range(lo, hi, BATCH_SIZE):
            paths = manifest.image_path.iloc[begin:min(begin + BATCH_SIZE, hi)]
            images = []
            for path in paths:
                with Image.open(path) as im:
                    images.append(im.convert("RGB"))
            vectors = encode(images)
            if vectors.shape != (len(images), spec["dimension"]) or not np.isfinite(vectors).all():
                raise RuntimeError(f"Invalid embedding batch at chunk {ci}: {vectors.shape}")
            values.extend(vectors.astype(np.float32, copy=False))

        arr = np.stack(values).astype(np.float32, copy=False)
        if arr.shape != (hi - lo, spec["dimension"]):
            raise RuntimeError(f"Invalid output chunk {ci}: {arr.shape}")
        tmp = chunk_path.with_suffix(".tmp.npy")
        np.save(tmp, arr)
        os.replace(tmp, chunk_path)
        print(f"{name} chunk {ci + 1}/{n_chunks} rows={hi}/{n}", flush=True)

    torch.cuda.synchronize(device)
    embedding_path = out / "embeddings.npy"
    matrix = np.lib.format.open_memmap(
        embedding_path, mode="w+", dtype=np.float32, shape=(n, spec["dimension"])
    )
    for ci in range(n_chunks):
        lo = ci * CHUNK_SIZE
        part = np.load(chunks / f"chunk_{ci:04d}.npy", mmap_mode="r")
        matrix[lo:lo + len(part)] = part
    matrix.flush()
    del matrix

    embeddings = np.load(embedding_path, mmap_mode="r")
    rng = np.random.default_rng(QC_SEED)
    qc_ids = np.sort(rng.choice(n, size=min(QC_N, n), replace=False))
    repeated = []
    for begin in range(0, len(qc_ids), BATCH_SIZE):
        ids = qc_ids[begin:begin + BATCH_SIZE]
        images = []
        for ix in ids:
            with Image.open(manifest.image_path.iloc[int(ix)]) as im:
                images.append(im.convert("RGB"))
        repeated.extend(encode(images))
    repeated = np.asarray(repeated, dtype=np.float64)
    original = np.asarray(embeddings[qc_ids], dtype=np.float64)
    cosines = np.sum(original * repeated, axis=1) / (
        np.linalg.norm(original, axis=1) * np.linalg.norm(repeated, axis=1)
    )
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Consolidated embedding array contains NaN/Inf")

    weight_hashes = {
        str(path.relative_to(spec["path"])): sha256_file(path)
        for path in spec["path"].rglob("*")
        if path.is_file() and path.suffix in {".bin", ".safetensors"}
    }
    qc = {
        "dataset": "CANDID-PTX",
        "model_repo": spec["repo"],
        "model_revision": spec["revision"],
        "input_size": spec["input_size"],
        "feature": spec["feature"],
        "pre_projection_feature": name == "biomedclip",
        "n_images": n,
        "dimension": spec["dimension"],
        "dtype": "float32",
        "nan": int(np.isnan(embeddings).sum()),
        "inf": int(np.isinf(embeddings).sum()),
        "norm_min": float(np.linalg.norm(embeddings, axis=1).min()),
        "norm_median": float(np.median(np.linalg.norm(embeddings, axis=1))),
        "norm_max": float(np.linalg.norm(embeddings, axis=1).max()),
        "repeatability_n": len(qc_ids),
        "repeatability_cosine_median": float(np.median(cosines)),
        "repeatability_cosine_min": float(np.min(cosines)),
        "repeatability_pass_median_ge_0p999": bool(np.median(cosines) >= 0.999),
        "embedding_sha256": sha256_file(embedding_path),
        "weight_sha256": weight_hashes,
        "manifest_key_sha256": hashlib.sha256(
            "\n".join(manifest.SOPInstanceUID.astype(str)).encode()
        ).hexdigest(),
        "outcome_columns_read": False,
        "elapsed_seconds": time.time() - started,
    }
    (out / "repeatability_qc_500.csv").write_text(
        pd.DataFrame({"manifest_index": qc_ids, "cosine_similarity": cosines}).to_csv(index=False)
    )
    (out / "qc.json").write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n")
    print(json.dumps(qc, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(SPECS), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    extract(args.model, args.device)


if __name__ == "__main__":
    main()
