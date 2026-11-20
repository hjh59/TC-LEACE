# Anonymous SP27 artifact

This repository accompanies an evaluation of concept erasure, conditional attribute inference, and exact within-/cross-stratum AUROC damage decomposition. Methods are raw representations, LEACE and TC-LEACE (`leace_cc2`), with separately identified post-hoc SPLINCE and Weber comparisons.

## Layout
- `src/`: original selected extraction, erasure, reader, statistics, and test scripts, with path-only anonymization.
- `results/`: aggregate tables, aggregate bootstrap arrays, and feature checksums only.
- `figs/`: existing plot input CSVs. The IEEE figure script is excluded.
- `splits/`: checksums only. The identity-only CelebA mapping is not included.

## Data access and reproduction
Obtain CelebA from its official project website and comply with its research-only terms. CANDID-PTX requires a data-use license; no images, attributes, patient identifiers, embeddings, trained readers, or erasers are distributed.
The intended order is feature extraction, main runs, statistics, then tables and figures. Set data paths in the anonymized configuration, retain the source checksums, and use the original data manifest ordering. Main runs use five identity/patient folds and A80/B20 with validation-only model selection. All AUROCs and decompositions are computed within folds and combined by relevant positive-negative pair counts.
Required legacy data-loader and permutation helpers were extracted verbatim; unrelated legacy analysis is excluded. Runtime scripts also retain Git provenance calls although Git history is deliberately excluded.

For CANDID splits, acquire authorized data and the original manifest ordering, regenerate the specified patient folds with the frozen seed policy, and compare each file with `splits/HASHES.txt`. Exact byte-level split hashes also depend on the original manifest order and NPZ serialization; hashes alone do not reconstruct a missing private manifest.

## Paper artifact map
| Figure/table | Input | Relevant fields |
|---|---|---|
| F1 | `figs/f1_audit.csv` | insample_linear_auc, heldout_linear_auc, heldout_mlp_auc |
| F2a | `figs/pairs.csv` | abs_beta, uncond_auc_leace, cond_auc_leace, diff_uncond_cond |
| F2b-d, F3, F4 | `figs/tertiles.csv` | tertile, metric, est, lo, hi, n_valid |
| CelebA primary tests | `results/case_studies/celeba/analysis/sp27_statistics/CELEBA_HYPOTHESES.csv` | estimate, p_value, p_holm |
| CANDID table | `results/case_studies/candid_ptx/sp27/statistics/CANDID_ESTIMATES_WITH_CI.csv` | method, metric, estimate, ci_low, ci_high |
| Structure approximation | `results/case_studies/celeba/analysis/sp27_statistics/TABLE2_STRUCTURAL.csv` | See original column headers |
| Post-hoc comparisons | `results/case_studies/candid_ptx/sp27_posthoc_v2_2/COMPARISONS_WITH_CI.csv` | quantity, estimate, ci95_low, ci95_high |


## Verification and known limitations
Run `python verify_release.py` for read-only aggregate numerical checks, including centered-bootstrap and Holm recomputation from saved bootstrap arrays. No experiments are executed by this verifier.
Historical extraction software versions were not recorded. Re-extraction can produce small numerical differences (approximately 1e-4 relative, image/model dependent). SigLIP uses learned attention-pooled `pooler_output`, not mean patch tokens. CLIP/BiomedCLIP use pre-projection CLS features.

## Compute
Recorded main allocations were one H100 GPU, 20 vCPUs and 200 GiB per model job; CelebA used four model jobs. The 5,100 method rows have 71.012 aggregate timed hours (50.126 seconds per method row); this is not parallel wall time. The CPU-only CelebA bootstrap used 40 vCPUs/200 GiB and 32 workers, taking 37.84 minutes. 

## Release status
All 12 selected unit-test scripts pass, and the aggregate verifier passes. The identity-only CelebA mapping remains outside the archive pending separate approval. 