# Extended analyses

Additional selective-prediction analyses on top of the v1 pipeline. No v1
artifact is modified. Everything is written under
`results/extended/v1/<dataset>/`.

| Analysis | Purpose | Script |
|---|---|---|
| Confidence **plus** auxiliary scores (C+U, C+Q, C+Q+U, logistic stacker) | Auxiliary scores combined with confidence, not only as replacements | `run_posthoc_selectors.py` |
| AUGRC next to AURC | Metric recommended by Traub et al. (2024) | `run_posthoc_selectors.py` |
| Risk at 70/80/90% coverage | Fixed operating points | `run_posthoc_selectors.py` |
| Temperature-scaled and threshold-margin confidence | Confidence that accounts for calibration and class thresholds | `run_posthoc_selectors.py` |
| Published SQI (bSQI from two QRS detectors) | Established quality index independent of the study-specific Q | `compute_published_sqi.py` |
| Nested lead masks (1 ⊂ 3 ⊂ 6) | Monotone lead-loss severity | `run_extended_inference.py` |
| Recorded MIT-BIH NSTDB noise (bw, em, ma) | Real recorded noise in addition to synthetic corruptions | `run_extended_inference.py` |
| PTB-XL seeds 20260827 and 20260828 on test | Seed variability | `run_extended_inference.py` |

All fitted quantities (thresholds for the new seeds, temperatures, percentile
references, stacker weights) use clean validation only.

## Setup

```bash
pip install neurokit2==0.2.13          # QRS detectors for bSQI
python -c "import wfdb; wfdb.dl_database('nstdb', 'data/raw/nstdb/1.0.0', records=['bw','em','ma'])"
```

The PTB-XL raw files, frozen cohort, normalization and Q training reference
must be in their usual `data/` locations; the Chapman cache and
`chapman_q_training_reference.npz` must be in
`results/external_validation/chapman/v1/` (as produced by the v1 pipeline).

## Run order

```bash
# 1. PTB-XL, primary seed: frozen test artifacts are reused, only the
#    clean validation and the new conditions are computed.
python scripts/extended/run_extended_inference.py --dataset ptbxl \
    --seed 20260826 --conditions validation nested recorded

# 2. PTB-XL, the two extra seeds (thresholds fitted on their validation).
python scripts/extended/run_extended_inference.py --dataset ptbxl \
    --seed 20260827 --conditions validation frozen nested recorded
python scripts/extended/run_extended_inference.py --dataset ptbxl \
    --seed 20260828 --conditions validation frozen nested recorded

# 3. Chapman-Shaoxing (primary model only).
python scripts/extended/run_extended_inference.py --dataset chapman \
    --conditions validation nested recorded

# 4. Published SQI (model-independent, once per dataset).
python scripts/extended/compute_published_sqi.py --dataset ptbxl \
    --conditions validation frozen nested recorded --workers 8
python scripts/extended/compute_published_sqi.py --dataset chapman \
    --conditions validation frozen nested recorded --workers 8

# 5. Post-hoc selectors, metrics and bootstrap (no inference).
for seed in 20260826 20260827 20260828; do
  python scripts/extended/run_posthoc_selectors.py --dataset ptbxl \
      --seed $seed --conditions frozen nested recorded
done
python scripts/extended/run_posthoc_selectors.py --dataset chapman \
    --conditions frozen nested recorded
```

Every step skips artifacts that already exist, so an interrupted run can be
restarted with the same command.

Rough cost: each test condition needs one deterministic pass and 30
MC-dropout passes. Steps 1 to 3 evaluate about 100 test conditions in
total, about the same per-condition cost as the v1 final evaluation.
Step 4 is CPU-bound (about 0.1 s per ECG per condition per worker).

## Sanity check

For seed 20260826 on PTB-XL, the post-hoc clean-test AURC printed for
`confidence`, `U`, `Q` and `Q+U` must equal the frozen v1 values
(0.0552, 0.0647, 0.1321, 0.0909); for Chapman they are 0.0171, 0.0203,
0.0643 and 0.0301. The primary seed reuses the frozen test artifacts and
the frozen Q/U percentile references, so any difference means a file is
out of place.

## Outputs

Summary tables are written to:

```
results/extended/v1/ptbxl/seed_20260826/posthoc/
results/extended/v1/ptbxl/seed_20260827/posthoc/
results/extended/v1/ptbxl/seed_20260828/posthoc/
results/extended/v1/chapman/seed_20260826/posthoc/
```

Each holds `selector_metrics.csv` (AURC, AUGRC, risk at coverage per
selector and condition), `contrasts.csv` (paired-bootstrap deltas against
confidence with Holm-adjusted p-values), `classification.csv` and
`summary.json` (temperatures, stacker coefficients, thresholds).
