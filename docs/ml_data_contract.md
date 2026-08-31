# ML Data Contract — PTB-XL Superdiagnostic Task

Each model example corresponds to exactly one row of the frozen cohort manifest.

Input:
- source: PTB-XL v1.0.3 records100
- native waveform: (1000 time samples, 12 leads)
- native physical unit: mV
- normalization: frozen global training-only z-score
- model layout: (12 leads, 1000 samples)
- model dtype: float32

Target:
- shape: (5,)
- dtype: float32
- multilabel binary vector
- fixed order:
  1. NORM
  2. MI
  3. STTC
  4. CD
  5. HYP

Dataset sizes:
- train: 17084
- validation: 2146
- test: 2158

No:
- augmentation
- clipping
- filtering
- per-record normalization
- per-lead normalization
- label thresholding
- split reassignment

Every loaded example must preserve:
ecg_id -> patient_id -> split -> filename -> waveform -> target
from the frozen cohort manifest.
