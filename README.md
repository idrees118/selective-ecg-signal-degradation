<div align="center">

Selective ECG Classification Under Signal Degradation

Can auxiliary reliability signals identify ECG prediction errors better than output confidence?







Muhammad Idrees · Adnan Amin · Salma Azizi
Institute of Management Sciences (IMSciences), Peshawar, Pakistan

</div>

Overview

Selective classification allows a model to defer cases it considers unreliable instead of forcing a prediction on every input.

For that to be useful, the ranking score must do more than react to noisy ECGs. It must identify which individual predictions are most likely to be wrong.

This repository evaluates whether three auxiliary reliability signals — annotation support, waveform quality, and model uncertainty — can outperform the classifier's own output confidence for that task.

The study uses PTB-XL v1.0.3, a patient-disjoint evaluation protocol, a ResNet1D classifier, and controlled ECG signal degradation.

Main result

Output confidence produced the strongest selective ranking.

Across 13 evaluation conditions and 3 prespecified statistical contrasts, 38 of 39 patient-clustered bootstrap intervals favored confidence over the compared auxiliary rankings.

Model uncertainty increased as corruption became more severe, but that did not translate into better case-level error ranking.

What we tested

The study asks a simple question:

When an ECG classifier makes a mistake, which reliability score is best at placing that case near the top of the review queue?

We compared four signals:

Symbol

Signal

Meaning

Available at inference time?

C

Output confidence

How far predicted probabilities are from 0.5

Yes

L

Annotation support

Strength of the reference diagnostic annotation

No

Q

Waveform quality

Estimated artifact and signal-quality burden

Yes

U

Model uncertainty

MC-dropout disagreement across stochastic passes

Yes

We also tested four equal-weight combinations:

L + Q, L + U, Q + U, and L + Q + U.

Confidence was kept separate so it could serve as the baseline.

Study design

Component

Setting

Dataset

PTB-XL v1.0.3

ECG recordings

21,388

Diagnostic targets

5 superclasses

Training folds

1–8

Validation fold

9

Test fold

10

Test ECGs

2,158

Test patients

1,877

Classifier

ResNet1DWang

Trainable parameters

473,349

Reliability rankings

8

Corruption families

4

Corruption severities

3 per family

Total test conditions

13

Primary endpoint

AURC

Bootstrap replicates

5,000

Experimental workflow

flowchart LR
    A[PTB-XL v1.0.3] --> B[Patient-disjoint folds]
    B --> C[Train ResNet1DWang]
    C --> D[Freeze validation-tuned thresholds]
    D --> E[Compute C, L, Q, U]
    D --> F[Generate 12 corruption conditions]
    E --> G[Evaluate 8 rankings]
    F --> G
    G --> H[Risk-Coverage Curves]
    H --> I[AURC]
    I --> J[5,000-replicate patient-clustered bootstrap]

The pipeline is designed so that preprocessing, threshold selection, and test evaluation remain separated.

1. Data

We use PTB-XL v1.0.3 and follow its recommended patient-disjoint fold structure.

Split

PTB-XL folds

ECGs

Training

1–8

17,084

Validation

9

2,146

Test

10

2,158

Total

1–10

21,388

The final test set contains 2,158 ECGs from 1,877 patients.

The raw PTB-XL waveform files are not included in this repository.

2. ECG classifier

The classifier is ResNet1DWang with 473,349 trainable parameters.

Training uses:

AdamW optimization

One-Cycle learning-rate scheduling

50 epochs

patient-disjoint data partitions

normalization statistics computed from the training split only

Class-specific thresholds are tuned using the validation fold and then frozen before final test evaluation.

3. Reliability signals

Output confidence — C

Confidence measures how far each predicted class probability is from the decision boundary around 0.5, averaged across the five labels.

Annotation support — L

Annotation support measures the strength of the reference diagnostic annotation. Because it uses reference information, it is evaluated only as a retrospective research signal.

Waveform quality — Q

Waveform quality summarizes signal degradation and artifact burden.

Model uncertainty — U

Model uncertainty is estimated using MC dropout across 30 stochastic forward passes.

4. Signal degradation

Four controlled corruption families are applied to the raw waveform before normalization:

Corruption

What it simulates

Baseline wander

Low-frequency baseline drift

White noise

Broadband measurement noise

Amplitude clipping

Signal saturation / restricted dynamic range

Lead masking

Missing lead information

Each corruption is evaluated at three severity levels:

1 clean condition
+ 4 corruption families × 3 severities
= 13 total evaluation conditions

5. Selective prediction evaluation

The primary metric is Area Under the Risk-Coverage Curve (AURC) using five-label Hamming error.

Lower AURC = better error ranking

A useful ranking removes incorrect predictions earlier than correct ones.

Key results

Clean-test AURC

Rank

Reliability ranking

AURC ↓

1

Confidence (C)

0.055157

2

L + U

0.060360

3

U

0.064734

4

L + Q + U

0.079144

5

Q + U

0.090890

6

L

0.101339

7

L + Q

0.117802

8

Q

0.132054

Confidence achieved the lowest AURC on the held-out clean test set.

Held-out classifier performance

Metric

Result

Macro-AUROC

0.9265

Macro-F1

0.7448

Full-coverage Hamming risk

0.1190

Statistical comparison

Three contrasts were prespecified:

L + U      vs Confidence
U          vs Confidence
L + Q + U  vs Confidence

Across 13 conditions:

3 contrasts × 13 conditions = 39 comparisons

Each comparison used:

5,000 bootstrap replicates

paired resampling

patient-level clustering

condition-specific 95% bootstrap intervals

Result

38 of 39 intervals favored confidence.

The only exception was L + U under moderate amplitude clipping, where the interval crossed zero.

Because the intervals are dependent and were not multiplicity-adjusted, they are interpreted as condition-specific evidence, not as a family-wise hypothesis test.

What the result means

One of the most important observations is that detecting degradation is not the same as ranking errors.

Model uncertainty increased consistently as corruption severity increased across all four corruption families.

That means the uncertainty signal responded to degraded inputs at the group level.

However, its AURC remained worse than confidence.

A score can recognize that a set of ECGs has become harder without being the best score for identifying which individual predictions are wrong.

That distinction is the central result of this study.

Repository structure

selective-ecg-signal-degradation/
│
├── preprocessing/              # PTB-XL loading and split preparation
├── model/                      # ResNet1D training
├── reliability_signals/        # C, L, Q, U computation
├── corruption/                 # Controlled ECG degradation
├── evaluation/                 # Risk-coverage and AURC
├── statistics/                 # Patient-clustered bootstrap
├── results/                    # Generated numerical outputs
├── figures/                    # Generated figures
├── tests/                      # Automated checks
├── requirements.txt
├── LICENSE
└── README.md

Reproduce the study

1. Clone the repository

git clone https://github.com/idrees118/selective-ecg-signal-degradation.git
cd selective-ecg-signal-degradation

2. Download PTB-XL

Download PTB-XL v1.0.3 from PhysioNet:

https://physionet.org/content/ptb-xl/1.0.3/

Keep the raw dataset outside the repository.

3. Create an environment

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Windows:

.venv\Scripts\activate
pip install -r requirements.txt

4. Prepare PTB-XL

python preprocessing/load_ptbxl.py --data-dir /path/to/ptbxl

5. Train the classifier

python model/train.py --seed 20260826

6. Compute reliability signals

python reliability_signals/compute_all.py

7. Generate corruption conditions

python corruption/generate_all.py

8. Evaluate selective prediction

python evaluation/aurc.py

9. Run the bootstrap analysis

python statistics/bootstrap_patient_clustered.py --seed 20260826

Outputs are written to:

results/
figures/

Reproducibility safeguards

patient-disjoint PTB-XL folds

training-only normalization statistics

validation-only threshold tuning

frozen thresholds before test evaluation

predefined corruption procedures

patient-clustered bootstrap resampling

three training seeds checked for validation stability

one selected checkpoint used for final held-out evaluation

299 integrity checks passed

118 automated tests passed

Training seeds:

20260826
20260827
20260828

Reported held-out checkpoint:

20260826

Data availability

PTB-XL is distributed separately through PhysioNet and is not redistributed here.

Dataset: PTB-XL v1.0.3
https://physionet.org/content/ptb-xl/1.0.3/

The MIT license in this repository applies to the code, not to PTB-XL.

Manuscript

Selective ECG Classification Under Signal Degradation: Evaluating Auxiliary Reliability Signals Against Output Confidence

Muhammad Idrees · Adnan Amin · Salma Azizi
Institute of Management Sciences (IMSciences), Peshawar, Pakistan

Status: Manuscript under review / not yet published.

Citation

@software{idrees2026selectiveecg,
  author  = {Idrees, Muhammad and Amin, Adnan and Azizi, Salma},
  title   = {Selective ECG Classification Under Signal Degradation:
             Code and Reproduction Pipeline},
  year    = {2026},
  url     = {https://github.com/idrees118/selective-ecg-signal-degradation}
}

The article DOI and full citation will be added after publication.

License

Released under the MIT License. See LICENSE.

Contact

Muhammad Idrees
GitHub: @idrees118

Dr. Adnan Amin
Corresponding author
Institute of Management Sciences (IMSciences), Peshawar, Pakistan
Email: adnan.amin@imsciences.edu.pk
