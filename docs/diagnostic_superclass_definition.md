# PTB-XL Diagnostic Superclass Target Definition

Dataset: PTB-XL v1.0.3
Task: Five diagnostic superclasses
Task type: Multi-label classification

Fixed output order:

1. NORM — Normal ECG
2. MI — Myocardial Infarction
3. STTC — ST/T Change
4. CD — Conduction Disturbance
5. HYP — Hypertrophy

## Mathematical definition

For ECG record i, let C_i be the set of SCP codes present in its
`scp_codes` dictionary.

Let g(c) be the diagnostic superclass assigned to SCP code c in
`scp_statements.csv`, defined only for statements where `diagnostic == 1`.

For superclass k:

    y_i,k = 1  if there exists c in C_i such that g(c) = k
            0  otherwise

Therefore each ECG receives a binary vector:

    y_i = [y_NORM, y_MI, y_STTC, y_CD, y_HYP]

Multiple entries may equal 1 because PTB-XL is a multi-label diagnostic task.

## SCP likelihood handling

The numerical likelihood values stored inside `scp_codes` are NOT used to
decide whether a diagnostic label is present.

No likelihood threshold is applied.

This follows the PTB-XL PhysioNet example and the PTB-XL benchmark
superdiagnostic aggregation procedure, both of which aggregate based on
diagnostic SCP-code presence.

This is particularly important because PTB-XL documents likelihood 0 as also
being used when likelihood information is unknown.

The likelihood values will be preserved separately for the later annotation-
reliability analysis. They must not silently redefine the diagnostic target.

## Inclusion rule

An ECG is eligible for the five-superclass diagnostic modelling task only if:

    sum_k y_i,k >= 1

Records containing no diagnostic SCP code mapped to one of the five
superclasses are not converted into an artificial "all negative" diagnostic
example. Their number and fold distribution must be reported explicitly.

## Data splitting

Among eligible records:

- folds 1–8: training
- fold 9: validation
- fold 10: test

The original PTB-XL patient-level fold assignments are preserved.
