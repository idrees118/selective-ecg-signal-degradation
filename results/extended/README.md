# Extended-analysis results

Condition-level results behind the robustness analyses in the paper.
`<dataset>` is `ptbxl` or `chapman`.

- `v1/<dataset>/seed_<seed>/posthoc/`: AURC, AUGRC and risk at 70/80/90%
  coverage for every selector (`selector_metrics.csv`); paired differences
  from confidence with 95% intervals, bootstrap p-values and Holm-adjusted
  p-values (`contrasts.csv`); classification metrics (`classification.csv`).
  PTB-XL has three seeds (20260826, 20260827, 20260828).
- `v1/<dataset>/seed_20260826/supplementary/`: any-label-error endpoint,
  Hamming-error stacker, risk at coverage with intervals, label-level
  results, quality-score components, and decision confidence.
- `v1/ptbxl/ensemble/`: three-member ensemble metrics and contrasts.
