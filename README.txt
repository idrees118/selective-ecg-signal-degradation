PTB-XL reliability manuscript v3

Purpose
-------
This version restructures the paper around one clear question:
Do annotation quality, waveform quality, and model uncertainty improve selective ECG prediction beyond ordinary confidence?

What changed from the previous draft
------------------------------------
1. Stronger, simpler title and abstract.
2. The problem and research question are explicit in the Introduction.
3. No displayed mathematical equations are used.
4. Four manuscript figures are included immediately:
   - study_design
   - reliability_signals
   - seed_reproducibility
   - bootstrap_forest_all39
5. Two additional result figures are generated from the audited local results:
   - corruption_hamming_risk
   - corruption_mean_uncertainty
6. The complete 39 bootstrap comparisons are retained in an appendix.
7. Exact clean fold-10 values and classwise/condition tables are imported from audited reporting files, not typed manually.
8. No Unicode em dash or en dash is used in the manuscript source.

How to use inside the audited ptbxl_reliability project
-------------------------------------------------------
Copy manuscript_v3.tex, populate_manuscript_results.py, and the supplied figures/ files into the project root / figures folder.

Then, from the project root, run:

    python populate_manuscript_results.py

The script first requires logs/final_integrity_audit.json to be a complete PASS. It then reads only audited reporting artifacts and creates:

    paper_values.tex
    generated_classwise_table.tex
    generated_condition_table.tex
    figures/corruption_hamming_risk.pdf
    figures/corruption_hamming_risk.png
    figures/corruption_mean_uncertainty.pdf
    figures/corruption_mean_uncertainty.png

It performs no training, inference, tuning, corruption generation, or bootstrap rerun.

Compile with:

    latexmk -pdf manuscript_v3.tex

Before submission
-----------------
Replace the author/affiliation placeholder, funding statement, author contributions, code archive URL/DOI, and any journal-specific declarations.
