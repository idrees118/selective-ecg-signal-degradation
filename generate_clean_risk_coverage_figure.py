#!/usr/bin/env python3
"""Generate a descriptive clean fold-10 risk-coverage figure from saved frozen artifacts.

Run from the ptbxl_reliability project root with the project's virtual environment active.
No training, inference, tuning, threshold selection, corruption generation, or bootstrap is performed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ptbxl_reliability.selective_prediction import grouped_risk_coverage, midrank_ecdf

ROOT = Path.cwd()
CLEAN = ROOT / 'results/final_evaluation/v1/test/clean/results.npz'
L_PATH = ROOT / 'results/final_evaluation/v1/test/test_annotation_confidence.csv'
REF = ROOT / 'results/selective_prediction_waveform_q/v1/validation/combination_reference.npz'
OUT_PDF = ROOT / 'fig3_risk_coverage_clean.pdf'
OUT_PNG = ROOT / 'fig3_risk_coverage_clean.png'
EXPECTED = {'Confidence':0.055157,'L+U':0.060360,'U':0.064734,'L+Q+U':0.079144}

def xy(curve):
    if 'coverage' not in curve or 'risk' not in curve:
        raise KeyError(f"Unexpected grouped_risk_coverage keys: {sorted(curve.keys())}")
    return np.asarray(curve['coverage'],float), np.asarray(curve['risk'],float)

with np.load(CLEAN, allow_pickle=False) as d:
    ids=np.asarray(d['ecg_ids'],np.int64)
    loss=np.asarray(d['hamming_error'],float)
    c=np.asarray(d['confidence'],float)
    q=np.asarray(d['Q_signal'],float)
    u=np.asarray(d['U'],float)
ldf=pd.read_csv(L_PATH,float_precision='round_trip')
if 'ecg_id' in ldf and not np.array_equal(ldf['ecg_id'].to_numpy(np.int64),ids):
    raise AssertionError('L ECG ordering mismatch')
l=ldf['label_confidence'].to_numpy(float)
with np.load(REF,allow_pickle=False) as ref:
    lp=midrank_ecdf(np.asarray(ref['L_badness_reference'],float),1-l)
    qp=midrank_ecdf(np.asarray(ref['Q_signal_badness_reference'],float),1-q)
    up=midrank_ecdf(np.asarray(ref['U_badness_reference'],float),u)
selectors={'Confidence':c,'L+U':-(lp+up)/2,'U':-u,'L+Q+U':-(lp+qp+up)/3}
curves={n:grouped_risk_coverage(s,loss) for n,s in selectors.items()}
for n,curve in curves.items():
    got=float(curve['aurc'])
    if round(got,6)!=round(EXPECTED[n],6):
        raise AssertionError(f'{n}: reconstructed AURC {got:.12f} != reported {EXPECTED[n]:.6f}')
    print(f'{n:12s} AURC={got:.12f} PASS')
plt.rcParams.update({'font.size':8.5,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8})
fig,ax=plt.subplots(figsize=(3.5,2.7))
for n,curve in curves.items():
    cov,risk=xy(curve)
    ax.step(cov,risk,where='post',linewidth=1.25,label=n)
ax.set_xlabel('Coverage (fraction of ECGs retained)')
ax.set_ylabel('Hamming risk among retained ECGs')
ax.set_xlim(0,1)
ax.grid(True,alpha=.18,linewidth=.5)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT_PDF,bbox_inches='tight')
fig.savefig(OUT_PNG,dpi=300,bbox_inches='tight')
print('Wrote',OUT_PDF)
print('Wrote',OUT_PNG)
