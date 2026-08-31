#!/usr/bin/env python3
"""Read-only helper for entropy, dropout, optimizer and scheduler sign-off."""
from pathlib import Path
import re
ROOT=Path.cwd()
paths={
 'uncertainty':ROOT/'src/ptbxl_reliability/uncertainty.py',
 'model':ROOT/'src/ptbxl_reliability/model.py',
 'training':ROOT/'scripts/train_baseline.py',
 'config':ROOT/'configs/baseline_v1.yaml',
}
for name,p in paths.items():
    if not p.exists(): raise FileNotFoundError(f'Missing expected {name}: {p}')
print('\n=== ENTROPY / U IMPLEMENTATION ===')
u=paths['uncertainty'].read_text()
for i,line in enumerate(u.splitlines(),1):
    if re.search(r'log2|log\s*\(|clip|entropy|mutual_information|mean_mutual',line,re.I): print(f'{i:4d}: {line}')
if re.search(r'\b(?:np|torch)\.log2\s*\(',u): print('AUTO-DETECT: base-2 logarithm -> bits, after checking surrounding formula.')
elif re.search(r'\b(?:np|torch)\.log\s*\(',u): print('AUTO-DETECT: natural logarithm -> nats, after checking surrounding formula.')
else: print('AUTO-DETECT: logarithm base unresolved. Inspect source above; do not add units yet.')
print('\n=== MODEL DROPOUT ===')
m=paths['model'].read_text()
for i,line in enumerate(m.splitlines(),1):
    if re.search(r'dropout',line,re.I): print(f'{i:4d}: {line}')
print('\n=== TRAINING / OPTIMIZER / SCHEDULER ===')
t=paths['training'].read_text()
for i,line in enumerate(t.splitlines(),1):
    if re.search(r'AdamW|OneCycle|pct_start|anneal_strategy|div_factor|final_div_factor|base_momentum|max_momentum|weight_decay|max_lr|betas|eps|epoch|batch',line,re.I): print(f'{i:4d}: {line}')
print('\n=== FROZEN BASELINE CONFIG ===')
print(paths['config'].read_text())
print('\nRead-only check complete; no project files modified.')
