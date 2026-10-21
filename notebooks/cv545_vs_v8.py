"""CV545 V2S s123 OOF vs. the train_v8 models (H5_v8 = holdout5_v8_cnt12, LB 0.945) on original_208.

The train_v8 runs held out only the 208 (they trained on the 242 and the 95 with LLM labels), so the
paired comparison is on original_208 alone. Single models: CV545 V2S / C0 (s123) vs the v8 V2S (the
only one, seed 42), v8 C0 s2 / s42, v8 CNT12 s42 / s2 and the v11 V2S runs. Ensembles: H5_v8 (R50 + V2S + C0 s2 +
CNT12 x2, sigmoid mean) vs. H5_v8 with its V2S / C0 / CNT12 swapped for the CV545 ones and H5_v8 +
the CV545 models (7 or 8 members). Paired study-level bootstrap (1000).

Usage:  python notebooks/cv545_vs_v8.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

R = Path('work/runs')
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
N_BOOT = 1000
H5 = {'v8_R50': 'train_v8/R50_E3_img320_trainv8_holdout208_20261007_003442',
      'v8_V2S': 'train_v8/V2S_E3_img320_trainv8_holdout208_20261007_053337',
      'v8_C0s2': 'train_v8/C0_E3_img320_trainv8_s2_holdout208_20261010_153116',
      'v8_CNT12s42': 'train_v8/CNT12_lr4e-5_E3_img320_trainv8_holdout208_20261012_133340',
      'v8_CNT12s2': 'train_v8/CNT12_lr4e-5_E3_img320_trainv8_s2_holdout208_20261012_122058'}
V11 = {f'v11_V2S_s{s}': f'V2S_E3_img320_s{s}_trainv11_holdout450_*' for s in (123, 2, 42, 7)}


def pivot(p):
    p = p.assign(reference=p.reference.where(p.reference_valid.astype(bool)))
    return (p.pivot(index='StudyInstanceUID', columns='target', values='score')[T],
            p.pivot(index='StudyInstanceUID', columns='target', values='reference')[T])


def read(path):
    return pivot(pd.read_csv(path, dtype={'StudyInstanceUID': str}))


man = pd.read_csv('notebooks/reference_450_manifest.csv', dtype={'StudyInstanceUID': str})
ids = pd.Index(sorted(man.StudyInstanceUID[man.subset == 'original_208']))
assert len(ids) == 208

CV = {'CV545_V2S': 'V2S_E3_img320_s123_cv545soft_20261020_010445',
      'CV545_C0': 'C0_E3_img320_s123_cv545soft_20261020_065853',
      'CV545_CNT12': 'CNT12_E3_img320_s123_cv545soft_20261020_150528'}
cv = {k: read(R / 'train_v12' / f'{v}_oof' / 'oof_predictions.csv') for k, v in CV.items()}
runs = {k: read(R / v / 'validation_predictions.csv') for k, v in H5.items()}
runs['v8_C0s42'] = read(R / 'train_v8/C0_E3_img320_trainv8_holdout208_20261010_151746/validation_predictions.csv')
runs.update({k: read(sorted((R / 'train_v11').glob(v))[-1] / 'validation_predictions.csv') for k, v in V11.items()})

# All references must agree on the 208 (v8 = ref208, v11 = ref450, CV545 = ref545)
ref = runs['v8_V2S'][1].loc[ids]
for k, (s, r) in {**runs, **cv}.items():
    a, b = r.loc[ids], ref
    assert ((a == b) | (a.isna() & b.isna())).all().all(), k
    assert s.index.isin(ids).sum() == 208, k
Y = ref.to_numpy()

A = {k: s.loc[ids].to_numpy() for k, (s, _) in runs.items()}
A.update({k: s.loc[ids].to_numpy() for k, (s, _) in cv.items()})
A['v11_V2S_avg4'] = np.mean([A[k] for k in V11], axis=0)
A['CV545_V2S+C0'] = (A['CV545_V2S'] + A['CV545_C0']) / 2
A['v8_V2S+C0s2'] = (A['v8_V2S'] + A['v8_C0s2']) / 2
A['H5_v8'] = np.mean([A[k] for k in H5], axis=0)
SWAP = {'v8_V2S': 'CV545_V2S', 'v8_C0s2': 'CV545_C0'}
A['H5 V2S->CV'] = np.mean([A[SWAP[k]] if k == 'v8_V2S' else A[k] for k in H5], axis=0)
A['H5 C0->CV'] = np.mean([A[SWAP[k]] if k == 'v8_C0s2' else A[k] for k in H5], axis=0)
A['H5 V2S,C0->CV'] = np.mean([A[SWAP.get(k, k)] for k in H5], axis=0)
A['H5 + CV V2S,C0'] = np.mean([A[k] for k in H5] + [A['CV545_V2S'], A['CV545_C0']], axis=0)
A['CV545_3arch'] = (A['CV545_V2S'] + A['CV545_C0'] + A['CV545_CNT12']) / 3
A['H5 + CV 3arch'] = np.mean([A[k] for k in H5] + [A['CV545_V2S'], A['CV545_C0'], A['CV545_CNT12']], axis=0)
# H5 composition from CV545 models: v8 R50 + CV V2S + CV C0 + CV CNT12 (weight 2, like the two CNT12 seeds)
SWAP.update({'v8_CNT12s42': 'CV545_CNT12', 'v8_CNT12s2': 'CV545_CNT12'})
A['H5 all->CV'] = np.mean([A[SWAP.get(k, k)] for k in H5], axis=0)


def auc_vec(s, idx):
    out = np.full(len(T), np.nan)
    for j in range(len(T)):
        y, x = Y[idx, j], s[idx, j]
        m = np.isin(y, [0, 1])
        y, x = y[m], x[m]
        npos, nneg = (y == 1).sum(), (y == 0).sum()
        if npos and nneg:
            out[j] = (rankdata(x)[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return out


ALL = np.arange(208)
print('== Macro ROC-AUC on original_208 ==')
for k, s in A.items():
    print(f'  {k:16s} {np.nanmean(auc_vec(s, ALL)):.4f}')

rng = np.random.default_rng(0)
boots = [rng.choice(ALL, 208) for _ in range(N_BOOT)]
print('\n== Paired bootstrap on original_208 (a - b: mean [2.5%, 97.5%], P(a > b)) ==')
for a, b in [('CV545_V2S', 'v8_V2S'), ('CV545_V2S', 'v11_V2S_s123'), ('CV545_C0', 'v8_C0s2'),
             ('CV545_C0', 'v8_C0s42'), ('CV545_V2S+C0', 'v8_V2S+C0s2'), ('CV545_V2S+C0', 'H5_v8'),
             ('H5 V2S->CV', 'H5_v8'), ('H5 C0->CV', 'H5_v8'), ('H5 V2S,C0->CV', 'H5_v8'),
             ('H5 + CV V2S,C0', 'H5_v8'), ('CV545_CNT12', 'v8_CNT12s42'), ('CV545_CNT12', 'v8_CNT12s2'),
             ('CV545_3arch', 'H5_v8'), ('H5 all->CV', 'H5_v8'), ('H5 + CV 3arch', 'H5_v8'),
             ('H5 + CV 3arch', 'H5 + CV V2S,C0')]:
    d = np.array([np.nanmean(auc_vec(A[a], b_)) - np.nanmean(auc_vec(A[b], b_)) for b_ in boots])
    print(f'  {a:16s} - {b:14s}: {d.mean():+.4f} [{np.percentile(d, 2.5):+.4f}, '
          f'{np.percentile(d, 97.5):+.4f}]  P>0 = {(d > 0).mean():.3f}')

print('\n== Per-target ROC-AUC on original_208 ==')
cols = ['CV545_CNT12', 'v8_CNT12s42', 'v8_CNT12s2', 'CV545_3arch', 'H5_v8', 'H5 + CV V2S,C0', 'H5 + CV 3arch']
pt = pd.DataFrame({k: auc_vec(A[k], ALL) for k in cols}, index=T)
pt['dH5+3'] = pt['H5 + CV 3arch'] - pt.H5_v8
pt['n_pos'] = (Y == 1).sum(0)
print(pt.round(4).to_string())
