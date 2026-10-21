"""CV545 V2S "oacal" (site-group LLM OA labels calibrated to the radiologist) vs the CV545 V2S "soft" run.

Same seed (123), same folds, same config; only the OA labels of the site group's LLM rows differ
(notebooks/build_ref545_oacal.py). Paired OOF comparison on full_545 / full_450 / original_208 /
new_242 / new_95 and on the site group vs the other sites; single V2S and the cv545_v12 ensemble with
its V2S swapped (V2S + C0 + CNT12).

Success (pre-registered): OA mean better on full_545, original_208 and new_242 (same sign), macro not
worse -> LB test with the V2S folds of cv545_v12 swapped.

Usage:  python notebooks/cv545_oacal_eval.py [<oacal run-group name>]   (default: newest V2S*cv545oacal*)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

R = Path('work/runs/train_v12')
U = 'StudyInstanceUID'
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
OA = ['Medial OA', 'Lateral OA', 'PF OA']
OAI = [T.index(t) for t in OA]
SOFT = 'V2S_E3_img320_s123_cv545soft_20261020_010445'
OTHERS = ['C0_E3_img320_s123_cv545soft_20261020_065853', 'CNT12_E3_img320_s123_cv545soft_20261020_150528']
if len(sys.argv) > 1:
    CAL = sys.argv[1]
else:
    CAL = sorted(p.name[:-4] for p in R.glob('V2S*cv545oacal*_oof'))[-1]


def pivot(path):
    p = pd.read_csv(path, dtype={U: str})
    p = p.assign(reference=p.reference.where(p.reference_valid.astype(bool)))
    return (p.pivot(index=U, columns='target', values='score')[T], p.pivot(index=U, columns='target', values='reference')[T])


man = pd.read_csv('notebooks/reference_545_manifest.csv', dtype={U: str}).set_index(U)
ids = pd.Index(sorted(man.index))
site = set(pd.read_csv('notebooks/oa_site_group_studies.csv', dtype=str)[U])
runs = {k: pivot(R / f'{r}_oof' / 'oof_predictions.csv') for k, r in
        [('soft', SOFT), ('oacal', CAL), ('C0', OTHERS[0]), ('CNT12', OTHERS[1])]}
Yd = runs['soft'][1].loc[ids]
for k, (_, r) in runs.items():
    a = r.loc[ids]
    assert ((a == Yd) | (a.isna() & Yd.isna())).all().all(), k
Y = Yd.to_numpy()
A = {k: s.loc[ids].to_numpy() for k, (s, _) in runs.items()}
A['ens soft'] = (A['soft'] + A['C0'] + A['CNT12']) / 3
A['ens oacal'] = (A['oacal'] + A['C0'] + A['CNT12']) / 3


def auc_vec(s, y):
    out = np.full(len(T), np.nan)
    for j in range(len(T)):
        m = np.isin(y[:, j], [0, 1])
        yy, x = y[m, j], s[m, j]
        npos, nneg = (yy == 1).sum(), (yy == 0).sum()
        if npos and nneg:
            out[j] = (rankdata(x)[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return out


sub = man.subset.loc[ids].to_numpy()
in_site = ids.isin(site)
SUB = {'full_545': np.arange(545), 'full_450': np.where(sub != 'new_95')[0],
       'original_208': np.where(sub == 'original_208')[0], 'new_242': np.where(sub == 'new_242')[0],
       'new_95': np.where(sub == 'new_95')[0], 'site group': np.where(in_site)[0], 'other sites': np.where(~in_site)[0]}
rng = np.random.default_rng(0)
print(f'oacal group: {CAL}\nbaseline  : {SOFT}')
for name, idx in SUB.items():
    y = Y[idx]
    print(f'\n== {name} (n={len(idx)}) ==')
    print(f"  {'':10s} {'macro':>7s} {'OA mean':>8s}  " + '  '.join(f'{t:>10s}' for t in OA))
    for k in ('soft', 'oacal', 'ens soft', 'ens oacal'):
        a = auc_vec(A[k][idx], y)
        print(f'  {k:10s} {np.nanmean(a):7.4f} {np.nanmean(a[OAI]):8.4f}  ' + '  '.join(f'{a[j]:10.4f}' for j in OAI))
    boots = [rng.choice(len(idx), len(idx)) for _ in range(1000)]
    for a_, b_ in (('oacal', 'soft'), ('ens oacal', 'ens soft')):
        dm, do = [], []
        for bt in boots:
            x, z = auc_vec(A[a_][idx][bt], y[bt]), auc_vec(A[b_][idx][bt], y[bt])
            dm.append(np.nanmean(x) - np.nanmean(z))
            do.append(np.nanmean(x[OAI]) - np.nanmean(z[OAI]))
        dm, do = np.array(dm), np.array(do)
        print(f'  {a_} - {b_}: macro {dm.mean():+.4f} [{np.percentile(dm, 2.5):+.4f}, {np.percentile(dm, 97.5):+.4f}] '
              f'| OA {do.mean():+.4f} [{np.percentile(do, 2.5):+.4f}, {np.percentile(do, 97.5):+.4f}] P>0 {(do > 0).mean():.3f}')

print('\n== per target, full_545: soft -> oacal (single V2S) ==')
a, b = auc_vec(A['soft'], Y), auc_vec(A['oacal'], Y)
for j, t in enumerate(T):
    print(f'  {t:17s} {a[j]:.4f} -> {b[j]:.4f}  ({b[j] - a[j]:+.4f})')
