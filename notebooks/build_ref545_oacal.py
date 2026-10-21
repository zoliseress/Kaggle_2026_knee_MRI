"""CV545 "oacal" label variant: the OA labels of one site group's LLM rows calibrated to the radiologist.

Background (doc/problemas_esetek.md B9): for the site group whose selected sagittal series is
`Sag_PDW_TSE_SPAIR` / `SAG PDW_TSE_SPAIR` / `pd_tse_fs_sag` (1288 of 4407 studies, all of new_95), the
report-based LLM OA labels agree poorly with the radiologist (LLM-vs-radiologist AUC 0.72-0.78, other
sites 0.86-0.88), and the model's OA is weakest there.

Per fold N and OA target (Medial OA, Lateral OA, PF OA): a Platt map p -> sigmoid(a * logit(p) + b),
fitted on that site group's reference studies in fold N's TRAINING part only (their LLM value from
train_v8.csv against the binary radiologist label; fold N's validation studies never enter the fit),
replaces the OA values of the site group's LLM rows. Everything else of train_v12_ref545_soft_foldN.csv is
kept cell for cell (other targets, other sites, all radiologist rows, empty cells stay empty).

Writes <data-root>/train_v12_ref545_oacal_foldN.csv (read by LABEL_MODE=oacal ./run_cv545.sh).
The site list is notebooks/oa_site_group_studies.csv (written from work/manifest/series_selection.csv with
--write-site-list; committed so that every machine uses the same list).

Usage (from the repo root, after build_ref545_cv.py):
  python -I notebooks/build_ref545_oacal.py --data-root F:/Kaggle/data [--write-site-list]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

U = 'StudyInstanceUID'
OA = ['Medial OA', 'Lateral OA', 'PF OA']
REPO = Path(__file__).resolve().parents[1]
SITE_CSV = REPO / 'notebooks/oa_site_group_studies.csv'
SITE_PATTERN = r'Sag_PDW_TSE_SPAIR|SAG PDW_TSE_SPAIR|pd_tse_fs_sag'
EPS = 1e-3

ap = argparse.ArgumentParser()
ap.add_argument('--data-root', default='/ai-storage/Temp/SZ/kaggle_knee/data')
ap.add_argument('--write-site-list', action='store_true',
                help='rebuild notebooks/oa_site_group_studies.csv from work/manifest/series_selection.csv')
args = ap.parse_args()
D = Path(args.data_root)

if args.write_site_list:
    sel = pd.read_csv(REPO / 'work/manifest/series_selection.csv', dtype=str)
    sel = sel[(sel.selected == 'True') & (sel.slot == 'sagittal')]
    hit = sel[sel.series_description.fillna('').str.contains(SITE_PATTERN, regex=True)]
    hit[[U, 'series_description']].drop_duplicates(U).sort_values(U).to_csv(SITE_CSV, index=False)
    print(f'site list: {SITE_CSV.name} {hit[U].nunique()} studies')
site = set(pd.read_csv(SITE_CSV, dtype=str)[U])
assert len(site) == 1288, len(site)

ref = pd.read_csv(REPO / 'notebooks/train_labeled_545_reference.csv', dtype={U: str}).set_index(U)
man = pd.read_csv(REPO / 'notebooks/reference_545_manifest.csv', dtype={U: str}).set_index(U)
llm_ref = pd.read_csv(D / 'train_v8.csv', dtype={U: str}).set_index(U)    # LLM values of the 242 + 95


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


for f in range(3):
    src = D / f'train_v12_ref545_soft_fold{f}.csv'
    tr = pd.read_csv(src, dtype=str, keep_default_na=False)
    val = set(man.index[man.fold.astype(int) == f])
    assert not set(tr[U]) & val, f'fold {f}: validation studies in the training table'
    is_ref = tr[U].isin(ref.index)
    rows = (~is_ref) & tr[U].isin(site)                                     # the site group's LLM rows
    fit_ids = [i for i in tr.loc[is_ref, U] if i in site and i in llm_ref.index]   # fold-f training refs only
    assert not set(fit_ids) & val
    out = tr.copy()
    for t in OA:
        x = pd.to_numeric(llm_ref.loc[fit_ids, t], errors='coerce')
        y = ref.loc[fit_ids, t]
        m = x.notna() & y.isin([0, 1])
        lr = LogisticRegression(C=100.0).fit(logit(x[m])[:, None], y[m].to_numpy().astype(int))
        a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        cell = out.loc[rows, t]
        filled = cell != ''
        v = pd.to_numeric(cell[filled])
        new = 1 / (1 + np.exp(-(a * logit(v) + b)))
        out.loc[cell[filled].index, t] = [f'{p:.4f}' for p in new]
        print(f'fold {f} {t:10s} fit n={int(m.sum()):3d} (pos {int(y[m].sum()):2d})  a={a:.3f} b={b:+.3f}  '
              f'map 0.02->{1 / (1 + np.exp(-(a * logit(0.02) + b))):.2f} 0.5->{1 / (1 + np.exp(-b)):.2f} '
              f'0.92->{1 / (1 + np.exp(-(a * logit(0.92) + b))):.2f}  rows {int(filled.sum())}, '
              f'label mass {v.sum():.0f} -> {new.sum():.0f}')
    # Nothing else changed
    other = [c for c in tr.columns if c not in OA]
    assert out[other].equals(tr[other])
    assert out.loc[~rows, OA].equals(tr.loc[~rows, OA])
    dst = D / f'train_v12_ref545_oacal_fold{f}.csv'
    out.to_csv(dst, index=False)
    print(f'fold {f}: {dst.name} {len(out)} rows, {int(rows.sum())} site LLM rows recalibrated '
          f'(fit on {len(fit_ids)} site refs of the fold-{f} training part)')
