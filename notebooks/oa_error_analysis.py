"""Error analysis of the OA trio (Medial OA, Lateral OA, PF OA) on the CV545 OOF (cv545_v12 = V2S + C0 + CNT12).

Per study and target the misranking rate e = the share of opposite-class studies the model ranks
on the wrong side (a positive: negatives scored above it; a negative: positives scored below it).
Its mean over the positives equals 1 - AUC, so group means of e show where the AUC is lost.
Groups: subset, selected sagittal / coronal series (sequence class, slice spacing, missing slot),
field strength, laterality source (tag / geometry / unresolved, tag vs geometry disagreement), the
radiologist's raw grade (0-4), and the Kaggle train.csv / LLM train_v3 labels against the radiologist.
Medial / Lateral swap check: per study, is the medial-lateral OA ordering of the model the reverse of
the reference's? Worst cases (top FN / FP per target) go to work/analysis/oa_error_cases.csv for
reading, with the report text.

Usage:  python notebooks/oa_error_analysis.py
"""
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

R = Path('work/runs/train_v12')
D = Path('F:/Kaggle/data')
OUT = Path('work/analysis')
U = 'StudyInstanceUID'
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
OA = ['Medial OA', 'Lateral OA', 'PF OA']
CV = ['V2S_E3_img320_s123_cv545soft_20261020_010445', 'C0_E3_img320_s123_cv545soft_20261020_065853',
      'CNT12_E3_img320_s123_cv545soft_20261020_150528']
N_CASES = 15
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 30)


def pivot(path):
    p = pd.read_csv(path, dtype={U: str})
    p = p.assign(reference=p.reference.where(p.reference_valid.astype(bool)))
    return (p.pivot(index=U, columns='target', values='score')[T], p.pivot(index=U, columns='target', values='reference')[T])


# --- scores and reference ----------------------------------------------------------------------
man = pd.read_csv('notebooks/reference_545_manifest.csv', dtype={U: str}).set_index(U)
ids = pd.Index(sorted(man.index))
runs = [pivot(R / f'{r}_oof' / 'oof_predictions.csv') for r in CV]
S = pd.DataFrame(np.mean([s.loc[ids].to_numpy() for s, _ in runs], axis=0), index=ids, columns=T)
Y = runs[0][1].loc[ids]

# --- study features -----------------------------------------------------------------------------
sel = pd.read_csv('work/manifest/series_selection.csv', dtype=str)
sel = sel[sel.selected == 'True']
mf = pd.read_csv('work/manifest/manifest.csv', dtype=str,
                 usecols=[U, 'SeriesInstanceUID', 'volume_key', 'echo_time', 'repetition_time', 'scan_options',
                          'slice_spacing_mm', 'row_spacing_mm', 'field_strength', 'n_slices'])
sel = sel.merge(mf.drop(columns=['n_slices']), on=[U, 'SeriesInstanceUID', 'volume_key'], how='left')


def seq_class(r):
    """PD / T2 / T1 / STIR by TE, TR and the description; +FS for fat suppression."""
    d = str(r.series_description).lower()
    te, tr = pd.to_numeric(r.echo_time, errors='coerce'), pd.to_numeric(r.repetition_time, errors='coerce')
    fs = bool(re.search(r'fs|fat|spair|spir|sat|stir|tirm|dixon|_w\b', d)) or 'FS' in str(r.scan_options)
    if re.search(r'stir|tirm', d):
        return 'STIR'
    if (pd.notna(tr) and tr < 1000) or re.search(r't1', d):
        base = 'T1'
    elif pd.notna(te) and te >= 60 or re.search(r't2', d) and not re.search(r'pd|dp', d):
        base = 'T2'
    else:
        base = 'PD'
    return base + ('-FS' if fs else '')


sel['seq'] = sel.apply(seq_class, axis=1)
feat = pd.DataFrame(index=ids)
feat['subset'] = man.subset.loc[ids]
feat['fold'] = man.fold.loc[ids].astype(int)
for slot in ('sagittal', 'coronal', 'axial'):
    s = sel[sel.slot == slot].drop_duplicates(U).set_index(U)
    feat[f'{slot}_seq'] = s.seq.reindex(ids).fillna('MISSING')
    feat[f'{slot}_desc'] = s.series_description.reindex(ids)
    feat[f'{slot}_gap'] = pd.to_numeric(s.slice_spacing_mm.reindex(ids), errors='coerce')
fs_ = pd.to_numeric(sel.drop_duplicates(U).set_index(U).field_strength.reindex(ids), errors='coerce')
feat['field'] = np.where(fs_ >= 2.5, '3T', np.where(fs_ > 0, '1.5T', 'unknown'))
lat = pd.read_csv('work/manifest/laterality.csv', dtype=str).set_index(U).reindex(ids)
feat['lat_source'] = lat.side_source.fillna('absent')
feat['lat_conflict'] = lat.tag_side.notna() & lat.geometry_side.notna() & (lat.tag_side != lat.geometry_side)
feat['slice_gap_cor'] = pd.cut(feat.coronal_gap, [0, 3.25, 3.75, 99], labels=['<=3.25mm', '3.25-3.75mm', '>3.75mm'])

z = zipfile.ZipFile(D / 'rsna_reference_545_3fold_soft.zip')
read = lambda n: pd.read_csv(io.TextIOWrapper(z.open('rsna_reference_545_3fold_soft/' + n), encoding='utf-8-sig'),
                             dtype=str).set_index(U)
raw = read('reference_545_raw_scores.csv').reindex(ids)
scale = read('soft_label_provenance.csv').raw_scale.reindex(ids)
kag = pd.read_csv(D / 'train.csv', dtype={U: str}).set_index(U).reindex(ids)
llm = pd.read_csv(D / 'train_v3.csv', dtype={U: str}).set_index(U).reindex(ids)


# --- misranking rate ------------------------------------------------------------------------------
def misrank(score, y):
    """Per study: share of the opposite class on the wrong side (ties count half); NaN where y unknown."""
    e = np.full(len(y), np.nan)
    m = np.isin(y, [0, 1])
    s, yy = score[m], y[m]
    r = rankdata(s)
    npos, nneg = (yy == 1).sum(), (yy == 0).sum()
    rank_among_neg = r - rankdata(s, method='average') * 0  # placeholder for clarity
    out = np.empty(len(yy))
    neg_sorted, pos_sorted = np.sort(s[yy == 0]), np.sort(s[yy == 1])
    for i, (v, c) in enumerate(zip(s, yy)):
        if c == 1:   # negatives above + half the ties
            out[i] = ((neg_sorted > v).sum() + 0.5 * (neg_sorted == v).sum()) / nneg
        else:        # positives below + half the ties
            out[i] = ((pos_sorted < v).sum() + 0.5 * (pos_sorted == v).sum()) / npos
    e[m] = out
    return e


E = pd.DataFrame({t: misrank(S[t].to_numpy(), Y[t].to_numpy()) for t in T}, index=ids)


def auc(score, y):
    m = np.isin(y, [0, 1])
    s, yy = score[m], y[m]
    npos, nneg = (yy == 1).sum(), (yy == 0).sum()
    if not npos or not nneg:
        return np.nan
    return (rankdata(s)[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


print('== OA AUC (CV3 OOF) per subset ==')
for name, m in [('full_545', np.ones(545, bool))] + [(s, (feat.subset == s).to_numpy()) for s in
                                                     ('original_208', 'new_242', 'new_95')]:
    print(f'  {name:13s} n={m.sum():3d}  ' + '  '.join(
        f'{t} {auc(S[t].to_numpy()[m], Y[t].to_numpy()[m]):.4f} (+{int((Y[t].to_numpy()[m] == 1).sum())})' for t in OA))


def group_table(col, min_n=8):
    """Per group: n, mean misranking of positives / negatives per OA target, and the group's AUC."""
    rows = []
    for g, idx in feat.groupby(col, observed=True).groups.items():
        if len(idx) < min_n:
            continue
        r = {col: g, 'n': len(idx)}
        for t in OA:
            y, e = Y.loc[idx, t], E.loc[idx, t]
            r[f'{t} +'] = int((y == 1).sum())
            r[f'{t} e+'] = e[y == 1].mean()
            r[f'{t} e-'] = e[y == 0].mean()
            r[f'{t} AUC'] = auc(S.loc[idx, t].to_numpy(), y.to_numpy())
        rows.append(r)
    return pd.DataFrame(rows).set_index(col).round(3)


for col in ('subset', 'sagittal_seq', 'coronal_seq', 'axial_seq', 'slice_gap_cor', 'field', 'lat_source', 'lat_conflict'):
    print(f'\n== by {col}  (e+ = mean misranking of positives, e- = of negatives; overall e+ = 1 - AUC) ==')
    print(group_table(col).to_string())

print('\n== by the radiologist raw grade (ordinal_0_4 studies), mean misranking ==')
for t in OA:
    g = raw.loc[scale == 'ordinal_0_4', t]
    e = E.loc[g.index, t]
    print(f'  {t:10s} ' + '  '.join(f'grade {k}: n={int((g == k).sum()):3d} e={e[g == k].mean():.3f}' for k in '01234'))

print('\n== label sources against the radiologist (hard cells), and the model error where they disagree ==')
for t in OA:
    y = Y[t]
    known = y.isin([0, 1])
    k = (pd.to_numeric(kag[t], errors='coerce') >= 0.5).astype(float).where(kag[t].notna())
    l = (pd.to_numeric(llm[t], errors='coerce') >= 0.5).astype(float).where(llm[t].notna())
    for name, lab in (('Kaggle train.csv', k), ('LLM train_v3', l)):
        agree = known & (lab == y)
        dis = known & lab.notna() & (lab != y)
        print(f'  {t:10s} {name:16s} agree {agree.sum():3d}  disagree {dis.sum():3d} '
              f'(FN-type {int((dis & (y == 1)).sum())}, FP-type {int((dis & (y == 0)).sum())})  '
              f'model e: agree {E.loc[agree, t].mean():.3f}  disagree {E.loc[dis, t].mean():.3f}')

print('\n== Medial / Lateral swap check ==')
both = Y['Medial OA'].isin([0, 1]) & Y['Lateral OA'].isin([0, 1]) & (Y['Medial OA'] != Y['Lateral OA'])
d_ref = (Y['Medial OA'] - Y['Lateral OA'])[both]                     # +1 medial only, -1 lateral only
z_m = pd.Series(rankdata(S['Medial OA']) / 545, index=ids)
z_l = pd.Series(rankdata(S['Lateral OA']) / 545, index=ids)
d_mod = (z_m - z_l)[both]
swapped = np.sign(d_mod) != np.sign(d_ref)
strong = swapped & (d_mod.abs() > 0.3)
print(f'  studies with exactly one of medial / lateral OA: {both.sum()}; model ordering reversed: {swapped.sum()} '
      f'({swapped.mean():.1%}); strongly reversed (|rank gap| > 0.3): {strong.sum()}')
print(feat.loc[d_ref.index].assign(swapped=swapped, strong=strong).groupby('lat_source')[['swapped', 'strong']]
      .agg(['sum', 'mean', 'count']).round(3).to_string())
print('  laterality conflict (tag != geometry): ' +
      str(feat.loc[d_ref.index].assign(swapped=swapped).groupby('lat_conflict').swapped.agg(['sum', 'count']).to_dict()))

# --- worst cases for reading ----------------------------------------------------------------------
OUT.mkdir(parents=True, exist_ok=True)
rows = []
for t in OA:
    y = Y[t]
    for kind, cls, asc in (('FN', 1, False), ('FP', 0, False)):
        idx = E.loc[y == cls, t].sort_values(ascending=asc).index[:N_CASES]
        for i in idx:
            rows.append({'target': t, 'type': kind, U: i, 'misrank': round(E.at[i, t], 3), 'score': round(S.at[i, t], 3),
                         'score_rank_pct': round(rankdata(S[t])[ids.get_loc(i)] / 545, 3), 'ref': y[i],
                         'raw_grade': raw.at[i, t], 'kaggle': kag.at[i, t], 'llm_v3': llm.at[i, t],
                         **{c: feat.at[i, c] for c in ('subset', 'fold', 'sagittal_seq', 'sagittal_desc', 'coronal_seq',
                                                       'coronal_desc', 'axial_desc', 'field', 'lat_source', 'lat_conflict')},
                         'other_OA_ref': '; '.join(f'{o}={Y.at[i, o]}' for o in OA if o != t),
                         'report': str(kag.at[i, 'Report'])[:1500]})
cases = pd.DataFrame(rows)
cases.to_csv(OUT / 'oa_error_cases.csv', index=False, encoding='utf-8-sig')
print(f'\nworst cases: {OUT / "oa_error_cases.csv"} ({len(cases)} rows)')
print(cases.groupby(['target', 'type'])[['subset', 'coronal_seq', 'lat_source']].agg(
    lambda s: ', '.join(f'{k}:{v}' for k, v in s.value_counts().items())).to_string())
