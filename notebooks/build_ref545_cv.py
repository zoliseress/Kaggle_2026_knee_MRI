"""Build the 545-study reference 3-fold CV inputs from rsna_reference_545_3fold_soft.zip.

The 545 radiologist reference = the 450 (original 208 + 242 new, values unchanged) + 95 new
studies (8_RSNA_YC_six_series_99_batch2), split into 3 folds (182/182/181 validation studies).

Per fold N (read by run_cv545.sh):
  train_v12_ref545_{soft,binary}_foldN.csv
      = train_v11.csv without ALL 545 reference studies (3862 LLM soft-label rows, Report kept)
      + fold_N_train_radiologist_{soft,binary}.csv (363/364 radiologist rows, empty Report).
      The validation studies of fold N are not in it under any label source.
  ref545_foldN_val.csv = fold_N_val_radiologist_binary.csv (binary validation reference).
  Variant "soft2" (raw radiologist score 2 is not masked; reference_545_raw_scores.csv):
  train_v12_ref545_soft2_foldN.csv = the soft table with score-2 cells of the training
      radiologist rows = 0.5, and without the studies of notebooks/train_exclude_studies.csv
      (LLM and radiologist rows alike);
  ref545_foldN_val_two0.csv = the binary validation with score-2 cells = 0 (negative).
Repo files:
  notebooks/train_labeled_545_reference.csv  full binary reference (empty cell = masked)
  notebooks/reference_545_manifest.csv       StudyInstanceUID, subset (original_208 | new_242 |
                                             new_95), source_file, fold (validation fold)

Usage (from the repo root):
  python -I notebooks/build_ref545_cv.py [--zip <rsna_reference_545_3fold_soft.zip>] [--data-root <dir>]
"""
import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd

T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
U = 'StudyInstanceUID'
REPO = Path(__file__).resolve().parents[1]
N_FOLDS = 3

ap = argparse.ArgumentParser()
ap.add_argument('--data-root', default='/ai-storage/Temp/SZ/kaggle_knee/data')
ap.add_argument('--zip', default=None, help='default: <data-root>/rsna_reference_545_3fold_soft.zip')
ap.add_argument('--base-train', default=None, help='default: <data-root>/train_v11.csv')
args = ap.parse_args()
D = Path(args.data_root)
zpath = Path(args.zip) if args.zip else D / 'rsna_reference_545_3fold_soft.zip'
base_path = Path(args.base_train) if args.base_train else D / 'train_v11.csv'

zf = zipfile.ZipFile(zpath)
prefix = 'rsna_reference_545_3fold_soft/'


def read_zip(name):
    # Text only, so values are written back exactly as the package holds them.
    with zf.open(prefix + name) as fh:
        return pd.read_csv(io.TextIOWrapper(fh, encoding='utf-8-sig'), dtype=str, keep_default_na=False)


def to_num(df):
    return df[T].apply(lambda c: pd.to_numeric(c.replace('', None)))


folds = read_zip('fold_assignments.csv')
ref = read_zip('train_labeled_545_reference.csv')
soft = read_zip('train_labeled_545_reference_soft.csv')
assert list(ref.columns) == [U] + T and list(soft.columns) == [U] + T
assert ref[U].is_unique and len(ref) == 545 and set(ref[U]) == set(soft[U]) == set(folds[U])
assert folds[U].is_unique and folds['fold'].value_counts().sort_index().tolist() == [182, 182, 181]
s545 = set(ref[U])

# The 450 reference is kept cell for cell (incl. masked cells).
r450 = pd.read_csv(REPO / 'notebooks/train_labeled_450_reference.csv', dtype={U: str}, encoding='utf-8-sig')
assert set(r450[U]) <= s545
a = to_num(ref).set_index(ref[U]).loc[r450[U]].to_numpy()
b = r450[T].to_numpy(dtype=float)
assert ((a == b) | (pd.isna(a) & pd.isna(b))).all(), '545 reference changed a 450 reference cell'

# Soft labels: values {0, .15, .85, 1}, same mask as binary, rounding back to binary.
sn = to_num(soft).set_index(soft[U]).loc[ref[U]].to_numpy()
bn = to_num(ref).to_numpy()
assert (pd.isna(sn) == pd.isna(bn)).all()
assert set(pd.unique(sn[~pd.isna(sn)])) <= {0.0, 0.15, 0.85, 1.0}
assert (sn[~pd.isna(sn)].round() == bn[~pd.isna(bn)]).all()

base = pd.read_csv(base_path, dtype=str, keep_default_na=False)
assert base[U].is_unique and set(T) <= set(base.columns)
new95 = s545 - set(r450[U])
assert len(new95) == 95 and new95 <= set(base[U]), 'the 95 new studies should be LLM rows of the base table'
llm = base[~base[U].isin(s545)]
print(f'base {base_path.name}: {len(base)} rows, {len(base) - len(llm)} reference rows dropped -> {len(llm)}')

# Raw radiologist score 2 (masked in both the soft and the binary files). The raw file holds the
# Kaggle-binary studies on a 0/1 scale, so only the ordinal_0_4 rows are read.
raw = read_zip('reference_545_raw_scores.csv').set_index(U)[T]
prov = read_zip('soft_label_provenance.csv').set_index(U)
ordinal = prov.index[prov['raw_scale'] == 'ordinal_0_4']
assert len(ordinal) == 487
two = (raw.loc[ordinal] == '2').reindex(ref[U], fill_value=False)
assert int(two.to_numpy().sum()) == 100
assert to_num(ref).set_index(ref[U])[two].isna().to_numpy()[two.to_numpy()].all()
assert to_num(soft).set_index(soft[U])[two].isna().to_numpy()[two.to_numpy()].all()
exclude = set(pd.read_csv(REPO / 'notebooks/train_exclude_studies.csv', dtype=str)[U])
assert exclude <= set(base[U]) | s545


def set_cells(df, mask, value):
    """Copy of the text table `df` with the cells under `mask` (study x target) set to `value`."""
    out = df.copy()
    m = mask.reindex(out[U]).fillna(False).to_numpy(bool)
    for j, t in enumerate(T):
        out.loc[m[:, j], t] = value
    return out


for f in range(N_FOLDS):
    val = read_zip(f'fold_{f}_val_radiologist_binary.csv')
    val_ids = set(val[U])
    assert val_ids == set(folds.loc[folds['fold'] == str(f), U])
    val.to_csv(D / f'ref545_fold{f}_val.csv', index=False)
    for mode in ('soft', 'binary'):
        rad = read_zip(f'fold_{f}_train_radiologist_{mode}.csv')
        assert set(rad[U]) == s545 - val_ids
        rad = rad.reindex(columns=llm.columns, fill_value='')
        train = pd.concat([llm, rad], ignore_index=True)
        assert train[U].is_unique and not set(train[U]) & val_ids
        assert len(train) == len(llm) + 545 - len(val_ids)
        out = D / f'train_v12_ref545_{mode}_fold{f}.csv'
        train.to_csv(out, index=False)
        print(f'fold {f} {mode:6s}: {out.name} {len(train)} rows '
              f'({len(llm)} LLM + {len(rad)} radiologist), validation {len(val_ids)}')

    # soft2: score 2 -> 0.5 in training, -> 0 in validation; excluded studies leave training only.
    val2 = set_cells(val, two, '0')
    n_val2 = int(two.loc[sorted(val_ids)].to_numpy().sum())
    assert n_val2 == {0: 35, 1: 35, 2: 30}[f]
    val2.to_csv(D / f'ref545_fold{f}_val_two0.csv', index=False)
    rad = read_zip(f'fold_{f}_train_radiologist_soft.csv')
    rad2 = set_cells(rad, two, '0.5').reindex(columns=llm.columns, fill_value='')
    train2 = pd.concat([llm, rad2], ignore_index=True)
    train2 = train2[~train2[U].isin(exclude)]
    assert train2[U].is_unique and not set(train2[U]) & val_ids and not set(train2[U]) & exclude
    out = D / f'train_v12_ref545_soft2_fold{f}.csv'
    train2.to_csv(out, index=False)
    print(f'fold {f} soft2 : {out.name} {len(train2)} rows ({100 - n_val2} score-2 cells = 0.5); '
          f'ref545_fold{f}_val_two0.csv ({n_val2} score-2 cells = 0)')

# Same format as notebooks/train_labeled_450_reference.csv (floats, empty = masked).
ref_out = ref[[U]].join(to_num(ref).astype(float))
ref_out.to_csv(REPO / 'notebooks/train_labeled_545_reference.csv', index=False)
man = folds.copy()
man.loc[man[U].isin(new95), 'subset'] = 'new_95'
assert man['subset'].value_counts().to_dict() == {'new_242': 242, 'original_208': 208, 'new_95': 95}
man[[U, 'subset', 'source_file', 'fold']].to_csv(REPO / 'notebooks/reference_545_manifest.csv', index=False)
print('reference_545:', len(ref_out), 'masked cells:', int(ref_out[T].isna().sum().sum()))
