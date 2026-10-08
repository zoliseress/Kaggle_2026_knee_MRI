"""Build train_v8: train_v7 with the "not mentioned" cells set to 0.07 instead of 0.25.

train_v6/v7 = llm_labels_v4_blend = 0.5 * llm_labels_v2 (stevenleehans) + 0.5 * labels_llm_gpt56sol (lixin73).
For a finding the report does not mention, v2 gives 0.5 and gpt56sol gives 0, so the blend gives 0.25.
On the 208 radiologist reference only 7.1% of these cells are positive, so they get one global value of 0.07
(a single scalar, not per-target values, to avoid tuning on the evaluation set).
The cells are found from the two sources (v2 == 0.5 & gpt56sol == 0), not from == 0.25, so the few cells
that are 0.25 by coincidence (e.g. 0.05 / 0.45) stay unchanged.
Synovitis is left out: it is the one column where llm_labels_v2 differs from llm_labels_full. Reports rarely
mention synovitis (full: 0.5 in 3689 rows), and v2 re-estimates those cells (~0.27 / ~0.65). Its remaining 0.5
means "undecided", not "not mentioned" (ref208 positive rate at blend 0.25: 19%, between the 0.13 and 0.33 groups).
"""
import numpy as np, pandas as pd

D = 'F:/Kaggle/data/'
L = 'F:/Kaggle/llm_labeling/output/'
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
NOT_MENTIONED = '0.07'
KEEP = ['Synovitis']

# values kept as text so the untouched cells are not reformatted
v7 = pd.read_csv(D + 'train_v7.csv', dtype=str, keep_default_na=False)
uid = v7.StudyInstanceUID
v2 = pd.read_csv(L + 'labels_lixin_gpt56sol_V2/llm_labels_v2.csv').set_index('StudyInstanceUID').loc[uid, T]
gpt = pd.read_csv(L + 'labels_lixin_gpt56sol/labels_llm_gpt56sol.csv').set_index('StudyInstanceUID').loc[uid, T]
assert not v2.isna().any().any() and not gpt.isna().any().any()

x7 = v7[T].astype(float).to_numpy()
assert np.abs(0.5 * v2.to_numpy() + 0.5 * gpt.to_numpy() - x7).max() < 1e-9, 'train_v7 is not the v2/gpt56sol blend'

mask = np.isclose(v2.to_numpy(), 0.5) & (gpt.to_numpy() == 0)
mask[:, [T.index(t) for t in KEEP]] = False
assert np.isclose(x7[mask], 0.25).all()

v8 = v7.copy()
vals = v8[T].to_numpy(dtype=object)
vals[mask] = NOT_MENTIONED
v8[T] = vals

r208 = pd.read_csv(D + 'train_labeled_208_reference.csv', encoding='utf-8-sig')
assert len(v8) == len(v7) == 4199 and list(v8.columns) == list(v7.columns)
assert not set(v8.StudyInstanceUID) & set(r208.StudyInstanceUID)
assert ((v8[T] != v7[T]).to_numpy() == mask).all()
v8.to_csv(D + 'train_v8.csv', index=False)

print('train_v8:', len(v8), 'studies, cells changed 0.25 ->', NOT_MENTIONED, ':', int(mask.sum()),
      f'of {mask.size} ({mask.mean():.1%})')
print(pd.Series(mask.sum(axis=0), index=T).to_string())
print('0.25 cells left (Synovitis + coincidental):',int(np.isclose(v8[T].astype(float), 0.25).sum().sum()))
