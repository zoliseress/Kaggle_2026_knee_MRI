"""Build the 208-study radiologist reference and train_v7.

train_labeled_208_reference.csv = train_labeled_158_reference.csv + RSNA_Andrew 50.xlsx
  (0-4 scores binarized like the new-100: 0-1 -> 0, 3-4 -> 1, 2 / X -> empty = masked).
train_v7.csv = train_v6.csv without the 50 new reference studies (4199 studies).
"""
import numpy as np, pandas as pd

D = 'F:/Kaggle/data/'
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

r158 = pd.read_csv(D + 'train_labeled_158_reference.csv', encoding='utf-8-sig')
a50 = pd.read_excel(D + 'RSNA_Andrew 50.xlsx', sheet_name='Reading')
assert len(a50) == 50 and (a50['Readiness'] == 'Ready').all()
s = a50[T].apply(pd.to_numeric, errors='coerce')  # X -> NaN
assert s.stack().isin([0, 1, 2, 3, 4]).all()
b50 = (s >= 3).astype(float).where(s.notna() & (s != 2))
b50.insert(0, 'StudyInstanceUID', a50['StudyInstanceUID'].astype(str))
assert not set(b50.StudyInstanceUID) & set(r158.StudyInstanceUID)

r208 = pd.concat([r158, b50], ignore_index=True)
r208[T] = r208[T].astype('Int64')
assert r208.StudyInstanceUID.is_unique and len(r208) == 208
for p in (D + 'train_labeled_208_reference.csv', 'notebooks/train_labeled_208_reference.csv'):
    r208.to_csv(p, index=False, encoding='utf-8-sig')
print('ref208:', len(r208), 'masked cells:', int(r208[T].isna().sum().sum()))

# train_v7: row filter of train_v6, values kept as text so nothing is reformatted
v6 = pd.read_csv(D + 'train_v6.csv', dtype=str, keep_default_na=False)
v7 = v6[~v6.StudyInstanceUID.isin(set(b50.StudyInstanceUID))]
assert len(v6) - len(v7) == 50 and not set(v7.StudyInstanceUID) & set(r208.StudyInstanceUID)
v7.to_csv(D + 'train_v7.csv', index=False)
print('train_v7:', len(v7), 'of train_v6', len(v6))
