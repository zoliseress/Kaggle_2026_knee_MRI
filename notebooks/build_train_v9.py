"""Build train_v9: train_v8 labels mixed with teacher out-of-fold predictions (self-distillation).

label_v9 = ALPHA * label_v8 + (1 - ALPHA) * teacher_oof
Teacher: V2S E3 img320, 3-fold CV on train_v8 (work/splits/cv3_train_v8, the 208 radiologist studies
never train). Every train_v8 study gets the prediction of the one fold model that did not train on it
(best.pt, epoch chosen on the fold's train_v8 labels). ALPHA is fixed a priori (0.7, >= 0.5 keeps the
LLM label dominant), not tuned on the 208 reference.
Label AUC on the 208 (teacher = mean of the 3 fold models): v8 0.8933, ALPHA 0.7 0.9358, teacher alone 0.9304.
"""
import numpy as np, pandas as pd

D = 'F:/Kaggle/data/'
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
ALPHA = 0.7
OOF = 'work/runs/train_v8/V2S_E3_img320_teacher_cv3_20261007_232654_oof/oof_predictions.csv'

v8 = pd.read_csv(D + 'train_v8.csv', dtype={'StudyInstanceUID': str}, keep_default_na=False)
oof = pd.read_csv(OOF, dtype={'StudyInstanceUID': str})
assert oof.groupby(['StudyInstanceUID', 'target']).size().max() == 1, 'duplicate OOF rows'
teacher = oof.pivot(index='StudyInstanceUID', columns='target', values='score')[T]
assert set(teacher.index) == set(v8.StudyInstanceUID), 'OOF must cover exactly the train_v8 studies'
assert teacher.notna().all().all() and ((teacher >= 0) & (teacher <= 1)).all().all()

r208 = pd.read_csv(D + 'train_labeled_208_reference.csv', encoding='utf-8-sig')
assert not set(v8.StudyInstanceUID) & set(r208.StudyInstanceUID)

lab = v8[T].astype(float).to_numpy()
mix = ALPHA * lab + (1 - ALPHA) * teacher.loc[v8.StudyInstanceUID, T].to_numpy()
assert np.isfinite(mix).all() and mix.min() >= 0 and mix.max() <= 1

v9 = v8.copy()
v9[T] = np.round(mix, 6)
assert len(v9) == len(v8) == 4199 and list(v9.columns) == list(v8.columns)
v9.to_csv(D + 'train_v9.csv', index=False)

print(f'train_v9: {len(v9)} studies, ALPHA={ALPHA}')
print('mean label v8 -> v9 per target:')
print(pd.DataFrame({'v8': lab.mean(axis=0), 'teacher': teacher.loc[v8.StudyInstanceUID, T].mean().to_numpy(),
                    'v9': mix.mean(axis=0)}, index=T).round(4).to_string())
