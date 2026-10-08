"""Pick 30 trauma + 20 degenerative train studies (outside the 158 ref) for radiologist labelling.
Pattern quotas on train_v6 LLM labels (pos >= 0.9, neg < 0.5) so the focus targets are not all co-positive."""
import pandas as pd, numpy as np
D = 'F:/Kaggle/data/'
v6 = pd.read_csv(D + 'train_v6.csv', encoding='utf-8-sig')
tr = pd.read_csv(D + 'train.csv', encoding='utf-8-sig')
ref = set(pd.read_csv('notebooks/train_labeled_158_reference.csv', encoding='utf-8-sig').StudyInstanceUID)
cols = [c for c in v6.columns if c not in ('StudyInstanceUID', 'Report')]
v6 = v6[~v6.StudyInstanceUID.isin(ref)].reset_index(drop=True)
POS, NEG = 0.9, 0.5
p = lambda c: v6[c] >= POS
n = lambda c: v6[c] < NEG
trauma_any = ~(n('MCL') & n('ACL') & n('Contusion') & n('Fracture'))  # any trauma label >= 0.5
patterns = [
    # (package, pattern id, description, mask, count)
    ('trauma', 'A', 'MCL+ ACL-',                   p('MCL') & n('ACL'), 10),
    ('trauma', 'B', 'MCL+ ACL+',                   p('MCL') & p('ACL'), 8),
    ('trauma', 'C', 'ACL+ MCL- (MCL hard neg)',    p('ACL') & n('MCL'), 2),
    ('trauma', 'D', 'Fracture+ MCL-',              p('Fracture') & n('MCL'), 5),
    ('trauma', 'E', 'Contusion+ Fracture- MCL-',   p('Contusion') & n('Fracture') & n('MCL'), 5),
    ('degenerative', 'F', 'Lateral OA+ Medial OA-', p('Lateral OA') & n('Medial OA') & ~trauma_any, 6),
    ('degenerative', 'G', 'Medial OA+ Lateral OA-', p('Medial OA') & n('Lateral OA') & ~trauma_any, 6),
    ('degenerative', 'H', 'Lateral OA+ Medial OA+', p('Lateral OA') & p('Medial OA') & ~trauma_any, 5),
    ('degenerative', 'I', 'Med. meniscus+ no OA (OA hard neg)',
     p('Medial Meniscus') & n('Medial OA') & n('Lateral OA') & ~trauma_any, 3),
]
rng = np.random.default_rng(2026)
taken, parts = set(), []
for pkg, pid, desc, mask, k in patterns:
    cand = v6[mask & ~v6.StudyInstanceUID.isin(taken)]
    print(f'{pkg:12s} {pid} {desc:36s} available {len(cand):4d} -> {k}')
    pick = cand.iloc[rng.choice(len(cand), size=k, replace=False)].assign(Package=pkg, Pattern=pid, Pattern_desc=desc)
    taken |= set(pick.StudyInstanceUID); parts.append(pick)
out = pd.concat(parts).reset_index(drop=True)
idx_map = dict(zip(tr.StudyInstanceUID, tr.index))
out['train_index_0based'] = out.StudyInstanceUID.map(idx_map)
focus = {'trauma': ['MCL', 'Contusion', 'Fracture', 'ACL'], 'degenerative': ['Lateral OA', 'Medial OA']}
out['Focus_LLM_pos'] = out.apply(lambda r: ', '.join(c for c in focus[r.Package] if r[c] >= NEG), axis=1)
out['Other_LLM_pos'] = out.apply(lambda r: ', '.join(c for c in cols if r[c] >= NEG and c not in focus[r.Package]), axis=1)
out['ID'] = [('T' if pk == 'trauma' else 'D') + f'{i:02d}' for pk, i in zip(out.Package, out.groupby('Package').cumcount() + 1)]
out[cols] = out[cols].round(2)
out = out[['ID', 'Package', 'Pattern', 'Pattern_desc', 'StudyInstanceUID', 'train_index_0based',
           'Focus_LLM_pos', 'Other_LLM_pos'] + cols + ['Report']]
out.to_csv('notebooks/ref_expansion_candidates_llm.csv', index=False, encoding='utf-8-sig')
# blinded form for the radiologist: no LLM labels, packages shuffled together
blind = out[['ID', 'StudyInstanceUID', 'train_index_0based']].sample(frac=1, random_state=7).reset_index(drop=True)
blind.insert(0, 'Order', range(1, len(blind) + 1))
blind['Open MRI'] = ('E:/rsna kaggle/train_series/' + blind.StudyInstanceUID).str.replace('/', chr(92))
for c in cols:
    blind[c] = ''
blind['Comments'] = ''
blind.to_csv('notebooks/ref_expansion_50_blinded.csv', index=False, encoding='utf-8-sig')
prec = {'ACL': .81, 'MCL': .46, 'Contusion': .60, 'Fracture': .77, 'Lateral OA': .54, 'Medial OA': .64}
for pkg in ['trauma', 'degenerative']:
    s = out[out.Package == pkg]
    cnt = (s[cols] >= POS).sum()
    print(pkg, {c: (int(cnt[c]), round(cnt[c] * prec[c], 1)) for c in prec})
