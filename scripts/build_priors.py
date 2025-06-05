
"""build_priors.py
從 data/samples/*.json 產生位置/號碼熱力圖 priors.npz
"""
import pathlib, json, collections, numpy as np, sys

BASE = pathlib.Path(__file__).resolve().parent.parent
SAMPLES = BASE / 'data' / 'samples'
DST = BASE / 'data' / 'priors'
DST.mkdir(parents=True, exist_ok=True)

files = list(SAMPLES.glob('*.json'))
if not files:
    print(f'❗ {SAMPLES} 沒有樣本，請先執行 convert_excel.py')
    sys.exit(1)

pos_hits = collections.defaultdict(lambda: None)
num_hits = collections.defaultdict(lambda: None)

for fp in files:
    s = json.load(open(fp,encoding='utf-8'))
    R, C = len(s['grid']), len(s['grid'][0])
    size = f"{R}x{C}"
    if pos_hits[size] is None:
        pos_hits[size] = np.zeros((R,C))
    r,c = s['answer']
    r-=1; c-=1
    if r>=0:
        pos_hits[size][r,c]+=1
        key=(size,str(s['target']))
        if num_hits.get(key) is None:
            num_hits[key]=np.zeros((R,C))
        num_hits[key][r,c]+=1

priors={}
for size,mat in pos_hits.items():
    R,C=mat.shape
    priors[f"{size}_pos"]=(mat+1)/(mat.sum()+R*C)
for (size,num),mat in num_hits.items():
    R,C=mat.shape
    priors[f"{size}_num{num}"]=(mat+1)/(mat.sum()+R*C)

np.savez(DST/'priors.npz', **priors)
print(f'✓ priors.npz 生成於 {DST}')
