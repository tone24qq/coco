
"""train_meta.py
使用 priors 特徵 + 已標 answer 樣本訓練 LogisticRegression
"""
import pathlib, json, numpy as np, joblib, warnings, sys
from sklearn.linear_model import LogisticRegression

BASE = pathlib.Path(__file__).resolve().parent.parent
SAMPLES = BASE/'data'/'samples'
PRIORS_PATH = BASE/'data'/'priors'/'priors.npz'
OUT_PATH = BASE/'data'/'meta_model.pkl'

if not PRIORS_PATH.exists():
    print('❗ priors.npz 不存在，請先執行 build_priors.py')
    sys.exit(1)
priors = np.load(PRIORS_PATH)

X=[]; y=[]
for fp in SAMPLES.glob('*.json'):
    s=json.load(open(fp,encoding='utf-8'))
    grid=np.array(s['grid'])
    R,C=grid.shape
    size=f"{R}x{C}"
    r,c=s['answer']
    r-=1; c-=1
    if r<0:
        continue   # 未標答案
    pos=priors[f"{size}_pos"]
    num_prior = priors.get(f"{size}_num{s['target']}", pos)
    center = 1 - (np.abs(np.arange(R)-(R+1)/2)[:,None] + np.abs(np.arange(C)-(C+1)/2)[None,:])/(R+C)
    feats=np.stack([pos,num_prior,center],-1).reshape(-1,3)
    X.append(feats)
    label=np.zeros(R*C); label[r*C+c]=1
    y.append(label)
if not y:
    raise RuntimeError('尚無任何已標答案樣本（answer），請先在 JSON 樣本中填 row/col（1-base）。')
X=np.vstack(X); y=np.hstack(y)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    model=LogisticRegression(class_weight='balanced',max_iter=500).fit(X,y)
joblib.dump(model, OUT_PATH)
print(f'✓ meta_model.pkl 生成於 {OUT_PATH}')
