import json, numpy as np, pathlib, joblib, warnings, sys
from sklearn.linear_model import LogisticRegression

BASE = pathlib.Path(__file__).resolve().parent.parent
SAMPLES = BASE / "data" / "samples"
PRIORS_PATH = BASE / "data" / "priors" / "priors.npz"

if not PRIORS_PATH.exists():
    print(f"❗ 先驗檔 {PRIORS_PATH} 不存在，請先執行 build_priors.py")
    sys.exit(1)

priors = np.load(PRIORS_PATH)

X, y = [], []
for file in SAMPLES.glob("*.json"):
    s = json.load(open(file, encoding="utf-8"))
    grid = np.array(s["grid"])
    R, C = grid.shape
    size = f"{R}x{C}"

    # 1‑based to 0‑based
    r, c = s["answer"]
    r -= 1
    c -= 1
    if r < 0:
        continue  # 未標答案

    pos = priors[f"{size}_pos"]
    num_prior = priors.get(f"{size}_num{s['target']}", pos)
    center = 1 - (np.abs(np.arange(R) - (R + 1) / 2)[:, None] + np.abs(np.arange(C) - (C + 1) / 2)[None, :]) / (R + C)

    feats = np.stack([pos, num_prior, center], -1)
    X.append(feats.reshape(-1, 3))

    label = np.zeros(R * C)
    label[r * C + c] = 1
    y.append(label)

if not y:
    raise RuntimeError("尚無任何已標答案樣本（answer），請先在 JSON 樣本中填 row/col（1-base）。")

X = np.vstack(X)
y = np.hstack(y)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = LogisticRegression(class_weight="balanced", max_iter=500).fit(X, y)

OUT = BASE / "data" / "meta_model.pkl"
joblib.dump(model, OUT)
print("✓ meta_model.pkl 生成於", OUT)
