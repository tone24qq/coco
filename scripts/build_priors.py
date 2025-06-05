import json, numpy as np, pathlib, collections, sys

BASE = pathlib.Path(__file__).resolve().parent.parent
SAMPLES = BASE / "data" / "samples"
DST     = BASE / "data" / "priors"
DST.mkdir(parents=True, exist_ok=True)

sample_files = list(SAMPLES.glob("*.json"))
if not sample_files:
    print(f"❗ 樣本資料夾 {SAMPLES} 為空，請先執行 convert_excel.py")
    sys.exit(1)

pos_hits = collections.defaultdict(lambda: None)  # size -> matrix
num_hits = collections.defaultdict(lambda: None)  # (size,target) -> matrix

for file in sample_files:
    s = json.load(open(file, encoding="utf-8"))
    R, C = len(s["grid"]), len(s["grid"][0])
    size = f"{R}x{C}"
    if pos_hits[size] is None:
        pos_hits[size] = np.zeros((R, C))
    r, c = s["answer"]
    # 1‑based to 0‑based
    r -= 1
    c -= 1
    if r >= 0:
        pos_hits[size][r, c] += 1
        key = (size, str(s["target"]))
        if num_hits.get(key) is None:
            num_hits[key] = np.zeros((R, C))
        num_hits[key][r, c] += 1

priors = {}
for size, mat in pos_hits.items():
    R, C = mat.shape
    priors[f"{size}_pos"] = (mat + 1) / (mat.sum() + R * C)

for (size, num), mat in num_hits.items():
    R, C = mat.shape
    priors[f"{size}_num{num}"] = (mat + 1) / (mat.sum() + R * C)

np.savez(DST / "priors.npz", **priors)
print("✓ priors.npz 生成於", DST)
