# convert_json_samples.py
import os, json, time, re
import numpy as np
from analyzer11 import collect_all_scores, normalize_tensor, fuse_scores

JSON_FOLDER = "json_samples"            # 放 sample1_8x10.json、sample2_8x10.json
OUTPUT_PATH = "memory_data/all.json"    # 转好的所有样本将写到这里

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
all_samples = []

def clean_cell(x):
    # 非数字一律当 -1；字串里抓第一个数字序列
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        m = re.search(r"\d+", x)
        return int(m.group(0)) if m else -1
    return -1

def load_and_clean(json_path):
    data = json.load(open(json_path, "r", encoding="utf-8"))
    raw = data.get("grid", [])
    rows = data.get("rows", len(raw))
    cols = data.get("cols", len(raw[0]) if raw else 0)
    cleaned = []
    for row in raw:
        cleaned_row = [clean_cell(cell) for cell in row]
        # 丢掉整行都是 -1 或长度不对的 Footer
        if len(cleaned_row) != cols: 
            continue
        if all(val == -1 for val in cleaned_row):
            continue
        cleaned.append(cleaned_row)
    return np.array(cleaned, dtype=int)

def process_card(arr: np.ndarray):
    samples = []
    rows, cols = arr.shape
    values = sorted(set(v for v in arr.flatten() if v > 0))
    for target in values:
        pos = np.argwhere(arr == target)
        if pos.shape[0] == 0:
            continue
        r, c = int(pos[0][0]), int(pos[0][1])
        grid_masked = arr.copy()
        grid_masked[r, c] = -1
        tensor = collect_all_scores(grid_masked, request_id="convert")
        if tensor.size == 0:
            continue
        raw_scores = {}
        names = list(collect_all_scores.__globals__['REGISTERED_MODULES_BRAIN'].keys())
        for i, name in enumerate(names):
            raw_scores[name] = float(tensor[i, r, c])
        normed = normalize_tensor(tensor, method="minmax")
        fused = fuse_scores(normed, weights=None)
        fused_score = float(fused[r, c])
        samples.append({
            "grid": grid_masked.tolist(),
            "target": target,
            "true_pos": [r, c],
            "scores": raw_scores,
            "fused_score": fused_score,
            "timestamp": int(time.time())
        })
    return samples

def main():
    all_samples.clear()
    for fn in os.listdir(JSON_FOLDER):
        if not fn.lower().endswith(".json"):
            continue
        arr = load_and_clean(os.path.join(JSON_FOLDER, fn))
        if arr.size == 0:
            print(f"跳过 {fn}：清洗后为空。")
            continue
        card_samples = process_card(arr)
        all_samples.extend(card_samples)
        print(f"{fn} → {len(card_samples)} 个样本")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"共输出 {len(all_samples)} 个样本到 {OUTPUT_PATH}")

if __name__ == "__main__":
    main()