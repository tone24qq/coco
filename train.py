import os
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib
import itertools

for d in ['data', 'models', 'results', 'fake']:
    os.makedirs(d, exist_ok=True)

def scan_all_numeric_blocks(filepath, min_row=2, min_col=3):
    df = pd.read_excel(filepath, header=None, dtype=str)
    blocks = []
    block = []
    for row in df.itertuples(index=False):
        cleaned_row = []
        for val in row:
            sval = str(val).strip().upper().replace('Ｏ', 'O').replace('０', '0')
            sval = sval.replace("O", "0").replace("I", "1")
            if sval.isdigit():
                cleaned_row.append(int(sval))
            else:
                cleaned_row.append(-1)
        valid_count = sum(1 for v in cleaned_row if v != -1)
        if valid_count >= min_col:
            block.append(cleaned_row)
        else:
            if len(block) >= min_row:
                blocks.append(block)
            block = []
    if len(block) >= min_row:
        blocks.append(block)
    for i in range(len(blocks)):
        maxw = max(len(row) for row in blocks[i])
        blocks[i] = [row + [-1]*(maxw-len(row)) for row in blocks[i]]
    return blocks

def feature_basic(grid, maxv):
    H, W = grid.shape
    C = 4
    tensor = np.zeros((H, W, C), dtype=float)
    for r in range(H):
        for c in range(W):
            v = grid[r, c]
            tensor[r, c, 0] = (v / maxv) if v != -1 else 0.0
            tensor[r, c, 1] = 1.0 if v == -1 else 0.0
            tensor[r, c, 2] = r / (H - 1) if H > 1 else 0.0
            tensor[r, c, 3] = c / (W - 1) if W > 1 else 0.0
    return tensor

def feature_with_tail(grid, maxv):
    H, W = grid.shape
    C = 4 + 10
    tensor = np.zeros((H, W, C), dtype=float)
    for r in range(H):
        for c in range(W):
            v = grid[r, c]
            idx = 0
            tensor[r, c, idx] = (v / maxv) if v != -1 else 0.0; idx += 1
            tensor[r, c, idx] = 1.0 if v == -1 else 0.0; idx += 1
            tensor[r, c, idx] = r / (H - 1) if H > 1 else 0.0; idx += 1
            tensor[r, c, idx] = c / (W - 1) if W > 1 else 0.0; idx += 1
            if v != -1:
                tail = v % 10
                tensor[r, c, idx + tail] = 1.0
    return tensor

def feature_with_odd_even(grid, maxv):
    H, W = grid.shape
    C = 4 + 1
    tensor = np.zeros((H, W, C), dtype=float)
    for r in range(H):
        for c in range(W):
            v = grid[r, c]
            tensor[r, c, 0] = (v / maxv) if v != -1 else 0.0
            tensor[r, c, 1] = 1.0 if v == -1 else 0.0
            tensor[r, c, 2] = r / (H - 1) if H > 1 else 0.0
            tensor[r, c, 3] = c / (W - 1) if W > 1 else 0.0
            tensor[r, c, 4] = (v % 2) if v != -1 else 0.0  # 0偶數, 1奇數
    return tensor

def feature_with_zone(grid, maxv):
    H, W = grid.shape
    C = 4 + 9
    tensor = np.zeros((H, W, C), dtype=float)
    for r in range(H):
        for c in range(W):
            v = grid[r, c]
            idx = 0
            tensor[r, c, idx] = (v / maxv) if v != -1 else 0.0; idx += 1
            tensor[r, c, idx] = 1.0 if v == -1 else 0.0; idx += 1
            tensor[r, c, idx] = r / (H - 1) if H > 1 else 0.0; idx += 1
            tensor[r, c, idx] = c / (W - 1) if W > 1 else 0.0; idx += 1
            # 九宮格區域 one-hot
            if H >= 3 and W >= 3 and v != -1:
                region = (r // (H // 3)) * 3 + (c // (W // 3))
                if region < 9:
                    tensor[r, c, idx + region] = 1.0
    return tensor

def feature_full(grid, maxv):
    H, W = grid.shape
    C = 4 + 10 + 1 + 9
    tensor = np.zeros((H, W, C), dtype=float)
    for r in range(H):
        for c in range(W):
            v = grid[r, c]
            idx = 0
            tensor[r, c, idx] = (v / maxv) if v != -1 else 0.0; idx += 1
            tensor[r, c, idx] = 1.0 if v == -1 else 0.0; idx += 1
            tensor[r, c, idx] = r / (H - 1) if H > 1 else 0.0; idx += 1
            tensor[r, c, idx] = c / (W - 1) if W > 1 else 0.0; idx += 1
            # 尾數
            if v != -1:
                tail = v % 10
                tensor[r, c, idx + tail] = 1.0
            idx += 10
            # 奇偶
            if v != -1:
                tensor[r, c, idx] = (v % 2)
            idx += 1
            # 區域
            if H >= 3 and W >= 3 and v != -1:
                region = (r // (H // 3)) * 3 + (c // (W // 3))
                if region < 9:
                    tensor[r, c, idx + region] = 1.0
    return tensor

feature_fns = [
    ("basic", feature_basic),
    ("tail", feature_with_tail),
    ("odd_even", feature_with_odd_even),
    ("zone", feature_with_zone),
    ("full", feature_full)
]
model_fns = [
    ("logistic", lambda: LogisticRegression(max_iter=1000, multi_class='multinomial')),
    ("rf", lambda: RandomForestClassifier(n_estimators=100))
]

def make_training_samples(cards, n_samples_per_card=10, n_mask=2):
    samples = []
    targets = []
    for card in cards:
        h, w = card.shape
        positions = [(r, c) for r in range(h) for c in range(w) if card[r, c] != -1]
        for _ in range(n_samples_per_card):
            holes = random.sample(positions, n_mask)
            masked = card.copy()
            t = []
            for (r, c) in holes:
                t.append(((r, c), card[r, c]))
                masked[r, c] = -1
            samples.append(masked)
            targets.append(t)
    return samples, targets

if __name__ == "__main__":
    DATA_DIR = 'data'
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".xlsx")]
    all_cards = []
    for file in files:
        blocks = scan_all_numeric_blocks(file)
        if blocks:
            card = np.array(blocks[0])
            all_cards.append(card)
    # **這裡不加假卡，只有真卡！**
    maxv = max(np.max(card[card != -1]) for card in all_cards)
    遮蔽組合 = [2, 3, 4]
    樣本組合 = [100, 300, 500]
    results = []

    for (f_name, f_fn), (m_name, m_fn), n_mask, n_sample in itertools.product(feature_fns, model_fns, 遮蔽組合, 樣本組合):
        samples, targets = make_training_samples(all_cards, n_samples_per_card=n_sample, n_mask=n_mask)
        X, y = [], []
        for masked, tlist in tqdm(zip(samples, targets), total=len(samples), desc=f"{f_name}/{m_name} mask={n_mask}, sample={n_sample}"):
            tensor = f_fn(masked, maxv)
            for (r, c), v in tlist:
                X.append(tensor[r, c])
                y.append(v)
        X = np.array(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = m_fn().fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        model_file = f"models/model_{f_name}_{m_name}_mask{n_mask}_n{n_sample}.joblib"
        joblib.dump(clf, model_file)
        results.append((f_name, m_name, n_mask, n_sample, acc, model_file))
        print(f"特徵:{f_name}, 模型:{m_name}, 遮蔽格:{n_mask}, 樣本:{n_sample}, 命中率:{acc:.3f}, 模型:{model_file}")

    pd.DataFrame(results, columns=["特徵", "模型", "遮蔽格", "樣本數", "命中率", "模型檔"]).to_csv("results/所有訓練結果.csv", index=False)
    print("全部真卡訓練完畢！")
