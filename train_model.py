import numpy as np
import json
from main import build_feature_tensor
from build_model import build_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

print("載入資料中...")

memory_array = np.load("memory_array.npy") # (N, 120)
grids = memory_array.reshape(-1, 10, 12)

with open("memory_samples.json") as f:
labels = json.load(f)

N = len(grids)
feature_tensors = np.zeros((N, 10, 12, 4), dtype=float)
mask_targets = np.zeros((N, 10, 12), dtype=bool)
value_targets = np.zeros((N, 10, 12), dtype=int)

for i in range(N):
grid = grids[i]
feature_tensors[i] = build_feature_tensor(grid)
r, c = labels[i]["target_rc"]
mask_targets[i, r, c] = True
value_targets[i, r, c] = labels[i]["target_value"]

print("建立訓練集...")
X_train, X_val, m_train, m_val, v_train, v_val = train_test_split(
feature_tensors, mask_targets, value_targets, test_size=0.1)

print("建立模型中...")
model = build_model(input_shape=(10,12,4))
model.compile(
optimizer="adam",
loss={"mask_out":"binary_crossentropy", "value_out":"sparse_categorical_crossentropy"},
metrics=["accuracy"]
)

print("開始訓練！")
model.fit(
X_train, {"mask_out": m_train, "value_out": v_train},
validation_data=(X_val, {"mask_out": m_val, "value_out": v_val}),
epochs=10,
batch_size=64,
callbacks=[ModelCheckpoint("model_best.h5", save_best_only=True)]
)
