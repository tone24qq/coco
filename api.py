import pandas as pd
import joblib
import numpy as np
from flask import Flask, request, jsonify

# 自動讀命中率總表，找最佳模型
results = pd.read_csv("results/所有訓練結果.csv")
best_row = results.loc[results["命中率"].idxmax()]
model_file = best_row["模型檔"]
clf = joblib.load(model_file)
print(f"已自動加載最佳經驗模型：{model_file}")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X = np.array(data['features'])
    y_pred = clf.predict(X).tolist()
    return jsonify({"result": y_pred})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
