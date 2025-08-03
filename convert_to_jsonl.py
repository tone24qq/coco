import os
import json
import orjson  # 建議安裝：pip install orjson
from pathlib import Path

base = Path("data_archives")

for fn in base.glob("*.json"):
    if fn.name.endswith(".jsonl"):
        continue  # 跳過已經是jsonl的
    out = fn.with_suffix(".jsonl")
    if out.exists():
        print(f"已存在：{out.name}，略過。")
        continue
    print(f"正在轉換：{fn.name} → {out.name}")
    with open(fn, "r", encoding="utf-8") as fin, open(out, "wb") as fout:
        data = json.load(fin)
        for obj in data:
            fout.write(orjson.dumps(obj))
            fout.write(b"\n")
print("✅ 全部轉換完成！")
