#!/usr/bin/env python
"""
dead_code_autorun.py
====================
* 找出沒被使用的函式／類別／方法
* 嘗試自動 import 並以「全預設參數」呼叫一次
* 產生 dead_code_report.md 供人工檢閱
Usage:
    pip install vulture
    python dead_code_autorun.py  [--root src]  [--confidence 80]
"""
import argparse
import importlib
import inspect
from pathlib import Path
from types import ModuleType
from typing import List

from vulture import Vulture

#######################
# 1. 參數與常數
#######################
parser = argparse.ArgumentParser()
parser.add_argument("--root", default=".", help="專案根資料夾")
parser.add_argument("--confidence", type=int, default=80, help="Vulture 最低信心分數")
args = parser.parse_args()
ROOT = Path(args.root).resolve()

#######################
# 2. 跑 Vulture 找死碼
#######################
v = Vulture()
v.scavenge([str(ROOT)])  # 把整個樹丟進去掃
unused_items = [
    item
    for item in v.get_unused_code(min_confidence=args.confidence)
    if item.typ in {"function", "class", "method"}
]


#######################
# 3. 工具函式
#######################
def path_to_module(path: Path) -> str:
    """把檔案路徑轉成 importlib 可以用的模組路徑"""
    rel = path.relative_to(ROOT).with_suffix("")
    return ".".join(rel.parts)


def safe_call(obj):
    """只呼叫「零必填參數」的可呼叫物件；其餘回傳待補資訊"""
    if not callable(obj):
        return "not-callable"
    sig = inspect.signature(obj)
    params = sig.parameters.values()
    mandatory = [
        p
        for p in params
        if p.default is inspect._empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if mandatory:
        return f"skip (requires {len(mandatory)} arg)"
    try:
        result = obj()
        return f"OK → {result!r}"
    except Exception as exc:
        return f"ERROR: {exc.__class__.__name__}: {exc}"


#######################
# 4. 一一 import & 測試
#######################
report_lines: List[str] = [
    "# Dead-code auto-run report",
    f"*Root scanned:* `{ROOT}`",
    f"*Confidence ≥* **{args.confidence}%**",
    "",
    "| File | Symbol | Auto-call result |",
    "|------|--------|-----------------|",
]

for item in unused_items:
    mod_name = path_to_module(item.filename)
    try:
        module: ModuleType = importlib.import_module(mod_name)
    except Exception as exc:
        report_lines.append(
            f"| {item.filename.name} | (module import failed) | {exc.__class__.__name__} |"
        )
        continue

    obj = getattr(module, item.name, None)
    if obj is None:
        report_lines.append(
            f"| {item.filename.name} | `{item.name}` | not found after import |"
        )
        continue

    result = safe_call(obj)
    report_lines.append(f"| {item.filename.name} | `{item.name}` | {result} |")

#######################
# 5. 輸出結果
#######################
report_path = ROOT / "dead_code_report.md"
report_path.write_text("\n".join(report_lines), encoding="utf-8")
print(f"✅  完成！結果已寫入 {report_path.relative_to(ROOT)}")
print("   未自動可呼叫者，請人工檢查參數或評估刪除 / 寫測試。")
