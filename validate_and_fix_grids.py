import os
import json

def is_valid_grid(grid):
    """確認是否為標準矩陣（每行等長且都是 list）"""
    if not isinstance(grid, list):
        return False
    row_lengths = [len(row) for row in grid if isinstance(row, list)]
    return (
        len(row_lengths) == len(grid) and
        len(set(row_lengths)) == 1
    )

def fix_flat_grid(grid):
    """若是平面 list，嘗試轉換成合理矩陣（針對意外攤平格式）"""
    flat = []
    for row in grid:
        if isinstance(row, list):
            flat.extend(row)
        elif isinstance(row, (int, float, str)):
            flat.append(row)
    flat = [int(x) for x in flat if str(x).isdigit()]

    for r in range(3, 11):
        if len(flat) % r == 0:
            c = len(flat) // r
            return [flat[i*c:(i+1)*c] for i in range(r)]
    return None

def validate_and_fix_folder(folder_path):
    fixed = []
    valid = []
    skipped = []
    error = []

    for file in os.listdir(folder_path):
        if not file.endswith(".json"):
            continue
        path = os.path.join(folder_path, file)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            grid = data.get("grid")
            if is_valid_grid(grid):
                valid.append(file)
                continue

            repaired = fix_flat_grid(grid)
            if repaired and is_valid_grid(repaired):
                data["grid"] = repaired
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                fixed.append(file)
            else:
                skipped.append(file)
        except Exception as e:
            error.append((file, str(e)))

    # 總結
    print(f"\n📊 驗證結果：")
    print(f"✅ 有效：{len(valid)}")
    print(f"🛠️ 自動修復成功：{len(fixed)}")
    print(f"⛔ 無法修復：{len(skipped)}")
    print(f"💥 讀取錯誤：{len(error)}")

    if fixed:
        print("🛠️ 修復：", fixed)
    if skipped:
        print("⛔ 略過：", skipped)
    if error:
        print("💥 錯誤：", error)

if __name__ == "__main__":
    folder = "data"  # 改成你的資料夾路徑
    if not os.path.exists(folder):
        print("❗ 找不到資料夾！請確認資料夾名稱")
    else:
        validate_and_fix_folder(folder)
