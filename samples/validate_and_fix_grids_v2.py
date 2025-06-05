import os
import json
import shutil
import argparse

def is_valid_grid(grid):
    if not isinstance(grid, list):
        return False
    row_lengths = [len(row) for row in grid if isinstance(row, list)]
    return len(row_lengths) == len(grid) and len(set(row_lengths)) == 1

def fix_flat_grid(grid):
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

def ensure_backup_folder(folder_path):
    backup_path = os.path.join(folder_path, "backup")
    os.makedirs(backup_path, exist_ok=True)
    return backup_path

def validate_and_fix_folder(folder_path, rewrite_all=False):
    fixed, valid, skipped, error = [], [], [], []
    backup_path = ensure_backup_folder(folder_path)

    for file in os.listdir(folder_path):
        if not file.endswith(".json") or file == "backup":
            continue
        path = os.path.join(folder_path, file)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            grid = data.get("grid")
            if is_valid_grid(grid):
                if rewrite_all:
                    shutil.copy(path, os.path.join(backup_path, file))
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    valid.append(file)
                else:
                    valid.append(file)
                continue

            repaired = fix_flat_grid(grid)
            if repaired and is_valid_grid(repaired):
                data["grid"] = repaired
                shutil.copy(path, os.path.join(backup_path, file))
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                fixed.append(file)
            else:
                skipped.append(file)
        except Exception as e:
            error.append((file, str(e)))

    print("\n📊 驗證結果：")
    print(f"✅ 有效（{ '重新儲存' if rewrite_all else '保留原樣' }）：{len(valid)}")
    print(f"🛠️ 修復成功：{len(fixed)}")
    print(f"⛔ 無法修復：{len(skipped)}")
    print(f"💥 語法錯誤：{len(error)}")

    if fixed:
        print("🛠️ 修復檔案：", fixed)
    if skipped:
        print("⛔ 略過檔案：", skipped)
    if error:
        print("💥 錯誤檔案：", error)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewrite", action="store_true", help="重新儲存所有合法 JSON")
    args = parser.parse_args()

    folder = "data"
    if not os.path.exists(folder):
        print("❗ 找不到資料夾：data/")
    else:
        validate_and_fix_folder(folder_path=folder, rewrite_all=args.rewrite)
