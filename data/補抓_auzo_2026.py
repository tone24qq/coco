from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import requests

START_DATE = date(2026, 3, 1)
END_DATE = date.today()

INPUT_CSV = Path("賓果賓果_2026.csv")
OUTPUT_CSV = Path("賓果賓果_2026_補齊.csv")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/132.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://winwin.tw/Bingo",
}

RAW_COLUMN_MAP = {
    "期別": "issue",
    "開獎日期": "draw_time",
    **{f"獎號{i}": f"n{i}" for i in range(1, 21)},
}
CANONICAL_COLUMNS = ["issue", "draw_time", *[f"n{i}" for i in range(1, 21)]]
OUTPUT_COLUMN_MAP = {
    "issue": "期別",
    "draw_time": "開獎日期",
    **{f"n{i}": f"獎號{i}" for i in range(1, 21)},
}


def normalize_local_schema(df: pd.DataFrame) -> pd.DataFrame:
    if all(c in df.columns for c in CANONICAL_COLUMNS):
        out = df[CANONICAL_COLUMNS].copy()
    elif all(c in df.columns for c in RAW_COLUMN_MAP):
        out = df.rename(columns=RAW_COLUMN_MAP)[CANONICAL_COLUMNS].copy()
    else:
        raise ValueError(f"不支援的欄位格式: {df.columns.tolist()}")

    out["issue"] = out["issue"].astype(str)
    for i in range(1, 21):
        out[f"n{i}"] = pd.to_numeric(out[f"n{i}"], errors="raise").astype(int)

    return out.sort_values("issue", kind="mergesort").reset_index(drop=True)


def parse_numbers(raw):
    if isinstance(raw, list):
        nums = [int(x) for x in raw]
    elif isinstance(raw, str):
        parts = [x for x in raw.replace(",", " ").split() if x]
        nums = [int(x) for x in parts]
    else:
        return None

    if len(nums) != 20 or len(set(nums)) != 20:
        return None
    if any(n < 1 or n > 80 for n in nums):
        return None
    return nums


def fetch_one_day(day: date, session: requests.Session):
    url = f"https://winwin.tw/Bingo/GetBingoData?date={day.isoformat()}"
    r = session.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    payload = r.json()

    rows = []
    if not isinstance(payload, list):
        return rows

    for item in payload:
        if not isinstance(item, dict):
            continue

        issue = str(item.get("Issue", item.get("No", ""))).strip()
        draw_time = str(item.get("DrawTime", item.get("OpenDate", ""))).strip()
        numbers = parse_numbers(item.get("BingoNumbers", item.get("BigShowOrder")))

        if len(issue) != 9 or numbers is None:
            continue

        row = {"issue": issue, "draw_time": draw_time}
        for i, n in enumerate(numbers, start=1):
            row[f"n{i}"] = n
        rows.append(row)

    if not rows:
        return rows

    df = pd.DataFrame(rows).drop_duplicates(subset=["issue"], keep="first")
    df = df.sort_values("issue", kind="mergesort").reset_index(drop=True)
    return df.to_dict("records")


def main():
    local_df = normalize_local_schema(pd.read_csv(INPUT_CSV))

    all_new = []
    with requests.Session() as session:
        d = START_DATE
        while d <= END_DATE:
            try:
                rows = fetch_one_day(d, session)
                if rows:
                    all_new.extend(rows)
                    print(f"[OK] {d} rows={len(rows)} first={rows[0]['issue']} last={rows[-1]['issue']}")
                else:
                    print(f"[EMPTY] {d}")
            except Exception as e:
                print(f"[ERR] {d} {e}")
            d += timedelta(days=1)

    if not all_new:
        print("沒有抓到任何新資料")
        return

    new_df = pd.DataFrame(all_new)[CANONICAL_COLUMNS].copy()
    merged = pd.concat([local_df, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["issue"], keep="first")
    merged = merged.sort_values("issue", kind="mergesort").reset_index(drop=True)

    out = merged.rename(columns=OUTPUT_COLUMN_MAP)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n=== 完成 ===")
    print("原始筆數:", len(local_df))
    print("新增筆數:", len(new_df.drop_duplicates(subset=["issue"])))
    print("合併後筆數:", len(merged))
    print("最新期別:", merged["issue"].iloc[-1])
    print("輸出檔案:", OUTPUT_CSV.resolve())


if __name__ == "__main__":
    main()