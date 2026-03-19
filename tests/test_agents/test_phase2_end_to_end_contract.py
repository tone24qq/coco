from pathlib import Path

from src.build_features import build_feature_rows, write_feature_store
from src.prepare_data import merge_histories
from src.ranking_dataset import attach_group_ids, validate_group_contract, write_rows


def test_fetch_prepare_feature_dataset_contract(tmp_path, synthetic_records) -> None:
    csv_path = tmp_path / "history.csv"
    csv_path.write_text(
        "期別,開獎日期," + ",".join([f"獎號{i}" for i in range(1, 21)]) + "\n"
        + "\n".join(
            [
                f"{r.issue},{r.draw_date.strftime('%Y/%m/%d')}," + ",".join(str(x) for x in r.numbers)
                for r in synthetic_records
            ]
        ),
        encoding="utf-8",
    )
    records = merge_histories([csv_path])
    features = build_feature_rows(records, min_history=100, retrieval_window=40, top_k=8)
    fp = tmp_path / "features.csv"
    write_feature_store(fp, features)

    import pandas as pd

    rows = pd.read_csv(fp).to_dict(orient="records")
    rows_g = attach_group_ids(rows)
    validate_group_contract(rows_g)
    rp = tmp_path / "ranking.csv"
    write_rows(rp, rows_g)
    assert Path(rp).exists()
