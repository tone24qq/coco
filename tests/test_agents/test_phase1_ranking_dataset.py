import pandas as pd

from src.modeling import load_ranking_dataset
from src.utils import DataContractError


def test_each_issue_exactly_80_candidates_fail_fast(tmp_path) -> None:
    bad = pd.DataFrame(
        [{"issue": "A", "candidate_number": i, "label": 0, "group_id": 0} for i in range(1, 80)]
    )
    p = tmp_path / "bad.csv"
    bad.to_csv(p, index=False)
    try:
        load_ranking_dataset(p)
    except DataContractError as exc:
        assert "1..80 each once" in str(exc)
    else:
        raise AssertionError("expected contract error")


def test_each_issue_label_sum_20_fail_fast(tmp_path):
    import pandas as pd
    from src.modeling import load_ranking_dataset
    from src.utils import DataContractError

    rows = [{"issue": "A", "candidate_number": i, "label": 1 if i <= 10 else 0, "group_id": 0} for i in range(1, 81)]
    p = tmp_path / "bad_label.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    try:
        load_ranking_dataset(p)
    except DataContractError as exc:
        assert "label sum" in str(exc)
    else:
        raise AssertionError("expected label sum fail-fast")
