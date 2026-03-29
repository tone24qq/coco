from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from winwin_service.fetcher import (
    FetchError,
    fetch_latest_draws,
    parse_draws_from_html,
    parse_draws_from_json,
)
from winwin_service.config import AppConfig


def _row(period: int, start: int) -> str:
    nums = ''.join(f'<td>{n}</td>' for n in range(start, start + 20))
    return f'<tr><td>{period}</td>{nums}</tr>'


def test_parse_draws_from_html_extracts_expected_rows() -> None:
    html = '<table>' + _row(114000001, 1) + '</table>'
    draws = parse_draws_from_html(html)
    assert draws == [(114000001, list(range(1, 21)))]


def test_parse_draws_from_json_extracts_expected_rows() -> None:
    payload = (
        '[{"No":"114000001","BigShowOrder":"01,02,03,04,05,06,07,08,09,10,'
        '11,12,13,14,15,16,17,18,19,20"}]'
    )
    draws = parse_draws_from_json(payload)
    assert draws == [(114000001, list(range(1, 21)))]


@patch('winwin_service.fetcher.requests.get')
def test_fetch_latest_draws_success(mock_get: Mock) -> None:
    rows = []
    for i in range(55):
        nums = ','.join(f'{n:02d}' for n in range(1, 21))
        rows.append({'No': str(114000001 + i), 'BigShowOrder': nums})

    response = Mock()
    response.text = str(rows).replace("'", '"')
    response.raise_for_status = Mock()
    mock_get.return_value = response

    draws, latest_period = fetch_latest_draws(
        config=AppConfig(
            min_prediction_draws=10,
            max_recent_draws_count=50,
        )
    )

    assert len(draws) == 50
    assert latest_period == 114000055


@patch('winwin_service.fetcher.requests.get')
def test_fetch_latest_draws_max_none_uses_all(mock_get: Mock) -> None:
    rows = []
    for i in range(48):
        nums = ','.join(f'{n:02d}' for n in range(1, 21))
        rows.append({'No': str(114100001 + i), 'BigShowOrder': nums})

    response = Mock()
    response.text = str(rows).replace("'", '"')
    response.raise_for_status = Mock()
    mock_get.return_value = response

    draws, latest_period = fetch_latest_draws(
        config=AppConfig(
            min_prediction_draws=10,
            max_recent_draws_count=None,
        )
    )

    assert len(draws) == 48
    assert latest_period == 114100048


@patch('winwin_service.fetcher.requests.get')
def test_fetch_latest_draws_fail_fast_no_draws(mock_get: Mock) -> None:
    empty_json = Mock(text='[]')
    empty_json.raise_for_status = Mock()
    empty_html = Mock(text='<html><body>empty</body></html>')
    empty_html.raise_for_status = Mock()
    mock_get.side_effect = [empty_json] * 7 + [empty_html]

    with pytest.raises(FetchError):
        fetch_latest_draws()


@patch('winwin_service.fetcher.requests.get')
def test_fetch_latest_draws_fail_when_below_min_prediction(
    mock_get: Mock,
) -> None:
    rows = []
    for i in range(8):
        nums = ','.join(f'{n:02d}' for n in range(1, 21))
        rows.append({'No': str(114200001 + i), 'BigShowOrder': nums})

    response = Mock()
    response.text = str(rows).replace("'", '"')
    response.raise_for_status = Mock()
    mock_get.return_value = response

    with pytest.raises(FetchError) as exc:
        fetch_latest_draws(
            config=AppConfig(
                min_prediction_draws=10,
                max_recent_draws_count=None,
            )
        )
    assert 'got_draws=8' in str(exc.value)
