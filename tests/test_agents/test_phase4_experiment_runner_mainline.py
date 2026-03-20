import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from src.train import main as train_main


def test_train_experiments_are_executed(ranking_dataset_path, tmp_path, monkeypatch) -> None:
    cfg = yaml.safe_load(Path('configs/train.yaml').read_text(encoding='utf-8'))
    cfg['validation'] = {'n_splits': 2, 'min_train_issues': 10}
    cfg_path = tmp_path / 'train.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg), encoding='utf-8')

    exp_path = tmp_path / 'experiments.yaml'
    exp_path.write_text(
        yaml.safe_dump({'experiments': [
            {'name': 'baseline_frequency'},
            {'name': 'ranker_main_qsm'},
            {'name': 'ablation_no_retrieval'},
            {'name': 'ablation_no_logistic'},
        ]}),
        encoding='utf-8',
    )

    fake_scored = pd.DataFrame(
        {
            'issue': ['I1', 'I1', 'I2', 'I2'],
            'candidate_number': [1, 2, 1, 2],
            'label': [1, 0, 0, 1],
            'final_score': [0.9, 0.1, 0.2, 0.8],
            'ranker_score': [0.9, 0.1, 0.2, 0.8],
            'logistic_score': [0.8, 0.2, 0.3, 0.7],
            'retrieval_score': [0.7, 0.3, 0.3, 0.7],
            'history_prior_score': [0.6, 0.4, 0.4, 0.6],
            'analysis_rerank_score': [0.5, 0.5, 0.5, 0.5],
            'local_peak_score': [0.5, 0.5, 0.5, 0.5],
            'cand_hits_last_100': [10, 1, 1, 10],
            'cand_hits_last_20': [3, 1, 1, 3],
        }
    )

    def fake_run_cv(*args, **kwargs):
        return [SimpleNamespace(fold_id=1, train_scored=fake_scored.copy(), val_scored=fake_scored.copy())]

    monkeypatch.setattr('src.train.run_cv', fake_run_cv)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('sys.argv', ['train', '--config', str(cfg_path), '--experiments', str(exp_path), '--input', str(ranking_dataset_path)])
    train_main()

    registry = pd.read_csv(tmp_path / 'reports' / 'train_experiment_registry.csv')
    assert set(registry['experiment']) == {'baseline_frequency', 'ranker_main_qsm', 'ablation_no_retrieval', 'ablation_no_logistic'}
    assert (registry['status'] == 'completed').all()

    train_fold = pd.read_csv(tmp_path / 'reports' / 'train_experiment_per_fold_metrics.csv')
    backtest_fold = pd.read_csv(tmp_path / 'reports' / 'backtest_experiment_per_fold_metrics.csv')
    assert train_fold['experiment'].nunique() == 4
    assert backtest_fold['experiment'].nunique() == 4

    summary = json.loads((tmp_path / 'reports' / 'backtest_experiment_summary.json').read_text(encoding='utf-8'))
    assert 'train_vs_backtest_gap_top3' in summary
    assert summary['experiment_count'] == 4
