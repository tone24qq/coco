
from env_config import EnvConfig
from main import CLIConfig, parse_args


def test_cli_parse_returns_dataclass():
    cfg = parse_args(["--grid", "1,-1;2,3", "--iterations", "4", "--target", "3"])
    assert isinstance(cfg, CLIConfig)
    assert cfg.grid == "1,-1;2,3"
    assert cfg.iterations == 4
    assert cfg.target == 3


def test_env_config(monkeypatch):
    monkeypatch.setenv("PHASE1_ITERATIONS", "111")
    monkeypatch.setenv("PHASE2_ITERATIONS", "222")
    cfg = EnvConfig()
    assert cfg.phase1_iter == 111
    assert cfg.phase2_iter == 222
