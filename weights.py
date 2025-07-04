"""
集中管理所有模組 (conn / focus / tail …) 的權重設定。
其他程式僅需 `from weights import BASE_WEIGHTS, AGG_WEIGHTS`。
"""

# ======  (1)  基本權重  =================================================
BASE_WEIGHTS = {
    "conn": 0.40,  # Connectivity Heatmap
    "focus": 0.30,  # 3×3 Density
    "tail": 0.01,  # Sequence Tail  ← 已大幅降低
    "diff": 0.05,  # Difference Trend
    "mirror": 0.05,  # Mirror Pattern
    "gdiff": 0.0,  # Global arithmetic difference
}

# ======  (2)  給 linear-fusion / legacy 用的權重表  ====================
AGG_WEIGHTS = {
    **BASE_WEIGHTS,
    "skip": 0.15,
    "affinity": 0.0,
    "gradient_affinity": 0.0,
    "row_col_bias": 0.0,
    "row_col_frequency_score": 0.0,
    "entropy_spread_score": 0.0,
    "diag": 0.0,
}


# ======  (3)  支援環境變數動態覆寫  ====================================
def _env_override(env_key: str, key: str, table: dict = BASE_WEIGHTS):
    import decimal
    import os

    if env_key in os.environ:
        try:
            table[key] = float(decimal.Decimal(os.environ[env_key]))
        except Exception:
            pass  # 忽略非法數值


_env_override("CONN_W", "conn")
_env_override("FOCUS_W", "focus")
_env_override("TAIL_W", "tail")
# ======================================================================

# Read centralized weight configuration if available
try:  # noqa: WPS501
    from weights_config import WEIGHTS as USER_WEIGHTS

    BASE_WEIGHTS.update(USER_WEIGHTS)
    AGG_WEIGHTS.update(USER_WEIGHTS)
except Exception:
    pass
# ======================================================================
