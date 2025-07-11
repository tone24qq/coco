import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvConfig:
    """Centralized environment configuration."""

    phase1_iter: int = field(
        default_factory=lambda: int(os.getenv("PHASE1_ITERATIONS", "5000"))
    )
    phase2_iter: int = field(
        default_factory=lambda: int(os.getenv("PHASE2_ITERATIONS", "1000"))
    )
    phase2_top_n: int = field(
        default_factory=lambda: int(os.getenv("PHASE2_TOP_N", "10"))
    )
    phase2_epsilon: float = field(
        default_factory=lambda: float(os.getenv("PHASE2_EPSILON", "0.05"))
    )
    result_top_k: int = field(
        default_factory=lambda: int(os.getenv("RESULT_TOP_K", "3"))
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    sim_time_limit: Optional[float] = field(
        default_factory=lambda: (
            float(v) if (v := os.getenv("SIM_TIME_LIMIT")) else None
        )
    )
