"""
IAQF 2026 - Models Module (Strategy Pattern)
根据不同 Regime 返回 VAR 估计对象
"""
from .var_strategy import VARModelStrategy, CrisisVARStrategy, NormalVARStrategy, RecoveryVARStrategy
from .var_runner import run_var_pipeline

__all__ = [
    "VARModelStrategy",
    "CrisisVARStrategy",
    "NormalVARStrategy",
    "RecoveryVARStrategy",
    "run_var_pipeline",
]
