"""
GENIUS Act 三种情景策略
A: 幅度缩小 (Magnitude scaling)
B: 尾部截断 (Tail truncation)
C: 持续时间缩短 (Duration damp)
"""
import numpy as np

from .base import SimulationStrategy
from ..config import get_config


class MagnitudeScalingStrategy(SimulationStrategy):
    """情景 A：c × IRF，监管降低冲击幅度"""
    name = "magnitude"

    def __init__(self, c: float = 0.5):
        self.c = c

    def transform_irf(
        self,
        base_irf: np.ndarray,
        idx_var: int,
        idx_shock: int,
        **kwargs,
    ) -> np.ndarray:
        return base_irf[:, idx_var, idx_shock] * self.c


class TailTruncationStrategy(SimulationStrategy):
    """情景 B：Winsorize 残差 95% 分位，有效 c 来自 std 比"""
    name = "tail_truncation"

    def __init__(self, results=None, quantile: float = 0.95):
        self.results = results
        self.quantile = quantile
        self._c: float | None = None

    def _compute_effective_c(self) -> float:
        if self.results is None:
            return 1.0
        resid = self.results.resid["Depeg_bps"].dropna()
        q = resid.abs().quantile(self.quantile)
        resid_win = resid.clip(lower=-q, upper=q)
        std_orig = resid.std()
        std_win = resid_win.std()
        return float(std_win / std_orig) if std_orig > 1e-10 else 1.0

    def transform_irf(
        self,
        base_irf: np.ndarray,
        idx_var: int,
        idx_shock: int,
        results=None,
        **kwargs,
    ) -> np.ndarray:
        self.results = results or self.results
        c = self._compute_effective_c()
        self._c = c
        return base_irf[:, idx_var, idx_shock] * c


class DurationDampStrategy(SimulationStrategy):
    """情景 C：IRF × exp(-t/τ)，监管缩短持续时间"""
    name = "duration_damp"

    def __init__(self, tau: float = 10.0):
        self.tau = tau

    def transform_irf(
        self,
        base_irf: np.ndarray,
        idx_var: int,
        idx_shock: int,
        **kwargs,
    ) -> np.ndarray:
        n = base_irf.shape[0]
        irf_vals = base_irf[:, idx_var, idx_shock]
        t_arr = np.arange(n, dtype=float)
        return irf_vals * np.exp(-t_arr / self.tau)
