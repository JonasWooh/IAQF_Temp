"""
SimulationStrategy - GENIUS Act 情景模拟基类
"""
from abc import ABC, abstractmethod

import numpy as np


class SimulationStrategy(ABC):
    """反事实情景策略基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def transform_irf(
        self,
        base_irf: np.ndarray,
        idx_var: int,
        idx_shock: int,
        **kwargs,
    ) -> np.ndarray:
        """
        将基准 IRF 转换为反事实 IRF
        base_irf: shape (T, n_vars, n_shocks)
        """
        pass
