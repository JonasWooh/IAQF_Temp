"""
IAQF 2026 - GENIUS Act Simulation (Strategy Pattern)
情景 A: 幅度缩减 | B: 尾部截断 | C: 持续时间缩短
"""
from .base import SimulationStrategy
from .strategies import MagnitudeScalingStrategy, TailTruncationStrategy, DurationDampStrategy
from .runner import run_simulation

__all__ = [
    "SimulationStrategy",
    "MagnitudeScalingStrategy",
    "TailTruncationStrategy",
    "DurationDampStrategy",
    "run_simulation",
]
