"""
IAQF 2026 - Diagnostics Module (Observer Pattern)
监听模型输出，生成文本报告和稳定性图表
"""
from .observer import DiagnosticsObserver

__all__ = ["DiagnosticsObserver"]
