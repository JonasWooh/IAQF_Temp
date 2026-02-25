"""
IAQF 2026 - Usage Module (Facade Pattern)
用户只需调用 Runner.start()，屏蔽内部复杂模块交互
"""
from .runner import Runner

__all__ = ["Runner"]
