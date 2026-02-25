"""
IAQF 2026 - Config Module (Singleton/Registry Pattern)
全局唯一配置源，载入 YAML，提供 get_params() 接口。
"""
from .registry import ConfigRegistry

# 单例入口
_config: ConfigRegistry | None = None


def get_config() -> ConfigRegistry:
    """获取全局配置单例"""
    global _config
    if _config is None:
        _config = ConfigRegistry()
    return _config


def get_params(path: str):
    """
    快捷访问：get_params("phases.normal.start") -> "2023-03-01"
    path 支持点分隔的嵌套键
    """
    return get_config().get(path)
