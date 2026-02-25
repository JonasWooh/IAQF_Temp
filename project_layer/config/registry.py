"""
ConfigRegistry - Singleton 配置注册表
载入 settings.yaml，提供 get_params() 接口
"""
from pathlib import Path
from typing import Any

import yaml


class ConfigRegistry:
    """单例配置：YAML 载入后注入所有模块"""

    _instance = None

    def __new__(cls):
        # ensure the singleton mode
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # check if the config has been loaded or not
        if hasattr(self, "_loaded") and self._loaded:
            return
        # if not loaded, reinitialize
        self._loaded = False
        self._raw: dict[str, Any] = {}
        self._load() # load YAML parameters

    def _load(self):
        root = Path(__file__).resolve().parent.parent  # project_layer/
        settings_path = root / "settings.yaml"
        # if the YAML file does not exist, stop and return
        if not settings_path.exists():
            self._raw = {}
            self._loaded = True
            return
        # parse YAML config file
        with open(settings_path, encoding="utf-8") as f:
            self._raw = yaml.safe_load(f) or {}
        self._raw.setdefault("project", {})["root"] = str(root) # set up the root path
        self._loaded = True

    def get(self, path: str, default: Any = None) -> Any:
        """
        path: 点分隔键，如 "phases.normal.start"
        """
        keys = path.split(".")
        val = self._raw
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def get_project_root(self) -> Path:
        return Path(self.get("project.root", Path(__file__).parent.parent))

    def get_data_root(self) -> Path:
        root = self.get_project_root()
        p1 = root / self.get("paths.data_root", "data/data_flatfiles")
        # p2 = root.parent / self.get("paths.data_root", "data/data_flatfiles")
        for p in [p1, root.parent / "data" / "data_flatfiles", root.parent / "data_flatfiles"]:
            if p.exists():
                return p
        return root / "data" / "data_flatfiles"

    def get_output_dir(self) -> Path:
        root = self.get_project_root()
        return root / self.get("paths.output_dir", "output_project_layer")

    def get_processed_1min_dir(self) -> Path:
        """优先使用主项目的 processed_1min，否则在 project_layer 输出下创建"""
        root = self.get_project_root()
        main_out = root.parent / "output" / "processed_1min"
        if main_out.exists():
            return main_out
        out = self.get_output_dir() / "processed_1min"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def get_master_feature_path(self) -> Path:
        return self.get_output_dir() / self.get("paths.master_feature_file", "master_features_1min.parquet")

    def get_figures_dir(self) -> Path:
        return self.get_output_dir() / self.get("paths.figures_dir", "figures")

    def get_all_processing_targets(self) -> list[tuple[str, str]]:
        """(exchange, pair) 列表"""
        out = []
        usdc = self.get("exchanges.usdc_usd", {})
        if isinstance(usdc, dict):
            out.append((usdc.get("exchange", "BINANCEUS"), usdc.get("pair", "USDC_USD")))
        btc = self.get("exchanges.btc", {})
        exs = btc.get("exchanges", ["BINANCEUS", "COINBASE"])
        pairs = btc.get("pairs", ["BTC_USD", "BTC_USDC", "BTC_USDT"])
        for ex in exs:
            for pair in pairs:
                out.append((ex, pair))
        return out

    def ensure_dirs(self):
        self.get_output_dir().mkdir(parents=True, exist_ok=True)
        self.get_figures_dir().mkdir(parents=True, exist_ok=True)
