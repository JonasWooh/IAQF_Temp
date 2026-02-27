"""
FeaturePipeline - Pipeline 模式：Data -> Features -> Master
run_stage() 流转机制
"""
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from ..config import get_config
from .utils import load_and_prefix_parquet, merge_1min_tables, make_prefix
from .pca import CMLSIPCA


def plot_pca_variance(meta: dict, figures_dir: Path, dpi: int = 150) -> None:
    """图08: PCA 方差解释率"""
    evr = meta.get("explained_variance_ratio", [])
    if not evr:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(1, len(evr) + 1)
    ax.bar(x, [v * 100 for v in evr], color="#3498DB", alpha=0.8)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    cumevr = meta.get("cmlsi_explained_variance_ratio", 0) * 100
    ax.set_title(f"CMLSI-pure: Top 3 PCs = {cumevr:.1f}%")
    plt.tight_layout()
    out = figures_dir / "08_pca_variance_comparison.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"  [OK] {out.name}")
    plt.close()


def _close_col(exchange: str, pair: str) -> str:
    return f"{make_prefix(exchange, pair)}_close"

# edit: dynamic generation of extended features based on available columns
def _add_extended_features(master: pd.DataFrame, bn_prefix: str) -> pd.DataFrame:
    close = f"{bn_prefix}_close"
    high = f"{bn_prefix}_high"
    low = f"{bn_prefix}_low"
    spread = f"{bn_prefix}_spread_mean"
    depth = f"{bn_prefix}_depth_mean"
    m = master.copy()
    if close in m.columns:
        m[f"{bn_prefix}_ret_1m"] = m[close].pct_change()
        m[f"{bn_prefix}_ret_abs"] = m[f"{bn_prefix}_ret_1m"].abs()
    if high in m.columns and low in m.columns:
        m[f"{bn_prefix}_range_1m"] = m[high] - m[low]
    if spread in m.columns and close in m.columns:
        mid = m[close].replace(0, np.nan)
        m[f"{bn_prefix}_rel_spread"] = m[spread] / mid
    if depth in m.columns:
        m[f"{bn_prefix}_log_depth"] = np.log(m[depth].clip(lower=1e-8) + 1e-8)
    return m


class FeaturePipeline:
    """管道：数据 -> 特征 -> Master 表"""

    def __init__(self):
        self.cfg = get_config()
        self._master: pd.DataFrame | None = None

    def run_stage_data(self, processed_dir: Path | None = None) -> pd.DataFrame:
        """Stage 1: 加载并合并 1min parquet"""
        in_dir = processed_dir or self.cfg.get_processed_1min_dir()
        if not in_dir.exists():
            raise FileNotFoundError(f"Processed dir not found: {in_dir}")
        targets = self.cfg.get_all_processing_targets()
        dfs = []
        for exchange, pair in targets:
            fp = in_dir / f"{exchange}_{pair}.parquet"
            if fp.exists():
                dfs.append(load_and_prefix_parquet(fp, exchange, pair))
        if not dfs:
            raise ValueError("No parquet files found")
        start = self.cfg.get("dates.start")
        end = self.cfg.get("dates.end")
        ffill = self.cfg.get("thresholds.ffill_limit", 5)
        self._master = merge_1min_tables(dfs, start, end, ffill_limit=ffill)
        return self._master

    def run_stage_features(self, master: pd.DataFrame | None = None) -> pd.DataFrame:
        """Stage 2: 计算 Depeg, Basis, Premium, 扩展因子"""
        m = master if master is not None else self._master
        if m is None:
            raise RuntimeError("Run run_stage_data first")
        cfg = self.cfg
        bn, cb = cfg.get("exchanges.btc.exchanges", ["BINANCEUS", "COINBASE"])[:2]
        usdc_ex = cfg.get("exchanges.usdc_usd.exchange", "BINANCEUS")
        usdc_pair = cfg.get("exchanges.usdc_usd.pair", "USDC_USD")
        usdc_close = _close_col(usdc_ex, usdc_pair)
        bn_usd = _close_col(bn, "BTC_USD")
        cb_usd = _close_col(cb, "BTC_USD")
        if usdc_close in m.columns:
            m["Depeg_Ratio"] = 1 - m[usdc_close]
            m["Depeg_bps"] = m["Depeg_Ratio"] * 10_000
        if bn_usd in m.columns and cb_usd in m.columns:
            m["Premium_BTC"] = m[bn_usd] - m[cb_usd]
            mid_prem = (m[bn_usd] + m[cb_usd]) / 2
            m["Premium_bps"] = (m["Premium_BTC"] / mid_prem.replace(0, np.nan)) * 10_000
        bn_usdc = _close_col(bn, "BTC_USDC")
        bn_usdt = _close_col(bn, "BTC_USDT")
        cb_usdc = _close_col(cb, "BTC_USDC")
        cb_usdt = _close_col(cb, "BTC_USDT")
        if bn_usd in m.columns and bn_usdc in m.columns:
            m["Basis_USD_USDC_BN"] = m[bn_usd] - m[bn_usdc]
            mid_basis = (m[bn_usd] + m[bn_usdc]) / 2
            m["Basis_bps"] = (m["Basis_USD_USDC_BN"] / mid_basis.replace(0, np.nan)) * 10_000
        if bn_usd in m.columns and bn_usdt in m.columns:
            m["Basis_USD_USDT_BN"] = m[bn_usd] - m[bn_usdt]
        if bn_usdc in m.columns and bn_usdt in m.columns:
            m["Basis_USDC_USDT_BN"] = m[bn_usdc] - m[bn_usdt]
            mid_ucdt = (m[bn_usdc] + m[bn_usdt]) / 2
            m["Basis_USDC_USDT_BN_bps"] = (
                m["Basis_USDC_USDT_BN"] / mid_ucdt.replace(0, np.nan)
            ) * 10_000

        if cb_usdc in m.columns and cb_usdt in m.columns:
            m["Basis_USDC_USDT_CB"] = m[cb_usdc] - m[cb_usdt]
            
        # edit: dynamic generation of extended features based on available columns
        target_pairs = ["BTC_USD", "BTC_USDC", "BTC_USDT"]
        for pair in target_pairs:
            prefix = make_prefix(bn, pair)
            if f"{prefix}_close" in m.columns:
                m = _add_extended_features(m, prefix)
            else:
                print(f"  [WARN] Missing base columns for {prefix}, skipping extended features.")
                
        self._master = m
        return m

    def run_stage_pca(self, master: pd.DataFrame | None = None) -> pd.DataFrame:
        """Stage 3: PCA (fit Normal, transform all) -> CMLSI"""
        m = master if master is not None else self._master
        if m is None:
            raise RuntimeError("Run run_stage_data and run_stage_features first")
        cfg = self.cfg
        bn_prefix = make_prefix(cfg.get("exchanges.btc.exchanges", ["BINANCEUS"])[0], "BTC_USD")
        features_basic = [
            f"{bn_prefix}_spread_mean",
            f"{bn_prefix}_depth_mean",
            f"{bn_prefix}_obi_mean",
        ]
        
        
        features_pure = features_basic.copy()
        
        # edit: dynamic generation of extended features based on available columns
        for col_suffix in ["_ret_abs", "_range_1m", "_rel_spread", "_log_depth"]:
            col_name = f"{bn_prefix}{col_suffix}"
            if col_name in m.columns:
                features_pure.append(col_name)
        features_pure = [f for f in features_pure if f in m.columns]
        
        if len(features_pure) < 3:
            raise ValueError(f"Missing features: {features_pure}")
        phase_n = (cfg.get("phases.normal.start"), cfg.get("phases.normal.end"))
        phase_s = (cfg.get("phases.stressed.start"), cfg.get("phases.stressed.end"))
        phase_r = (cfg.get("phases.recovery.start"), cfg.get("phases.recovery.end"))
        regime_windows = [phase_n, phase_s, phase_r]
        pca = CMLSIPCA(features_pure, n_pcs=3)
        pca.fit(m, phase_n[0], phase_n[1])
        cmlsi = pca.transform(m, regime_windows)
        m = m.copy()
        m["CMLSI"] = cmlsi.values
        meta = pca.get_meta(m, regime_windows)
        cfg.ensure_dirs()
        scaler_path = cfg.get_output_dir() / cfg.get("paths.cmlsi_scaler", "cmlsi_scaler.pkl")
        loadings_path = cfg.get_output_dir() / cfg.get("paths.cmlsi_loadings", "cmlsi_loadings.pkl")
        meta_path = cfg.get_output_dir() / cfg.get("paths.cmlsi_meta", "cmlsi_meta.json")
        joblib.dump(pca.scaler, scaler_path)
        joblib.dump(pca.loadings_, loadings_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        plot_pca_variance(meta, self.cfg.get_figures_dir(), self.cfg.get("visualization.figure_dpi", 150))
        self._master = m
        return m

    def run_all(self, processed_dir: Path | None = None) -> pd.DataFrame:
        """Pipeline: Data -> Features -> PCA"""
        self.run_stage_data(processed_dir)
        self.run_stage_features()
        self.run_stage_pca()
        out_path = self.cfg.get_master_feature_path()
        self.cfg.ensure_dirs()
        self._master.to_parquet(out_path)
        print(f"  [OK] Master table: {out_path}")
        return self._master
