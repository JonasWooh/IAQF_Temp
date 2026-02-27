"""
EDA 探索性数据分析 - 图01-05, 09, 10
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from pathlib import Path

from ..config import get_config
from ..features.utils import make_prefix
from ..models.var_runner import load_master
# edit: added fragmentation analysis imports
from .liquidity_fragmentation import (
    analyze_dimension_1_static_differences,
    analyze_dimension_2_dynamic_fragmentation,
    analyze_dimension_3_correlation_breakdown,
)


def _try_load_master() -> pd.DataFrame | None:
    try:
        return load_master()
    except FileNotFoundError:
        return None


def _load_1min(processed_dir: Path, exchange: str, pair: str) -> pd.DataFrame | None:
    """pair 如 USDC_USD, BTC_USD -> 文件 BINANCEUS_USDC_USD.parquet"""
    fp = processed_dir / f"{exchange}_{pair}.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    df["time_exchange"] = pd.to_datetime(df["time_exchange"])
    df = df.set_index("time_exchange")
    return df


def filter_date_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[start:end].copy()


def _feature_short_name(feat: str) -> str:
    m = {"spread_mean": "Spread", "depth_mean": "Depth", "obi_mean": "OBI",
         "_ret_abs": "|ret|", "_range_1m": "Range", "_rel_spread": "RelSpread",
         "_log_depth": "LogDepth"}
    for k, v in m.items():
        if k in feat or feat == k:
            return v
    return feat.split("_")[-1] if "_" in feat else feat


def plot_01_depeg_timeline():
    cfg = get_config()
    df = _try_load_master()
    usdc_ex = cfg.get("exchanges.usdc_usd.exchange", "BINANCEUS")
    usdc_pair = cfg.get("exchanges.usdc_usd.pair", "USDC_USD")
    col = f"{make_prefix(usdc_ex, usdc_pair)}_close"
    if df is None or col not in df.columns:
        processed = cfg.get_processed_1min_dir()
        df = _load_1min(processed, usdc_ex, usdc_pair)
        col = "close" if df is not None else None
        df = df[["close"]] if df is not None and "close" in df.columns else None
    else:
        df = df[[col]].rename(columns={col: "close"})
    if df is None or df.empty:
        print("  [SKIP] 01: No USDC/USD data")
        return
    df = filter_date_range(df, cfg.get("dates.start"), cfg.get("dates.eda_usdc_end"))
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["close"], color="#2E86AB", linewidth=0.8, alpha=0.9)
    ax.axvspan(
        pd.Timestamp(cfg.get("phases.stressed.start")),
        pd.Timestamp(cfg.get("phases.stressed.end")),
        alpha=0.2, color="red", label="De-peg period (Mar 10–13)",
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("USDC/USD Price")
    ax.set_title("Chart 1: USDC De-peg Timeline (1-min Close)")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    out = cfg.get_figures_dir() / "01_depeg_timeline.png"
    fig.savefig(out, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out.name}")
    plt.close()


def plot_02_liquidity_evaporation():
    cfg = get_config()
    df = _try_load_master()
    bn = cfg.get("exchanges.btc.exchanges", ["BINANCEUS"])[0]
    prefix = make_prefix(bn, "BTC_USD")
    if df is not None and f"{prefix}_spread_mean" in df.columns:
        df = pd.DataFrame({
            "spread": df[f"{prefix}_spread_mean"],
            "depth": df[f"{prefix}_depth_last"],
        })
    else:
        processed = cfg.get_processed_1min_dir()
        df = _load_1min(processed, bn, "BTC_USD")
        if df is None:
            cb = cfg.get("exchanges.btc.exchanges", ["BINANCEUS", "COINBASE"])[1]
            df = _load_1min(processed, cb, "BTC_USD")
        if df is None or df.empty:
            print("  [SKIP] 02: No BTC/USD data")
            return
        df = pd.DataFrame({"spread": df["spread_mean"], "depth": df["depth_last"]})
    df = filter_date_range(df, cfg.get("dates.start"), cfg.get("dates.end"))
    if df.empty:
        return
    w = cfg.get("thresholds.spread_ma_window", 60)
    alpha = cfg.get("visualization.raw_data_alpha", 0.3)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(df.index, df["spread"], color="gray", alpha=alpha, linewidth=0.8, label="1-Min Spread")
    ax1.plot(df.index, df["spread"].rolling(w, min_periods=1).mean(), color="#E94F37", linewidth=2, label="60-Min MA")
    ax1.axvspan(pd.Timestamp(cfg.get("phases.stressed.start")), pd.Timestamp(cfg.get("phases.stressed.end")),
                alpha=0.15, color="red")
    ax1.set_ylabel("Spread ($)")
    ax1.set_title("Chart 2a: BTC/USD Bid-Ask Spread")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax2.plot(df.index, df["depth"], color="gray", alpha=alpha, linewidth=0.8, label="1-Min Depth")
    ax2.plot(df.index, df["depth"].rolling(w, min_periods=1).mean(), color="#44AF69", linewidth=2, label="60-Min MA")
    ax2.axvspan(pd.Timestamp(cfg.get("phases.stressed.start")), pd.Timestamp(cfg.get("phases.stressed.end")),
                alpha=0.15, color="red")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Top-of-book Depth")
    ax2.set_title("Chart 2b: BTC/USD Order Book Depth")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    out = cfg.get_figures_dir() / "02_liquidity_evaporation.png"
    fig.savefig(out, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out.name}")
    plt.close()


def plot_03_cross_exchange_spread():
    cfg = get_config()
    df = _try_load_master()
    if df is None or "Premium_BTC" not in df.columns:
        processed = cfg.get_processed_1min_dir()
        exs = cfg.get("exchanges.btc.exchanges", ["BINANCEUS", "COINBASE"])
        df_b = _load_1min(processed, exs[0], "BTC_USD")
        df_c = _load_1min(processed, exs[1], "BTC_USD")
        if df_b is None or df_c is None or df_b.empty or df_c.empty:
            print("  [SKIP] 03: Missing exchange data")
            return
        b = df_b[["close"]].rename(columns={"close": "binanceus"})
        c = df_c[["close"]].rename(columns={"close": "coinbase"})
        merged = b.join(c, how="inner")
        merged["spread"] = merged["binanceus"] - merged["coinbase"]
    else:
        merged = df[["Premium_BTC"]].rename(columns={"Premium_BTC": "spread"})
    merged = filter_date_range(merged, cfg.get("dates.start"), cfg.get("dates.end"))
    if merged.empty:
        return
    col = "spread" if "spread" in merged.columns else merged.columns[0]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(merged.index, merged[col], color="#6C5CE7", linewidth=0.7, alpha=0.9)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.axvspan(pd.Timestamp(cfg.get("phases.stressed.start")), pd.Timestamp(cfg.get("phases.stressed.end")),
                alpha=0.2, color="red", label="De-peg period")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price Spread ($)")
    ax.set_title("Chart 3: Cross-Exchange Spread (BinanceUS − Coinbase BTC/USD)")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=30)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = cfg.get_figures_dir() / "03_cross_exchange_spread.png"
    fig.savefig(out, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out.name}")
    plt.close()


def plot_04_cmlsi_loadings():
    cfg = get_config()
    meta_path = cfg.get_output_dir() / cfg.get("paths.cmlsi_meta", "cmlsi_meta.json")
    if not meta_path.exists():
        print("  [SKIP] 04: CMLSI meta not found")
        return
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    loadings_dict = meta.get("loadings_dict", {})
    features = meta.get("features", [])
    if not features:
        return
    labels = []
    values = []
    flabel = {"spread_mean": "Spread", "depth_mean": "Depth", "obi_mean": "OBI",
              "_ret_abs": "|ret|", "_range_1m": "Range", "_rel_spread": "RelSpread", "_log_depth": "LogDepth"}
    for feat in features:
        short = _feature_short_name(feat)
        labels.append(short)
        values.append(loadings_dict.get(feat, 0))
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E74C3C" if v > 0 else "#3498DB" for v in values]
    bars = ax.barh(labels, values, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    for bar in bars:
        w = bar.get_width()
        xpos = w + 0.02 if w > 0 else w - 0.08
        ax.text(xpos, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=11, fontweight="bold")
    evr = meta.get("cmlsi_explained_variance_ratio", 0)
    ev = (float(evr[0]) if isinstance(evr, list) else float(evr)) * 100
    ax.set_title(f"CMLSI Factor Loadings (Top 3 PCs)\nVariance: {ev:.1f}%")
    ax.set_xlabel("Loading Weight")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = cfg.get_figures_dir() / "04_cmlsi_loadings.png"
    fig.savefig(out, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out.name}")
    plt.close()


def plot_05_cmlsi_decomposition():
    cfg = get_config()
    meta_path = cfg.get_output_dir() / cfg.get("paths.cmlsi_meta", "cmlsi_meta.json")
    scaler_path = cfg.get_output_dir() / cfg.get("paths.cmlsi_scaler", "cmlsi_scaler.pkl")
    loadings_path = cfg.get_output_dir() / cfg.get("paths.cmlsi_loadings", "cmlsi_loadings.pkl")
    if not meta_path.exists() or not scaler_path.exists() or not loadings_path.exists():
        print("  [SKIP] 05: CMLSI artifacts not found")
        return
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    features = meta.get("features", [])
    if not features:
        return
    df = _try_load_master()
    if df is None or "CMLSI" not in df.columns:
        return
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  [SKIP] 05: Missing features: {missing[:3]}")
        return
    scaler = joblib.load(scaler_path)
    loadings = np.asarray(joblib.load(loadings_path))
    crisis_df = filter_date_range(df, cfg.get("phases.stressed.start"), cfg.get("phases.stressed.end"))
    raw_data = crisis_df[features].dropna(how="any")
    if raw_data.empty:
        return
    scaled_data = scaler.transform(raw_data)
    contributions = scaled_data * loadings
    labels = [_feature_short_name(f) for f in features]
    contrib_df = pd.DataFrame(contributions, index=raw_data.index, columns=labels)
    contrib_df_smooth = contrib_df.rolling(window=15).mean().dropna()
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, col in enumerate(contrib_df_smooth.columns):
        ax.plot(contrib_df_smooth.index, contrib_df_smooth[col], label=col, color=colors[i], alpha=0.85, linewidth=1.2)
    cmlsi_smooth = crisis_df.loc[contrib_df_smooth.index, "CMLSI"].rolling(window=15).mean()
    ax.plot(cmlsi_smooth.index, cmlsi_smooth, label="CMLSI", color="black", linewidth=2.5, linestyle="--")
    ax.set_title("Dynamic Decomposition of CMLSI during De-peg Crisis (15-Min MA)")
    ax.set_ylabel("Standardized Contribution")
    ax.set_xlabel("Time")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M"))
    plt.xticks(rotation=25)
    plt.tight_layout()
    out = cfg.get_figures_dir() / "05_cmlsi_decomposition.png"
    fig.savefig(out, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out.name}")
    plt.close()


def plot_09_cross_currency_basis():
    cfg = get_config()
    df = _try_load_master()
    basis_cols = ["Basis_USD_USDC_BN", "Basis_USD_USDT_BN"]
    avail = [c for c in basis_cols if df is not None and c in df.columns]
    if not avail:
        print("  [SKIP] 09: No cross-currency basis columns")
        return
    plot_df = filter_date_range(df[avail], cfg.get("dates.start"), cfg.get("dates.eda_usdc_end"))
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    labels_map = {"Basis_USD_USDC_BN": "BTC/USD − BTC/USDC", "Basis_USD_USDT_BN": "BTC/USD − BTC/USDT"}
    colors = ["#2E86AB", "#E94F37"]
    for i, col in enumerate(avail):
        ax.plot(plot_df.index, plot_df[col], label=labels_map.get(col, col), color=colors[i % 2], linewidth=0.8, alpha=0.9)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvspan(pd.Timestamp(cfg.get("phases.stressed.start")), pd.Timestamp(cfg.get("phases.stressed.end")),
                alpha=0.2, color="red", label="De-peg period")
    ax.set_xlabel("Date")
    ax.set_ylabel("Basis (USD)")
    ax.set_title("Chart 9: Cross-Currency Basis (BTC/USD vs Stablecoins)")
    ax.legend(loc="best", fontsize=9)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=30)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = cfg.get_figures_dir() / "09_cross_currency_basis.png"
    fig.savefig(out, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out.name}")
    plt.close()


def plot_10_stablecoin_premium():
    cfg = get_config()
    df = _try_load_master()
    basis_cols = ["Basis_USDC_USDT_BN", "Basis_USDC_USDT_CB"]
    avail = [c for c in basis_cols if df is not None and c in df.columns]
    if not avail:
        print("  [SKIP] 10: No USDC vs USDT basis")
        return
    plot_df = filter_date_range(df[avail], cfg.get("dates.start"), cfg.get("dates.eda_usdc_end"))
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    labels_map = {"Basis_USDC_USDT_BN": "BinanceUS: BTC/USDC − BTC/USDT", "Basis_USDC_USDT_CB": "Coinbase: BTC/USDC − BTC/USDT"}
    colors = ["#6C5CE7", "#00B894"]
    for i, col in enumerate(avail):
        ax.plot(plot_df.index, plot_df[col], label=labels_map.get(col, col), color=colors[i % 2], linewidth=0.8, alpha=0.9)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvspan(pd.Timestamp(cfg.get("phases.stressed.start")), pd.Timestamp(cfg.get("phases.stressed.end")),
                alpha=0.2, color="red", label="De-peg period")
    ax.set_xlabel("Date")
    ax.set_ylabel("Basis (USD)")
    ax.set_title("Chart 10: Stablecoin Dynamics — USDC vs USDT Basis")
    ax.legend(loc="best", fontsize=9)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=30)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = cfg.get_figures_dir() / "10_stablecoin_premium.png"
    fig.savefig(out, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out.name}")
    plt.close()


def save_gas_fee_eda_report(out_path: Path | None = None) -> dict:
    """生成 Gas Fee EDA 数据报告，保存为 txt，返回汇总 dict"""
    cfg = get_config()
    try:
        from ..data.gas_preprocessing import load_gas_fee, adf_test_gas_level_for_regime
    except ImportError:
        print("  [SKIP] gas_fee_eda_report: gas_preprocessing not found")
        return {}
    try:
        gas_df = load_gas_fee()
    except FileNotFoundError:
        print("  [SKIP] gas_fee_eda_report: Gas fee CSV not found")
        return {}
    start = cfg.get("dates.start")
    end = cfg.get("dates.end")
    gas = gas_df["base_fee_gwei"].loc[start:end].dropna()
    if gas.empty:
        return {}
    pn, pe_n = cfg.get("phases.normal.start"), cfg.get("phases.normal.end")
    ps, pe_s = cfg.get("phases.stressed.start"), cfg.get("phases.stressed.end")
    pr, pe_r = cfg.get("phases.recovery.start"), cfg.get("phases.recovery.end")
    g_n = gas.loc[pn:pe_n]
    g_s = gas.loc[ps:pe_s]
    g_r = gas.loc[pr:pe_r]

    def _stats(s: pd.Series, name: str) -> dict:
        if s.empty:
            return {}
        return {
            "regime": name,
            "n": len(s),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "max": s.max(),
            "p25": np.percentile(s, 25),
            "p50": np.percentile(s, 50),
            "p75": np.percentile(s, 75),
            "p95": np.percentile(s, 95),
        }

    stats_n = _stats(g_n, "Normal (Mar 1–9)")
    stats_s = _stats(g_s, "Crisis (Mar 10–13)")
    stats_r = _stats(g_r, "Recovery (Mar 14–21)")
    try:
        from ..models.var_runner import load_master
        df = load_master()
    except Exception:
        df = None
    midx_n = df.loc[pn:pe_n].index if df is not None and len(df) > 0 else g_n.index
    midx_s = df.loc[ps:pe_s].index if df is not None and len(df) > 0 else g_s.index
    midx_r = df.loc[pr:pe_r].index if df is not None and len(df) > 0 else g_r.index
    adf_n = adf_test_gas_level_for_regime(gas_df, midx_n) if len(midx_n) > 20 else {}
    adf_s = adf_test_gas_level_for_regime(gas_df, midx_s) if len(midx_s) > 20 else {}
    adf_r = adf_test_gas_level_for_regime(gas_df, midx_r) if len(midx_r) > 20 else {}

    lines = [
        "=" * 70,
        "Gas Fee (base_fee_gwei) EDA — 2023 SVB Period",
        "=" * 70,
        "",
        "[Full Sample] " + start + " to " + end,
        f"  n = {len(gas):,}  |  mean = {gas.mean():.2f}  |  std = {gas.std():.2f}  |  min = {gas.min():.2f}  |  max = {gas.max():.2f}",
        "",
        "[Normal] " + pn + " to " + pe_n,
        f"  n = {stats_n.get('n', 0):,}  |  mean = {stats_n.get('mean', 0):.2f}  |  median = {stats_n.get('p50', 0):.2f}  |  std = {stats_n.get('std', 0):.2f}",
        f"  min = {stats_n.get('min', 0):.2f}  |  max = {stats_n.get('max', 0):.2f}  |  p95 = {stats_n.get('p95', 0):.2f} Gwei",
        f"  ADF: p = {adf_n.get('pvalue', 1.0):.4f}" + ("  stationary" if adf_n.get("is_stationary") else "  non-stationary"),
        "",
        "[Crisis] " + ps + " to " + pe_s,
        f"  n = {stats_s.get('n', 0):,}  |  mean = {stats_s.get('mean', 0):.2f}  |  median = {stats_s.get('p50', 0):.2f}  |  std = {stats_s.get('std', 0):.2f}",
        f"  min = {stats_s.get('min', 0):.2f}  |  max = {stats_s.get('max', 0):.2f}  |  p95 = {stats_s.get('p95', 0):.2f} Gwei",
        f"  ADF: p = {adf_s.get('pvalue', 1.0):.4f}" + ("  stationary" if adf_s.get("is_stationary") else "  non-stationary"),
        "",
        "[Recovery] " + pr + " to " + pe_r,
        f"  n = {stats_r.get('n', 0):,}  |  mean = {stats_r.get('mean', 0):.2f}  |  median = {stats_r.get('p50', 0):.2f}  |  std = {stats_r.get('std', 0):.2f}",
        f"  min = {stats_r.get('min', 0):.2f}  |  max = {stats_r.get('max', 0):.2f}  |  p95 = {stats_r.get('p95', 0):.2f} Gwei",
        f"  ADF: p = {adf_r.get('pvalue', 1.0):.4f}" + ("  stationary" if adf_r.get("is_stationary") else "  non-stationary"),
        "",
    ]
    path = out_path or cfg.get_output_dir() / "gas_fee_eda_report.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] Gas Fee EDA report: {path}")

    out = {
        "full": {"n": len(gas), "mean": gas.mean(), "std": gas.std(), "min": gas.min(), "max": gas.max()},
        "normal": stats_n,
        "crisis": stats_s,
        "recovery": stats_r,
        "adf": {"normal": adf_n, "crisis": adf_s, "recovery": adf_r},
    }
    return out


def save_depeg_eda_report(out_path: Path | None = None) -> dict:
    """生成 Depeg (脱锚) EDA 数据报告，保存为 txt，返回汇总 dict"""
    cfg = get_config()
    try:
        df = load_master()
    except FileNotFoundError:
        print("  [SKIP] depeg_eda_report: Master not found")
        return {}
    if df is None or "Depeg_bps" not in df.columns:
        print("  [SKIP] depeg_eda_report: Depeg_bps not in Master")
        return {}
    from statsmodels.tsa.stattools import adfuller
    d = df["Depeg_bps"].dropna()
    pn, pe_n = cfg.get("phases.normal.start"), cfg.get("phases.normal.end")
    ps, pe_s = cfg.get("phases.stressed.start"), cfg.get("phases.stressed.end")
    pr, pe_r = cfg.get("phases.recovery.start"), cfg.get("phases.recovery.end")
    g_n = d.loc[pn:pe_n]
    g_s = d.loc[ps:pe_s]
    g_r = d.loc[pr:pe_r]

    def _stats(s: pd.Series) -> dict:
        if s.empty:
            return {}
        return {
            "n": len(s),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "min": s.min(),
            "max": s.max(),
            "p95": np.percentile(s, 95),
        }

    stats_n, stats_s, stats_r = _stats(g_n), _stats(g_s), _stats(g_r)
    adf_n = adf_s = adf_r = {}
    if len(g_n) > 50:
        _, pval, *_ = adfuller(g_n, autolag="AIC", regression="c")
        adf_n = {"pvalue": pval, "is_stationary": pval < 0.05}
    if len(g_s) > 50:
        _, pval, *_ = adfuller(g_s, autolag="AIC", regression="c")
        adf_s = {"pvalue": pval, "is_stationary": pval < 0.05}
    if len(g_r) > 50:
        _, pval, *_ = adfuller(g_r, autolag="AIC", regression="c")
        adf_r = {"pvalue": pval, "is_stationary": pval < 0.05}

    lines = [
        "=" * 70,
        "Depeg (Depeg_bps) EDA — 2023 SVB Period",
        "USDC De-pegging Intensity: (1 - USDC/USD) × 10,000 bps",
        "=" * 70,
        "",
        "[Full Sample] " + pn + " to " + pe_r,
        f"  n = {len(d):,}  |  mean = {d.mean():.2f} bps  |  std = {d.std():.2f}  |  min = {d.min():.2f}  |  max = {d.max():.2f} bps",
        "",
        "[Normal (Mar 1–9)] " + pn + " to " + pe_n,
        f"  n = {stats_n.get('n', 0):,}  |  mean = {stats_n.get('mean', 0):.2f} bps  |  median = {stats_n.get('median', 0):.2f}  |  std = {stats_n.get('std', 0):.2f}",
        f"  min = {stats_n.get('min', 0):.2f}  |  max = {stats_n.get('max', 0):.2f}  |  p95 = {stats_n.get('p95', 0):.2f} bps",
        f"  ADF: p = {adf_n.get('pvalue', 1.0):.4f}" + ("  stationary" if adf_n.get("is_stationary") else "  non-stationary"),
        "",
        "[Crisis (Mar 10–13)] " + ps + " to " + pe_s,
        f"  n = {stats_s.get('n', 0):,}  |  mean = {stats_s.get('mean', 0):.2f} bps  |  median = {stats_s.get('median', 0):.2f}  |  std = {stats_s.get('std', 0):.2f}",
        f"  min = {stats_s.get('min', 0):.2f}  |  max = {stats_s.get('max', 0):.2f}  |  p95 = {stats_s.get('p95', 0):.2f} bps",
        f"  ADF: p = {adf_s.get('pvalue', 1.0):.4f}" + ("  stationary" if adf_s.get("is_stationary") else "  non-stationary  →  (ΔDepeg_bps)"),
        "",
        "[Recovery (Mar 14–21)] " + pr + " to " + pe_r,
        f"  n = {stats_r.get('n', 0):,}  |  mean = {stats_r.get('mean', 0):.2f} bps  |  median = {stats_r.get('median', 0):.2f}  |  std = {stats_r.get('std', 0):.2f}",
        f"  min = {stats_r.get('min', 0):.2f}  |  max = {stats_r.get('max', 0):.2f}  |  p95 = {stats_r.get('p95', 0):.2f} bps",
        f"  ADF: p = {adf_r.get('pvalue', 1.0):.4f}" + ("  stationary" if adf_r.get("is_stationary") else "  non-stationary"),
        "",
        "=" * 70,
        "解读：Normal 期脱锚极轻；Crisis 期脱锚显著放大；Recovery 期回落但仍高于 Normal。",
        "=" * 70,
    ]
    path = out_path or cfg.get_output_dir() / "depeg_eda_report.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] Depeg EDA report: {path}")
    return {"full": {"n": len(d), "mean": d.mean(), "std": d.std()}, "normal": stats_n, "crisis": stats_s, "recovery": stats_r}


def plot_19_gas_fee_eda():
    """Gas Fee EDA: 时间序列、分 regime 分布、滚动统计"""
    cfg = get_config()
    try:
        from ..data.gas_preprocessing import load_gas_fee
    except ImportError:
        print("  [SKIP] 19: gas_preprocessing not found")
        return
    try:
        gas_df = load_gas_fee()
    except FileNotFoundError:
        print("  [SKIP] 19: Gas fee CSV not found")
        return
    start = cfg.get("dates.start")
    end = cfg.get("dates.end")
    gas = gas_df["base_fee_gwei"].loc[start:end]
    if gas.empty:
        print("  [SKIP] 19: No gas data in date range")
        return
    pn, ps, pr = cfg.get("phases.normal.start"), cfg.get("phases.stressed.start"), cfg.get("phases.recovery.start")
    pe_n, pe_s, pe_r = cfg.get("phases.normal.end"), cfg.get("phases.stressed.end"), cfg.get("phases.recovery.end")
    g_n = gas.loc[pn:pe_n].dropna()
    g_s = gas.loc[ps:pe_s].dropna()
    g_r = gas.loc[pr:pe_r].dropna()
    w = cfg.get("thresholds.spread_ma_window", 60)
    alpha = cfg.get("visualization.raw_data_alpha", 0.3)

    fig = plt.figure(figsize=(14, 10))
    # 1. 时间序列 + 危机期
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(gas.index, gas.values, color="gray", alpha=alpha, linewidth=0.6, label="1-min base_fee_gwei")
    ax1.plot(gas.index, gas.rolling(w, min_periods=1).mean(), color="#F39C12", linewidth=2, label=f"{w}-min MA")
    ax1.axvspan(pd.Timestamp(ps), pd.Timestamp(pe_s), alpha=0.2, color="red", label="Crisis (Mar 10–13)")
    ax1.set_ylabel("base_fee_gwei")
    ax1.set_title("Gas Fee EDA: Ethereum Base Fee (Gwei) — 2023 SVB Period")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30)

    # 2. 分 regime 分布
    ax2 = fig.add_subplot(3, 1, 2)
    bins = np.linspace(gas.min(), min(gas.max(), 200), 60)
    ax2.hist(g_n, bins=bins, alpha=0.6, color="#2ECC71", label=f"Normal (n={len(g_n):,})", density=True, edgecolor="none")
    ax2.hist(g_s, bins=bins, alpha=0.6, color="#E74C3C", label=f"Crisis (n={len(g_s):,})", density=True, edgecolor="none")
    ax2.hist(g_r, bins=bins, alpha=0.6, color="#3498DB", label=f"Recovery (n={len(g_r):,})", density=True, edgecolor="none")
    p95_n = np.percentile(g_n, 95) if len(g_n) >= 10 else gas.median()
    ax2.axvline(p95_n, color="#27AE60", linestyle="--", linewidth=1.5, label=f"Normal 95th = {p95_n:.1f} Gwei")
    ax2.set_xlabel("base_fee_gwei (Gwei)")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution by Regime (Normal | Crisis | Recovery)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. 按 regime 的箱线图 + 汇总
    ax3 = fig.add_subplot(3, 1, 3)
    data_reg = [g_n, g_s, g_r]
    bp = ax3.boxplot(
        data_reg,
        labels=["Normal\n(Mar 1–9)", "Crisis\n(Mar 10–13)", "Recovery\n(Mar 14–21)"],
        patch_artist=True,
        showfliers=False,
    )
    colors = ["#2ECC71", "#E74C3C", "#3498DB"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax3.set_ylabel("base_fee_gwei (Gwei)")
    ax3.set_title("Gas Fee by Regime (Box Plot)")
    ax3.grid(True, alpha=0.3, axis="y")
    stats_text = (
        f"Normal: mean={g_n.mean():.1f} median={g_n.median():.1f} 95p={p95_n:.1f}  |  "
        f"Crisis: mean={g_s.mean():.1f} median={g_s.median():.1f}  |  "
        f"Recovery: mean={g_r.mean():.1f} median={g_r.median():.1f}"
    )
    ax3.set_xlabel(stats_text, fontsize=9)

    plt.tight_layout()
    out = cfg.get_figures_dir() / "19_gas_fee_eda.png"
    fig.savefig(out, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out.name}")
    plt.close()


def run_eda():
    cfg = get_config()
    cfg.ensure_dirs()
    print("\n--- EDA (Figs 01-05, 09, 10, 19 + Fragmentation) ---")
    if not cfg.get_processed_1min_dir().exists() and not cfg.get_master_feature_path().exists():
        print("  [WARN] No processed data or master, skipping EDA")
        return
    plot_01_depeg_timeline()
    plot_02_liquidity_evaporation()
    plot_03_cross_exchange_spread()
    plot_09_cross_currency_basis()
    plot_10_stablecoin_premium()
    if cfg.get_master_feature_path().exists():
        plot_04_cmlsi_loadings()
        plot_05_cmlsi_decomposition()
        # edit: integrated Dim 1-3 liquidity fragmentation analysis into run_eda
        print("\n--- Liquidity Fragmentation (Dim 1-3) ---")
        analyze_dimension_1_static_differences()
        analyze_dimension_2_dynamic_fragmentation()
        analyze_dimension_3_correlation_breakdown()
    plot_19_gas_fee_eda()
    save_gas_fee_eda_report()
    save_depeg_eda_report()
