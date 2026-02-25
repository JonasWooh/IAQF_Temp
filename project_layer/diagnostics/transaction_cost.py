"""交易成本与无套利区间 - 图18"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..config import get_config
from ..features.utils import make_prefix
from ..models.var_runner import load_master

BASIS_COL = "Basis_bps"
TAKER_FEE_BPS = 10
TC_ROUNDTRIP_BPS = 20
CONSERVATIVE_TC_BPS = 15


def run_transaction_cost_analysis() -> None:
    cfg = get_config()
    cfg.ensure_dirs()
    df = load_master()
    bn = cfg.get("exchanges.btc.exchanges", ["BINANCEUS"])[0]
    close_usd = f"{make_prefix(bn, 'BTC_USD')}_close"
    close_usdc = f"{make_prefix(bn, 'BTC_USDC')}_close"
    spread_usd = f"{make_prefix(bn, 'BTC_USD')}_spread_mean"
    spread_usdc = f"{make_prefix(bn, 'BTC_USDC')}_spread_mean"
    basis_col = BASIS_COL if BASIS_COL in df.columns else "Basis_USD_USDC_BN"
    if basis_col not in df.columns:
        print("  [WARN] Basis 列不存在，跳过交易成本分析")
        return
    for col in [close_usd, close_usdc, spread_usd, spread_usdc]:
        if col not in df.columns:
            print(f"  [WARN] 缺少 {col}，跳过交易成本分析")
            return
    mid = (df[close_usd] + df[close_usdc]) / 2
    mid = mid.replace(0, np.nan)
    if basis_col == "Basis_bps":
        basis_abs_bps = df[basis_col].abs().replace([np.inf, -np.inf], np.nan)
    else:
        basis_abs_bps = (df[basis_col].abs() / mid * 10_000).replace([np.inf, -np.inf], np.nan)
    spread_usd_bps = (df[spread_usd] / df[close_usd].replace(0, np.nan) * 10_000).replace([np.inf, -np.inf], np.nan)
    spread_usdc_bps = (df[spread_usdc] / df[close_usdc].replace(0, np.nan) * 10_000).replace([np.inf, -np.inf], np.nan)
    tc_bps = TC_ROUNDTRIP_BPS + spread_usd_bps + spread_usdc_bps
    regimes = [
        ("normal", cfg.get("phases.normal.start"), cfg.get("phases.normal.end"), "Normal (Mar 1–9)", "#2ECC71"),
        ("crisis", cfg.get("phases.stressed.start"), cfg.get("phases.stressed.end"), "Crisis (Mar 10–13)", "#E74C3C"),
        ("recovery", cfg.get("phases.recovery.start"), cfg.get("phases.recovery.end"), "Recovery (Mar 14–21)", "#3498DB"),
    ]
    summary = []
    for regime, start, end, label, color in regimes:
        mask = (df.index >= start) & (df.index <= end)
        sub_b = basis_abs_bps.loc[mask].dropna()
        sub_tc = tc_bps.loc[mask].dropna()
        common = sub_b.index.intersection(sub_tc.index)
        if len(common) < 10:
            continue
        b_sub = sub_b.loc[common]
        tc_sub = sub_tc.loc[common]
        mean_b = b_sub.mean()
        mean_tc = tc_sub.mean()
        pct_exceed_15 = (b_sub > CONSERVATIVE_TC_BPS).mean() * 100
        summary.append({
            "label": label, "mean_basis_bps": mean_b, "mean_tc_bps": mean_tc,
            "pct_exceed_15": pct_exceed_15, "color": color,
        })
    if not summary:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [s["label"] for s in summary]
    mean_bps = [s["mean_basis_bps"] for s in summary]
    mean_tc = [s["mean_tc_bps"] for s in summary]
    colors = [s["color"] for s in summary]
    x = np.arange(len(labels))
    w = 0.35
    axes[0].bar(x - w/2, mean_bps, width=w, color=colors, alpha=0.9, label="Mean |Basis|")
    axes[0].bar(x + w/2, mean_tc, width=w, color="gray", alpha=0.6, label="Mean TC")
    axes[0].axhline(CONSERVATIVE_TC_BPS, color="black", linestyle="--", linewidth=2, label=f"{CONSERVATIVE_TC_BPS} bps")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("Basis Points")
    axes[0].set_title("Mean |Basis| vs Transaction Cost")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    pct_15 = [s["pct_exceed_15"] for s in summary]
    axes[1].bar(x, pct_15, color=colors, alpha=0.9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("% of Minutes")
    axes[1].set_title(f"% Minutes with |Basis| > {CONSERVATIVE_TC_BPS} bps")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("No-Arbitrage Bounds: Basis vs Transaction Costs", fontsize=12, y=1.02)
    plt.tight_layout()
    out = cfg.get_figures_dir() / "18_transaction_cost_bounds.png"
    fig.savefig(out, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out.name}")
    plt.close()
