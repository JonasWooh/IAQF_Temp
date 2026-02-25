"""
Visualizations for CMLSI Advanced Robustness Tests
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.dates import DateFormatter

from ..config import get_config
from .structure_tests import run_structure_tests
from .rolling_pca import run_adaptive_cmlsi, compute_rolling_pca_metrics, _get_features_and_data
from .var_comparison import run_stable_vs_adaptive_comparison
from ..models.var_runner import load_master


def plot_structure_tests(results: dict, out_dir: Path) -> None:
    """Plot covariance, eigenvalue, subspace angle, RMT results."""
    cfg = get_config()
    dpi = cfg.get("visualization.figure_dpi", 150)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Frobenius norm: ||Σ_N - Σ_C||_F (main) + cov norms
    ax = axes[0, 0]
    frob = results.get("frobenius", {})
    diff_norm = frob.get("frobenius_norm", 0)
    ax.bar(["||Σ_N - Σ_C||_F"], [diff_norm], color="#3498DB", alpha=0.8, label="Regime shift")
    ax.set_ylabel("Frobenius Norm")
    ax.set_title("Covariance Difference (Large → Regime Shift)")
    ax.tick_params(axis="x", rotation=15)

    # 2. Eigenvalue ratio（展示全部 PC1–PC7）
    ax = axes[0, 1]
    ev = results.get("eigenvalue", {})
    ratios = ev.get("eigenvalue_ratio_crisis_over_normal", [])
    if ratios:
        x = range(1, len(ratios) + 1)
        colors = ["#E74C3C" if r > 1 else "#2ECC71" for r in ratios]
        ax.bar(x, ratios, color=colors, alpha=0.8)
        ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Eigenvalue Ratio (Crisis/Normal)")
    ax.set_title("Eigenvalue Ratio Collapse")

    # 3. Subspace angle
    ax = axes[1, 0]
    sub = results.get("subspace_angle", {})
    angles = sub.get("subspace_angle_deg", [])[:3]
    if angles:
        x = range(1, len(angles) + 1)
        ax.bar(x, angles, color="#9B59B6", alpha=0.8)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("PCA Subspace Angle θ = arccos(|v_N^T v_C|)")

    # 4. RMT: Eigenvalues vs λ_max^noise
    ax = axes[1, 1]
    rmt = results.get("rmt", {})
    if rmt:
        evs = rmt.get("eigenvalues", [])
        lam_max = rmt.get("lambda_max_noise", 1.0)
        x = range(1, len(evs) + 1)
        colors = ["#2ECC71" if ev > lam_max else "#95A5A6" for ev in evs]
        ax.bar(x, evs, color=colors, alpha=0.8, label="Eigenvalue")
        ax.axhline(lam_max, color="#E74C3C", linestyle="--", linewidth=2, label=f"λ_max^noise={lam_max:.3f}")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Eigenvalue λ")
        ax.set_title("RMT: Signal (green) vs Noise (gray)")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "RMT: No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("RMT (Marchenko-Pastur)")

    fig.suptitle("Step 1: Structure Change Tests + RMT", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "01_structure_tests.png", dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out_dir.name}/01_structure_tests.png")


# 特征名 → 图例标签
_FEATURE_LABELS = {
    "spread_mean": "Spread",
    "depth_mean": "Depth",
    "obi_mean": "OBI",
    "_ret_abs": "|ret|",
    "_range_1m": "Range",
    "_rel_spread": "RelSpread",
    "_log_depth": "LogDepth",
}


def plot_rolling_pca(rolling_metrics: pd.DataFrame, out_dir: Path) -> None:
    """Plot rolling PC1 EVR and loadings drift."""
    if rolling_metrics.empty:
        return
    cfg = get_config()
    dpi = cfg.get("visualization.figure_dpi", 150)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Subplot 1: PC1 EVR（蓝色线，避免与 Crisis 红色阴影混淆）
    ax = axes[0]
    if "pc1_evr" in rolling_metrics.columns:
        ax.plot(rolling_metrics.index, rolling_metrics["pc1_evr"], color="#3498DB", linewidth=2, label="PC1 EVR")
    ax.axvspan(pd.Timestamp("2023-03-10"), pd.Timestamp("2023-03-13"), alpha=0.2, color="#E74C3C", label="Crisis (Mar 10–13)")
    ax.set_ylabel("PC1 Explained Variance Ratio")
    ax.set_title("Rolling PCA: PC1 EVR Over Time (24h window)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Subplot 2: 全部 7 个载荷，带清晰图例与颜色
    ax = axes[1]
    load_cols = [c for c in rolling_metrics.columns if c.startswith("load_")]
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C", "#E67E22"]
    if load_cols:
        for i, col in enumerate(load_cols):
            # 从 load_BINANCEUS_BTCUSD_spread_mean 提取 spread_mean
            key = col.replace("load_", "").replace("BINANCEUS_BTCUSD_", "")
            label = _FEATURE_LABELS.get(key, key)
            ax.plot(rolling_metrics.index, rolling_metrics[col], alpha=0.85, linewidth=1.2,
                    color=colors[i % len(colors)], label=label)
    ax.axvspan(pd.Timestamp("2023-03-10"), pd.Timestamp("2023-03-13"), alpha=0.2, color="#E74C3C", label="Crisis (Mar 10–13)")
    ax.set_xlabel("Time")
    ax.set_ylabel("PC1 Loading")
    ax.set_title("Rolling PCA: Loadings Drift (7 features)")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))

    fig.suptitle("Step 2: Dynamic PCA — Loadings & EVR Over Time", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "02_rolling_pca.png", dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out_dir.name}/02_rolling_pca.png")


def plot_stable_vs_adaptive_cmlsi(df: pd.DataFrame, cmlsi_adaptive: pd.Series, out_dir: Path) -> None:
    """Plot Stable CMLSI vs Adaptive CMLSI time series."""
    if "CMLSI" not in df.columns or cmlsi_adaptive.empty:
        return
    cfg = get_config()
    dpi = cfg.get("visualization.figure_dpi", 150)
    fig, ax = plt.subplots(figsize=(14, 5))
    common = df.index.intersection(cmlsi_adaptive.index)
    cmlsi_s = df.loc[common, "CMLSI"]
    cmlsi_a = cmlsi_adaptive.reindex(common).ffill(limit=5)
    ax.plot(common, cmlsi_s, color="#3498DB", linewidth=1, alpha=0.9, label="Stable CMLSI (Normal fit)")
    ax.plot(common, cmlsi_a, color="#E74C3C", linewidth=1, alpha=0.9, label="Adaptive CMLSI (Rolling PCA)")
    ax.axvspan(pd.Timestamp("2023-03-10"), pd.Timestamp("2023-03-13"), alpha=0.2, color="red", label="Crisis")
    ax.set_xlabel("Time")
    ax.set_ylabel("CMLSI")
    ax.set_title("Stable vs Adaptive CMLSI")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
    fig.suptitle("Step 2: Stable vs Adaptive CMLSI", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "03_stable_vs_adaptive_cmlsi.png", dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out_dir.name}/03_stable_vs_adaptive_cmlsi.png")


def plot_var_comparison(comparison: dict, out_dir: Path) -> None:
    """Plot IRF and FEVD: Stable vs Adaptive CMLSI."""
    if not comparison:
        return
    cfg = get_config()
    dpi = cfg.get("visualization.figure_dpi", 150)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    titles = [
        ("Depeg → CMLSI", "CMLSI Response", "depeg_cmlsi"),
        ("Depeg → Basis", "Basis Response (bps)", "depeg_basis"),
        ("Depeg → Premium", "Premium Response (bps)", "depeg_premium"),
    ]
    diff_cols = comparison.get("differenced_cols", set())
    n_steps = len(comparison["irf_stable"]["depeg_premium"])
    x = np.arange(n_steps)
    for col, (title, ylabel, key) in enumerate(titles):
        ax = axes[0, col]
        irf_s = comparison["irf_stable"][key]
        irf_a = comparison["irf_adaptive"][key]
        ax.plot(x, irf_s, color="#3498DB", linewidth=2, label="Stable CMLSI")
        ax.plot(x, irf_a, color="#E74C3C", linewidth=2, label="Adaptive CMLSI")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fevd_titles = [
        ("CMLSI FEVD from Depeg", "cmlsi"),
        ("Basis FEVD from Depeg", "basis"),
        ("Premium FEVD from Depeg", "premium"),
    ]
    for col, (title, k) in enumerate(fevd_titles):
        ax = axes[1, col]
        f_s = comparison["fevd_stable"][k]
        f_a = comparison["fevd_adaptive"][k]
        n_fevd = len(f_s)
        ax.plot(np.arange(1, n_fevd + 1), f_s * 100, color="#3498DB", linewidth=2, label="Stable")
        ax.plot(np.arange(1, n_fevd + 1), f_a * 100, color="#E74C3C", linewidth=2, label="Adaptive")
        ax.set_title(title)
        ax.set_xlabel("Forecast Horizon (min)")
        ax.set_ylabel("Variance Explained (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    note = f" (Δ: {', '.join(sorted(diff_cols))})" if diff_cols else ""
    fig.suptitle(f"Step 3: VAR Comparison — Stable vs Adaptive CMLSI (IRF & FEVD){note}", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "04_var_stable_vs_adaptive.png", dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out_dir.name}/04_var_stable_vs_adaptive.png")


def run_all_visualizations() -> None:
    """Run all CMLSI advanced tests and generate figures."""
    cfg = get_config()
    out_dir = cfg.get_output_dir() / "CMLSI_test_advanced"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1
    results = run_structure_tests(out_dir)
    plot_structure_tests(results, out_dir)

    # Step 2
    features, df = _get_features_and_data()
    cmlsi_adaptive, rolling_metrics = run_adaptive_cmlsi(out_dir)
    plot_rolling_pca(rolling_metrics, out_dir)
    plot_stable_vs_adaptive_cmlsi(df, cmlsi_adaptive, out_dir)

    # Step 3
    comparison = run_stable_vs_adaptive_comparison(out_dir)
    plot_var_comparison(comparison, out_dir)
