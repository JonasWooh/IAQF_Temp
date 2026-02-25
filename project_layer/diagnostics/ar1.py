"""AR(1) 均值回归分析 - 图17"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from ..config import get_config
from ..models.var_runner import load_master

BASIS_COL = "Basis_bps"


def fit_ar1(basis: pd.Series) -> dict:
    y = basis.dropna()
    if len(y) < 10:
        return {"rho": np.nan, "alpha": np.nan, "n": len(y)}
    lag1 = y.shift(1).dropna()
    y = y.iloc[1:]
    lag1 = lag1.iloc[1:]
    common_idx = y.index.intersection(lag1.index)
    y = y.loc[common_idx]
    X = add_constant(lag1.loc[common_idx])
    model = OLS(y, X).fit()
    rho = model.params.iloc[1]
    alpha = 1 - rho
    return {"rho": float(rho), "alpha": float(alpha), "n": int(len(y))}


def run_ar1_mean_reversion() -> None:
    cfg = get_config()
    cfg.ensure_dirs()
    df = load_master()
    if BASIS_COL not in df.columns:
        print("  [WARN] Basis_bps 不存在，跳过 AR1")
        return
    basis = df[BASIS_COL].dropna()
    regimes = [
        ("normal", cfg.get("phases.normal.start"), cfg.get("phases.normal.end"), "Normal (Mar 1–9)", "#2ECC71"),
        ("crisis", cfg.get("phases.stressed.start"), cfg.get("phases.stressed.end"), "Crisis (Mar 10–13)", "#E74C3C"),
        ("recovery", cfg.get("phases.recovery.start"), cfg.get("phases.recovery.end"), "Recovery (Mar 14–21)", "#3498DB"),
    ]
    results = []
    for regime, start, end, label, color in regimes:
        sub = basis.loc[start:end]
        if len(sub) < 10:
            results.append({"regime": regime, "label": label, "rho": np.nan, "alpha": np.nan, "n": len(sub), "color": color})
            continue
        res = fit_ar1(sub)
        res["regime"] = regime
        res["label"] = label
        res["color"] = color
        results.append(res)
    out_txt = cfg.get_output_dir() / "ar1_mean_reversion.txt"
    lines = ["AR(1) Mean Reversion: Basis_t = c + ρ·Basis_{t-1}", "α = 1 - ρ", ""]
    for r in results:
        lines.append(f"  {r['label']}: ρ={r['rho']:.4f}, α={r.get('alpha', 1-r['rho']):.4f}")
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [r["label"] for r in results]
    rhos = [r["rho"] for r in results]
    alphas = [r.get("alpha", 1 - r["rho"]) for r in results]
    colors = [r.get("color", "#3498DB") for r in results]
    x = np.arange(len(labels))
    w = 0.35
    axes[0].bar(x - w/2, rhos, width=w, color=colors, alpha=0.9)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("ρ (AR(1) coefficient)")
    axes[0].set_title("AR(1) Persistence")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(x - w/2, alphas, width=w, color=colors, alpha=0.9)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("α = 1 - ρ")
    axes[1].set_title("Mean Reversion Speed")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("AR(1) on Basis: Persistent Differences After Transaction Costs?", fontsize=12, y=1.02)
    plt.tight_layout()
    out = cfg.get_figures_dir() / "17_ar1_mean_reversion.png"
    fig.savefig(out, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out.name}")
    plt.close()
