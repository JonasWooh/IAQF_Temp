"""
VAR 相关图表生成 (06, 07, 11, 12, 14, 15, 16)
"""
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ..config import get_config
from .var_runner import load_master, get_var_strategy

IDX_DEPEG, IDX_CMLSI, IDX_BASIS, IDX_PREMIUM = 0, 1, 2, 3
VAR_COLS = ["Depeg_bps", "CMLSI", "Basis_bps", "Premium_bps"]


def _irf_level(irf_arr: np.ndarray, col_idx: int, differenced_cols: set) -> np.ndarray:
    """Cumsum IRF for differenced vars → level response."""
    col = VAR_COLS[col_idx] if col_idx < len(VAR_COLS) else None
    if col and col in differenced_cols:
        return np.cumsum(irf_arr)
    return irf_arr


def plot_implied_micro_irf(irf_obj, lower=None, upper=None, differenced_cols=None):
    cfg = get_config()
    scaler_path = cfg.get_output_dir() / cfg.get("paths.cmlsi_scaler", "cmlsi_scaler.pkl")
    meta_path = cfg.get_output_dir() / cfg.get("paths.cmlsi_meta", "cmlsi_meta.json")
    if not scaler_path.exists() or not meta_path.exists():
        return
    scaler = joblib.load(scaler_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    features = meta.get("features", [])
    ld = meta.get("loadings_dict", {})
    f1 = next((f for f in features if "spread_mean" in str(f)), features[0] if features else None)
    f2 = next((f for f in features if f in ("_rel_spread", "_range_1m")), features[1] if len(features) > 1 else features[0])
    idx1 = features.index(f1) if f1 and f1 in features else 0
    idx2 = features.index(f2) if f2 and f2 in features else 1
    load1, load2 = ld.get(f1, 0), ld.get(f2, 0)
    std1 = scaler.scale_[idx1] if idx1 < len(scaler.scale_) else scaler.scale_[0]
    std2 = scaler.scale_[idx2] if idx2 < len(scaler.scale_) else scaler.scale_[1]
    diff_cols = differenced_cols or set()
    cmlsi_irf = _irf_level(irf_obj.orth_irfs[:, IDX_CMLSI, IDX_DEPEG], IDX_CMLSI, diff_cols)
    implied_1 = cmlsi_irf * load1 * std1
    implied_2 = cmlsi_irf * load2 * std2
    has_ci = lower is not None and upper is not None
    if has_ci:
        cmlsi_lo = _irf_level(lower[:, IDX_CMLSI, IDX_DEPEG], IDX_CMLSI, diff_cols)
        cmlsi_hi = _irf_level(upper[:, IDX_CMLSI, IDX_DEPEG], IDX_CMLSI, diff_cols)
        lo1, hi1 = cmlsi_lo * load1 * std1, cmlsi_hi * load1 * std1
        lo2, hi2 = cmlsi_lo * load2 * std2, cmlsi_hi * load2 * std2
    else:
        lo1 = hi1 = lo2 = hi2 = None
    ylabel1 = "Spread (USD)" if "spread" in str(f1) else ("Rel Spread" if "rel" in str(f1) else "Range")
    ylabel2 = "Rel Spread" if "rel" in str(f2) else ("Range (USD)" if "range" in str(f2) else "Spread (USD)")
    x = np.arange(len(implied_1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, series, lo, hi, title, ylabel, color in [
        (axes[0], implied_1, lo1 if has_ci else None, hi1 if has_ci else None,
         f"Implied IRF: {ylabel1}", ylabel1, "#E74C3C"),
        (axes[1], implied_2, lo2 if has_ci else None, hi2 if has_ci else None,
         f"Implied IRF: {ylabel2}", ylabel2, "#3498DB"),
    ]:
        if has_ci and lo is not None:
            ax.fill_between(x, lo, hi, alpha=0.25, color=color, label="95% CI")
        ax.plot(series, color=color, linewidth=2.5, label="Factual")
        ax.axhline(0, color="black", linestyle="--", linewidth=2, label="Counterfactual")
        ax.set_title(title)
        ax.set_xlabel("Minutes after Shock")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    out = cfg.get_figures_dir() / "06_implied_irf_genius_act.png"
    fig.savefig(out, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out.name}")
    plt.close()


def plot_fragmentation_irf(irf_obj, lower=None, upper=None, differenced_cols=None):
    cfg = get_config()
    diff_cols = differenced_cols or set()
    premium_irf = _irf_level(irf_obj.orth_irfs[:, IDX_PREMIUM, IDX_DEPEG], IDX_PREMIUM, diff_cols)
    has_ci = lower is not None and upper is not None
    if has_ci:
        prem_lo = _irf_level(lower[:, IDX_PREMIUM, IDX_DEPEG], IDX_PREMIUM, diff_cols)
        prem_hi = _irf_level(upper[:, IDX_PREMIUM, IDX_DEPEG], IDX_PREMIUM, diff_cols)
    else:
        prem_lo, prem_hi = None, None
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(premium_irf))
    if has_ci:
        ax.fill_between(x, prem_lo, prem_hi, alpha=0.25, color="#9B59B6", label="95% CI")
    ax.plot(premium_irf, color="#9B59B6", linewidth=2.5, label="Factual (2023 SVB Crisis)")
    ax.axhline(0, color="black", linestyle="--", linewidth=2, label="Counterfactual")
    ax.set_title("Market Fragmentation: Premium Response to USDC De-peg")
    ax.set_xlabel("Minutes after Shock")
    ax.set_ylabel("Premium Response (bps)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out = cfg.get_figures_dir() / "07_fragmentation_irf.png"
    fig.savefig(out, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out.name}")
    plt.close()


def plot_fevd(results):
    cfg = get_config()
    periods = cfg.get("var.irf_periods", 60)
    diff_cols = getattr(results, "differenced_cols", set())
    fevd = results.fevd(periods=periods)
    decomp = fevd.decomp
    cmlsi_from_depeg = decomp[IDX_CMLSI, :, IDX_DEPEG]
    basis_from_depeg = decomp[IDX_BASIS, :, IDX_DEPEG]
    premium_from_depeg = decomp[IDX_PREMIUM, :, IDX_DEPEG]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(1, periods + 1)
    axes[0].plot(x, cmlsi_from_depeg * 100, color="#E74C3C", linewidth=2, label="Depeg → CMLSI")
    axes[0].set_title("CMLSI Variance from Depeg Shock (%)")
    axes[0].set_xlabel("Forecast horizon (min)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(x, basis_from_depeg * 100, color="#F39C12", linewidth=2, label="Depeg → Basis")
    axes[1].set_title("Basis Variance from Depeg Shock (%)")
    axes[1].set_xlabel("Forecast horizon (min)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[2].plot(x, premium_from_depeg * 100, color="#9B59B6", linewidth=2, label="Depeg → Premium")
    axes[2].set_title("Premium Variance from Depeg Shock (%)")
    axes[2].set_xlabel("Forecast horizon (min)")
    axes[2].set_ylim(0, 105)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    note = f" (Δ: {', '.join(sorted(diff_cols))})" if diff_cols else ""
    fig.suptitle(f"FEVD (CMLSI-pure){note}", fontsize=13, y=1.02)
    plt.tight_layout()
    out = cfg.get_figures_dir() / "11_fevd_depeg_contribution.png"
    fig.savefig(out, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out.name}")
    plt.close()


def plot_basis_irf(irf_obj, lower=None, upper=None, differenced_cols=None):
    cfg = get_config()
    diff_cols = differenced_cols or set()
    has_ci = lower is not None and upper is not None
    x = np.arange(irf_obj.orth_irfs.shape[0])
    shocks = [(IDX_DEPEG, "Depeg (bps)", "#E74C3C"), (IDX_CMLSI, "CMLSI", "#3498DB"), (IDX_PREMIUM, "Premium (bps)", "#9B59B6")]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for ax, (shock_idx, label, color) in zip(axes, shocks):
        irf = _irf_level(irf_obj.orth_irfs[:, IDX_BASIS, shock_idx], IDX_BASIS, diff_cols)
        ax.plot(x, irf, color=color, linewidth=2, label=f"{label} → Basis")
        if has_ci:
            lo = _irf_level(lower[:, IDX_BASIS, shock_idx], IDX_BASIS, diff_cols)
            hi = _irf_level(upper[:, IDX_BASIS, shock_idx], IDX_BASIS, diff_cols)
            ax.fill_between(x, lo, hi, alpha=0.25, color=color)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax.set_ylabel("Basis Response (bps)")
        ax.set_title(f"Cross-Currency Basis IRF to {label} Shock")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Minutes after Shock")
    fig.suptitle("Cross-Currency Basis: IRF to Depeg, CMLSI, Premium", fontsize=13, y=1.02)
    plt.tight_layout()
    out = cfg.get_figures_dir() / "15_basis_irf.png"
    fig.savefig(out, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out.name}")
    plt.close()


def plot_gas_comparison(comparison: dict):
    """
    有/无 Gas Fee 对比图：Depeg → Premium IRF 与 FEVD。
    comparison 来自 run_var_gas_comparison()。
    """
    cfg = get_config()
    periods = cfg.get("var.irf_periods", 60)
    regime = comparison.get("regime", "crisis")

    irf_b = comparison["depeg_to_premium_irf_baseline"]
    irf_v = comparison["depeg_to_premium_irf_varx"]
    n_irf = min(len(irf_b), len(irf_v), periods)
    irf_b, irf_v = irf_b[:n_irf], irf_v[:n_irf]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(n_irf)

    # IRF: Depeg → Premium
    axes[0].plot(x, irf_b, color="#3498DB", linewidth=2.5, label="VAR (no Gas Fee)")
    axes[0].plot(x, irf_v, color="#E74C3C", linewidth=2.5, label="VARX (with Gas Fee)")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].set_title("Depeg → Premium IRF: Baseline vs VARX")
    axes[0].set_xlabel("Minutes after Shock")
    axes[0].set_ylabel("Premium Response (bps)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # FEVD: Premium variance from Depeg (horizon 1-indexed)
    fevd_b = comparison["fevd_premium_from_depeg_baseline"]
    fevd_v = comparison["fevd_premium_from_depeg_varx"]
    n_fevd = min(len(fevd_b), len(fevd_v), periods)
    fevd_b, fevd_v = fevd_b[:n_fevd], fevd_v[:n_fevd]
    x_fevd = np.arange(1, n_fevd + 1)
    axes[1].plot(x_fevd, fevd_b * 100, color="#3498DB", linewidth=2.5, label="VAR (no Gas Fee)")
    axes[1].plot(x_fevd, fevd_v * 100, color="#E74C3C", linewidth=2.5, label="VARX (with Gas Fee)")
    axes[1].set_title("Premium FEVD from Depeg Shock (%)")
    axes[1].set_xlabel("Forecast horizon (min)")
    axes[1].set_ylabel("Variance explained (%)")
    axes[1].set_ylim(0, 105)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Transaction Cost (Gas Fee exog) — {regime.title()}", fontsize=13, y=1.02)
    plt.tight_layout()
    out = cfg.get_figures_dir() / "17_gas_fee_comparison_baseline_vs_varx.png"
    fig.savefig(out, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out.name}")
    plt.close()


def plot_institutional_comparison(comparison: dict):
    """
    制度摩擦对比图：Baseline(gas) vs Main(gas+weekend) vs Robustness(gas+pause)
    comparison 来自 run_var_institutional_comparison()。
    """
    cfg = get_config()
    periods = cfg.get("var.irf_periods", 60)
    regime = comparison.get("regime", "crisis")

    irf_b = comparison["depeg_to_premium_irf_baseline"]
    irf_m = comparison["depeg_to_premium_irf_main"]
    irf_r = comparison["depeg_to_premium_irf_robustness"]
    n = min(len(irf_b), len(irf_m), len(irf_r), periods)
    irf_b, irf_m, irf_r = irf_b[:n], irf_m[:n], irf_r[:n]
    x = np.arange(n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(x, irf_b, color="#3498DB", linewidth=2, label="gas only")
    axes[0].plot(x, irf_m, color="#2ECC71", linewidth=2, label="gas + weekend")
    axes[0].plot(x, irf_r, color="#9B59B6", linewidth=2, label="gas + pause")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].set_title("Depeg → Premium IRF: Cost vs Institutional Friction")
    axes[0].set_xlabel("Minutes after Shock")
    axes[0].set_ylabel("Premium Response (bps)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    fevd_b = comparison["fevd_premium_baseline"][:n]
    fevd_m = comparison["fevd_premium_main"][:n]
    fevd_r = comparison["fevd_premium_robustness"][:n]
    x_fevd = np.arange(1, len(fevd_b) + 1)
    axes[1].plot(x_fevd, fevd_b * 100, color="#3498DB", linewidth=2, label="gas only")
    axes[1].plot(x_fevd, fevd_m * 100, color="#2ECC71", linewidth=2, label="gas + weekend")
    axes[1].plot(x_fevd, fevd_r * 100, color="#9B59B6", linewidth=2, label="gas + pause")
    axes[1].set_title("Premium FEVD from Depeg Shock (%)")
    axes[1].set_xlabel("Forecast horizon (min)")
    axes[1].set_ylabel("Variance explained (%)")
    axes[1].set_ylim(0, 105)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Institutional Friction (Gas + Weekend/Pause) — {regime.title()}", fontsize=13, y=1.02)
    plt.tight_layout()
    out = cfg.get_figures_dir() / "20_institutional_friction_comparison.png"
    fig.savefig(out, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out.name}")
    plt.close()


def plot_nested_model_comparison(comparison: dict):
    """
    嵌套模型对比图：M0 vs M1 vs M2 vs M3
    Depeg→Basis 与 Depeg→Premium 的 IRF 与 FEVD
    """
    cfg = get_config()
    regime = comparison.get("regime", "crisis")
    colors = {"M0": "#3498DB", "M1": "#2ECC71", "M2": "#9B59B6", "M3": "#E74C3C"}
    labels = {
        "M0": "M0 (baseline)",
        "M1": "M1 (+Credit)",
        "M2": "M2 (+Institution)",
        "M3": "M3 (+Credit+Inst)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex="col")
    # Row 0: IRF Depeg→Basis, IRF Depeg→Premium
    # Row 1: FEVD Basis, FEVD Premium
    for col, (title, key_irf, key_fevd, fevd_title) in enumerate([
        ("Depeg → Basis IRF", "irf_depeg_to_basis", "fevd_basis_from_depeg", "Basis FEVD from Depeg (%)"),
        ("Depeg → Premium IRF", "irf_depeg_to_premium", "fevd_premium_from_depeg", "Premium FEVD from Depeg (%)"),
    ]):
        ax_irf, ax_fevd = axes[0, col], axes[1, col]
        for mkey in ["M0", "M1", "M2", "M3"]:
            m = comparison.get(mkey)
            if m is None:
                continue
            irf = m[key_irf]
            fevd = m[key_fevd]
            x_irf = np.arange(len(irf))
            x_fevd = np.arange(1, len(fevd) + 1)
            ax_irf.plot(x_irf, irf, color=colors.get(mkey, "gray"), linewidth=2, label=labels.get(mkey, mkey))
            ax_fevd.plot(x_fevd, fevd * 100, color=colors.get(mkey, "gray"), linewidth=2, label=labels.get(mkey, mkey))
        ax_irf.axhline(0, color="black", linestyle="--", alpha=0.7)
        ax_irf.set_title(title)
        ax_irf.set_ylabel("Response (bps)")
        ax_irf.legend()
        ax_irf.grid(True, alpha=0.3)
        ax_fevd.set_title(fevd_title)
        ax_fevd.set_xlabel("Forecast horizon (min)")
        ax_fevd.set_ylabel("Variance explained (%)")
        ax_fevd.set_ylim(0, 105)
        ax_fevd.legend()
        ax_fevd.grid(True, alpha=0.3)

    axes[0, 0].set_xlabel("Minutes after Shock")
    axes[0, 1].set_xlabel("Minutes after Shock")
    fig.suptitle(f"Nested Model: M0 vs M1 (+Credit) vs M2 (+Inst) vs M3 — {regime.title()}", fontsize=13, y=1.02)
    plt.tight_layout()
    out = cfg.get_figures_dir() / "21_nested_model_comparison.png"
    fig.savefig(out, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out.name}")
    plt.close()


def plot_regime_comparison(results_crisis):
    cfg = get_config()
    df = load_master()
    strategy_n = get_var_strategy("normal")
    var_normal = strategy_n.prepare_data(df)
    var_normal, diff_n = strategy_n._ensure_stationarity(var_normal)
    if len(var_normal) < 100:
        print("  [WARN] Normal 样本不足，跳过 regime 对比")
        return
    results_normal = strategy_n.fit(var_normal, ensure_stationary=False)
    if not results_normal.is_stable():
        print("  [WARN] Normal VAR 不稳定，跳过 regime 对比")
        return
    diff_c = getattr(results_crisis, "differenced_cols", set())
    periods = cfg.get("var.irf_periods", 60)
    irf_crisis = results_crisis.irf(periods=periods)
    irf_normal = results_normal.irf(periods=periods)
    n_periods = irf_normal.orth_irfs.shape[0]
    x = np.arange(n_periods)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, idx, ylabel in [(axes[0], IDX_CMLSI, "CMLSI Response"), (axes[1], IDX_BASIS, "Basis Response (bps)"), (axes[2], IDX_PREMIUM, "Premium Response (bps)")]:
        irf_n = _irf_level(irf_normal.orth_irfs[:, idx, IDX_DEPEG], idx, diff_n)
        irf_c = _irf_level(irf_crisis.orth_irfs[:, idx, IDX_DEPEG], idx, diff_c)
        ax.plot(x, irf_n, color="#2ECC71", linewidth=2, label="Normal (Mar 1–9)" if idx == IDX_CMLSI else "Normal")
        ax.plot(x, irf_c, color="#E74C3C" if idx == IDX_CMLSI else "#F39C12" if idx == IDX_BASIS else "#9B59B6", linewidth=2, label="Crisis (Mar 10–13)" if idx == IDX_CMLSI else "Crisis")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax.set_title(f"Depeg → {ylabel.split()[0]}: Normal vs Crisis")
        ax.set_xlabel("Minutes after Shock")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Regime Switching (CMLSI-pure)", fontsize=13, y=1.02)
    plt.tight_layout()
    out = cfg.get_figures_dir() / "12_regime_irf_comparison.png"
    fig.savefig(out, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out.name}")
    plt.close()


def plot_triple_regime_irf(results_crisis):
    cfg = get_config()
    periods = cfg.get("var.irf_periods", 60)
    repl = cfg.get("var.irf_ci_repl_triple", 300)
    signif = cfg.get("var.irf_ci_signif", 0.05)
    seed = cfg.get("var.irf_ci_seed", 42)
    regimes = [
        ("normal", "Normal (Mar 1–9)", "#2ECC71"),
        ("crisis", "Crisis (Mar 10–13)", "#E74C3C"),
        ("recovery", "Recovery (Mar 14–21)", "#3498DB"),
    ]
    results_list, irf_list, lower_list, upper_list, diff_list = [], [], [], [], []
    df = load_master()
    for regime, label, _ in regimes:
        strategy = get_var_strategy(regime)
        var_data = strategy.prepare_data(df)
        var_data, diff_cols = strategy._ensure_stationarity(var_data)
        if len(var_data) < 100:
            print(f"  [WARN] {regime} 样本不足，跳过三期 IRF")
            return
        res = strategy.fit(var_data, ensure_stationary=False)
        if not res.is_stable():
            print(f"  [WARN] {regime} VAR 不稳定，跳过")
            return
        irf_obj = res.irf(periods=periods)
        lower, upper = res.irf_errband_mc(orth=True, repl=repl, steps=periods, signif=signif, seed=seed)
        results_list.append(res)
        irf_list.append(irf_obj)
        lower_list.append(lower)
        upper_list.append(upper)
        diff_list.append(diff_cols)
    n_steps = irf_list[0].orth_irfs.shape[0]
    x = np.arange(n_steps)
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (regime, label, color) in enumerate(regimes):
        irf_basis = _irf_level(irf_list[i].orth_irfs[:, IDX_BASIS, IDX_DEPEG], IDX_BASIS, diff_list[i])
        lo = _irf_level(lower_list[i][:, IDX_BASIS, IDX_DEPEG], IDX_BASIS, diff_list[i])
        hi = _irf_level(upper_list[i][:, IDX_BASIS, IDX_DEPEG], IDX_BASIS, diff_list[i])
        ax.plot(x, irf_basis, color=color, linewidth=2.5, label=label)
        ax.fill_between(x, lo, hi, alpha=0.2, color=color)
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Minutes after Depeg Shock")
    ax.set_ylabel("Basis Response (bps)")
    ax.set_title("Triple-Regime IRF: Depeg → Basis")
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = cfg.get_figures_dir() / "14_triple_regime_irf_depeg_to_basis.png"
    fig.savefig(out, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out.name}")
    plt.close()
    fig2, axes = plt.subplots(1, 3, figsize=(16, 6))
    titles = [
        ("Depeg → CMLSI", "CMLSI Response", IDX_CMLSI),
        ("Depeg → Basis", "Basis Response (bps)", IDX_BASIS),
        ("Depeg → Premium", "Premium Response (bps)", IDX_PREMIUM),
    ]
    for ax, (title, ylabel, idx) in zip(axes, titles):
        for i, (regime, label, color) in enumerate(regimes):
            irf_vals = _irf_level(irf_list[i].orth_irfs[:, idx, IDX_DEPEG], idx, diff_list[i])
            lo = _irf_level(lower_list[i][:, idx, IDX_DEPEG], idx, diff_list[i])
            hi = _irf_level(upper_list[i][:, idx, IDX_DEPEG], idx, diff_list[i])
            ax.plot(x, irf_vals, color=color, linewidth=2, label=label)
            ax.fill_between(x, lo, hi, alpha=0.15, color=color)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel("Minutes after Shock")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig2.suptitle("Triple-Regime IRF: Normal | Crisis | Recovery", fontsize=13, y=1.02)
    plt.tight_layout()
    out2 = cfg.get_figures_dir() / "16_triple_regime_irf_full.png"
    fig2.savefig(out2, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out2.name}")
    plt.close()
