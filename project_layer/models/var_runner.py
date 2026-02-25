"""
VAR Runner - 使用 Strategy 执行 VAR 流程
支持 VARX（Gas Fee 外生变量）及 Baseline vs VARX 对比实验
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from ..config import get_config
from .var_strategy import CrisisVARStrategy, NormalVARStrategy, RecoveryVARStrategy
from ..features.pipeline import FeaturePipeline


def load_master() -> pd.DataFrame:
    cfg = get_config()
    path = cfg.get_master_feature_path()
    if not path.exists():
        raise FileNotFoundError(f"Master not found: {path}. Run FeaturePipeline first.")
    df = pd.read_parquet(path)
    if "time_exchange" not in df.columns and df.index.name in (None, "time_exchange"):
        df.index = pd.to_datetime(df.index)
    elif "time_exchange" in df.columns:
        df = df.set_index("time_exchange")
    df.index = pd.to_datetime(df.index)
    return df


def get_var_strategy(regime: str):
    """根据 regime 字符串返回策略实例"""
    strategies = {
        "normal": NormalVARStrategy(),
        "crisis": CrisisVARStrategy(),
        "recovery": RecoveryVARStrategy(),
    }
    if regime not in strategies:
        raise ValueError(f"regime must be one of {list(strategies.keys())}")
    return strategies[regime]


def run_var_pipeline(regime: str = "crisis") -> tuple:
    """运行 VAR 流程，返回 (results, irf_obj)"""
    cfg = get_config()
    cfg.ensure_dirs()
    df = load_master()
    strategy = get_var_strategy(regime)
    var_data = strategy.prepare_data(df)
    if len(var_data) < 100:
        raise ValueError(f"Insufficient samples for {regime}: {len(var_data)}")
    results = strategy.fit(var_data)
    periods = cfg.get("var.irf_periods", 60)
    irf = results.irf(periods=periods)
    return results, irf


def run_varx_pipeline(regime: str = "crisis") -> tuple:
    """运行 VARX 流程（含 Gas Fee 外生变量），返回 (results, irf_obj)"""
    cfg = get_config()
    cfg.ensure_dirs()
    df = load_master()
    strategy = get_var_strategy(regime)
    prepared = strategy.prepare_data(df, include_gas=True)
    if isinstance(prepared, tuple):
        var_data, gas_exog = prepared
    else:
        raise ValueError("prepare_data with include_gas=True must return (var_data, gas_exog)")
    if len(var_data) < 100:
        raise ValueError(f"Insufficient samples for {regime}: {len(var_data)}")
    results = strategy.fit(var_data, exog=gas_exog)
    periods = cfg.get("var.irf_periods", 60)
    irf = results.irf(periods=periods)
    return results, irf


def run_var_gas_comparison(regime: str = "crisis") -> dict:
    """
    有/无 Gas Fee 对比实验。
    返回 dict:
      - baseline: (results, irf)
      - varx: (results, irf)
      - depeg_to_premium_irf_baseline, depeg_to_premium_irf_varx
      - fevd_premium_from_depeg_baseline, fevd_premium_from_depeg_varx
    """
    cfg = get_config()
    cfg.ensure_dirs()
    df = load_master()
    strategy = get_var_strategy(regime)
    periods = cfg.get("var.irf_periods", 60)

    # Baseline: 纯 VAR（无 Gas Fee）
    var_data = strategy.prepare_data(df)
    if len(var_data) < 100:
        raise ValueError(f"Insufficient samples for {regime}: {len(var_data)}")
    results_baseline = strategy.fit(var_data)
    irf_baseline = results_baseline.irf(periods=periods)

    # VARX: 含 Gas exog（log_level 或 dummy）
    prepared = strategy.prepare_data(df, include_gas=True)
    var_data, gas_exog = prepared
    results_varx = strategy.fit(var_data, exog=gas_exog)
    irf_varx = results_varx.irf(periods=periods)
    attrs = getattr(gas_exog, "attrs", {})
    threshold_gwei = attrs.get("threshold_gwei")
    adf_pvalue = attrs.get("adf_pvalue")
    exog_name = getattr(gas_exog, "name", "") or ""
    if adf_pvalue is not None:
        gas_exog_desc = f"{'base_fee_gwei' if 'base_fee' in exog_name else 'ln(gas)'} (ADF p={adf_pvalue:.4f})"
    else:
        gas_exog_desc = f"dummy thresh={threshold_gwei}"

    # Depeg → Premium 索引 (0=Depeg, 3=Premium)
    IDX_DEPEG, IDX_PREMIUM = 0, 3
    diff_cols = getattr(results_baseline, "differenced_cols", set())
    irf_b_raw = irf_baseline.orth_irfs[:, IDX_PREMIUM, IDX_DEPEG]
    irf_v_raw = irf_varx.orth_irfs[:, IDX_PREMIUM, IDX_DEPEG]
    if "Premium_bps" in diff_cols:
        irf_b_raw, irf_v_raw = np.cumsum(irf_b_raw), np.cumsum(irf_v_raw)
    out = {
        "baseline": (results_baseline, irf_baseline),
        "varx": (results_varx, irf_varx),
        "regime": regime,
        "gas_threshold_gwei": threshold_gwei,
        "gas_adf_pvalue": adf_pvalue,
        "gas_exog_desc": gas_exog_desc,
        "depeg_to_premium_irf_baseline": irf_b_raw,
        "depeg_to_premium_irf_varx": irf_v_raw,
    }
    fevd_b = results_baseline.fevd(periods=periods)
    fevd_v = results_varx.fevd(periods=periods)
    out["fevd_premium_from_depeg_baseline"] = fevd_b.decomp[IDX_PREMIUM, :, IDX_DEPEG]
    out["fevd_premium_from_depeg_varx"] = fevd_v.decomp[IDX_PREMIUM, :, IDX_DEPEG]
    return out


def save_var_gas_comparison_report(comparison: dict, out_path: Path | None = None) -> None:
    """将 Gas Fee 对比实验写入文本报告，便于论文引用（含时区与对齐验证）"""
    cfg = get_config()
    path = out_path or cfg.get_output_dir() / "var_gas_comparison_report.txt"
    lines = []
    # 时区与对齐验证
    try:
        from ..data.gas_preprocessing import load_gas_fee, validate_gas_master_alignment
        df = load_master()
        strategy = get_var_strategy(comparison["regime"])
        var_data = strategy.prepare_data(df)
        gas = load_gas_fee()
        diag = validate_gas_master_alignment(gas, var_data.index)
        lines = [
            "=" * 70,
            "VAR vs VARX: Gas Fee (Transaction Cost) Comparison",
            f"IAQF 2026 - Regime: {comparison['regime']}",
            "=" * 70,
            "",
            "[Data Alignment & Timezone]",
            f"  Master: {diag['master_tz']}",
            f"  Gas range: {diag['gas_range'][0]} .. {diag['gas_range'][1]}",
            f"  Master range: {diag['master_range'][0]} .. {diag['master_range'][1]}",
            f"  Overlap (aligned minutes): {diag['overlap_minutes']} / {diag['master_total']}",
            "",
        ]
    except Exception as e:
        lines = ["=" * 70, "VAR vs VARX: Gas Fee Comparison", "=" * 70, "", f"[Alignment check skipped: {e}]", ""]

    irf_b = comparison["depeg_to_premium_irf_baseline"]
    irf_v = comparison["depeg_to_premium_irf_varx"]
    fevd_b = comparison["fevd_premium_from_depeg_baseline"]
    fevd_v = comparison["fevd_premium_from_depeg_varx"]
    peak_b, peak_v = abs(irf_b).max(), abs(irf_v).max()
    fevd_final_b = fevd_b[-1] * 100
    fevd_final_v = fevd_v[-1] * 100
    desc = comparison.get("gas_exog_desc", "")
    thresh = comparison.get("gas_threshold_gwei")
    exog_line = (
        desc if desc and ("ADF" in desc or "dummy" in desc)
        else f"gas_congestion_dummy I(gas > {thresh:.1f} Gwei)" if isinstance(thresh, (int, float))
        else "gas exog"
    )
    lines.extend([
        "Baseline (VAR): endog = [Depeg_bps, CMLSI, Basis_bps, Premium_bps]",
        f"VARX: same endog + exog = {exog_line}",
        "",
        "Depeg → Premium IRF:",
        f"  Peak |IRF| (baseline): {peak_b:.4f} bps",
        f"  Peak |IRF| (VARX):     {peak_v:.4f} bps",
        "",
        "Premium FEVD from Depeg (final horizon):",
        f"  Baseline: {fevd_final_b:.2f}%",
        f"  VARX:     {fevd_final_v:.2f}%",
        "",
        "Interpretation: VARX controls for on-chain congestion (Gas).",
        "Gas exog is chosen by config + ADF checks (level / log_level / dummy fallback).",
        "If FEVD/IRF differ, gas friction affects the Depeg→Premium transmission.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] Gas comparison report: {path}")


def run_var_institutional_comparison(regime: str = "crisis") -> dict:
    """
    Crisis 制度摩擦对比：Baseline(gas) vs Main(gas+weekend) vs Robustness(gas+pause)
    返回 IRF、FEVD、重合度 corr(weekend, pause)
    """
    cfg = get_config()
    cfg.ensure_dirs()
    df = load_master()
    strategy = get_var_strategy(regime)
    periods = cfg.get("var.irf_periods", 60)
    IDX_DEPEG, IDX_PREMIUM = 0, 3

    var_data = strategy.prepare_data(df)
    if len(var_data) < 100:
        raise ValueError(f"Insufficient samples for {regime}: {len(var_data)}")

    from ..data.institutional_friction import (
        build_weekend_dummy,
        build_conversion_pause_dummy,
        check_dummy_overlap,
    )

    weekend = build_weekend_dummy(var_data.index)
    pause = build_conversion_pause_dummy(var_data.index)
    overlap = check_dummy_overlap(weekend, pause)

    # 1. Baseline: gas only
    prep_baseline = strategy.prepare_data(df, include_gas=True)
    var_b, exog_b = prep_baseline
    res_baseline = strategy.fit(var_b, exog=exog_b)
    irf_b = res_baseline.irf(periods=periods)
    fevd_b = res_baseline.fevd(periods=periods)

    # 2. Main: gas + weekend
    prep_main = strategy.prepare_data(df, include_gas=True, institutional="weekend")
    var_m, exog_m = prep_main
    res_main = strategy.fit(var_m, exog=exog_m)
    irf_m = res_main.irf(periods=periods)
    fevd_m = res_main.fevd(periods=periods)

    # 3. Robustness: gas + pause
    prep_rob = strategy.prepare_data(df, include_gas=True, institutional="pause")
    var_r, exog_r = prep_rob
    res_rob = strategy.fit(var_r, exog=exog_r)
    irf_r = res_rob.irf(periods=periods)
    fevd_r = res_rob.fevd(periods=periods)

    diff_cols = getattr(res_baseline, "differenced_cols", set())
    irf_b_raw = irf_b.orth_irfs[:, IDX_PREMIUM, IDX_DEPEG]
    irf_m_raw = irf_m.orth_irfs[:, IDX_PREMIUM, IDX_DEPEG]
    irf_r_raw = irf_r.orth_irfs[:, IDX_PREMIUM, IDX_DEPEG]
    if "Premium_bps" in diff_cols:
        irf_b_raw, irf_m_raw, irf_r_raw = np.cumsum(irf_b_raw), np.cumsum(irf_m_raw), np.cumsum(irf_r_raw)
    out = {
        "regime": regime,
        "overlap": overlap,
        "baseline": (res_baseline, irf_b),
        "main": (res_main, irf_m),
        "robustness": (res_rob, irf_r),
        "depeg_to_premium_irf_baseline": irf_b_raw,
        "depeg_to_premium_irf_main": irf_m_raw,
        "depeg_to_premium_irf_robustness": irf_r_raw,
        "fevd_premium_baseline": fevd_b.decomp[IDX_PREMIUM, :, IDX_DEPEG],
        "fevd_premium_main": fevd_m.decomp[IDX_PREMIUM, :, IDX_DEPEG],
        "fevd_premium_robustness": fevd_r.decomp[IDX_PREMIUM, :, IDX_DEPEG],
    }
    return out


IDX_DEPEG, IDX_CMLSI, IDX_BASIS, IDX_PREMIUM = 0, 1, 2, 3


def _irf_to_level(irf_arr: np.ndarray, col_idx: int, diff_cols: set) -> np.ndarray:
    """Cumsum IRF for differenced vars → level response."""
    cols = ["Depeg_bps", "CMLSI", "Basis_bps", "Premium_bps"]
    col = cols[col_idx] if col_idx < len(cols) else None
    if col and col in diff_cols:
        return np.cumsum(irf_arr)
    return irf_arr


def run_nested_model_comparison(regime: str = "crisis") -> dict:
    """
    嵌套模型对比：M0 vs M1 vs M2 vs M3
    M0: 基准 VAR（无 exog）
    M1: M0 + Credit block (Basis_USDC_USDT_BN_bps)
    M2: M0 + Institution block (gas + weekend)
    M3: M0 + Credit + Institution
    返回各模型 Depeg→Basis、Depeg→Premium 的 IRF 与 FEVD，用于检验传导路径衰减。
    """
    cfg = get_config()
    cfg.ensure_dirs()
    df = load_master()
    strategy = get_var_strategy(regime)
    periods = cfg.get("var.irf_periods", 60)

    var_data = strategy.prepare_data(df)
    if len(var_data) < 100:
        raise ValueError(f"Insufficient samples for {regime}: {len(var_data)}")

    models = {}

    # M0: Baseline VAR
    res_m0 = strategy.fit(var_data)
    irf_m0 = res_m0.irf(periods=periods)
    diff_cols = getattr(res_m0, "differenced_cols", set())
    models["M0"] = (res_m0, irf_m0)

    # M1: M0 + Credit block
    try:
        prep_m1 = strategy.prepare_data(df, include_credit=True)
        var_m1, exog_m1 = prep_m1
        res_m1 = strategy.fit(var_m1, exog=exog_m1)
        irf_m1 = res_m1.irf(periods=periods)
        models["M1"] = (res_m1, irf_m1)
    except (ValueError, FileNotFoundError) as e:
        models["M1"] = None
        print(f"  [WARN] M1 (Credit block) skipped: {e}")

    # M2: M0 + Institution block (gas + weekend)
    try:
        prep_m2 = strategy.prepare_data(df, include_gas=True, institutional="weekend")
        var_m2, exog_m2 = prep_m2
        res_m2 = strategy.fit(var_m2, exog=exog_m2)
        irf_m2 = res_m2.irf(periods=periods)
        models["M2"] = (res_m2, irf_m2)
    except (ValueError, FileNotFoundError) as e:
        models["M2"] = None
        print(f"  [WARN] M2 (Institution block) skipped: {e}")

    # M3: M0 + Credit + Institution
    try:
        prep_m3 = strategy.prepare_data(
            df, include_gas=True, institutional="weekend", include_credit=True
        )
        var_m3, exog_m3 = prep_m3
        res_m3 = strategy.fit(var_m3, exog=exog_m3)
        irf_m3 = res_m3.irf(periods=periods)
        models["M3"] = (res_m3, irf_m3)
    except (ValueError, FileNotFoundError) as e:
        models["M3"] = None
        print(f"  [WARN] M3 (Credit + Institution) skipped: {e}")

    # Extract IRF & FEVD for Depeg → Basis, Depeg → Premium
    def _extract(model_key: str):
        if models.get(model_key) is None:
            return None
        res, irf = models[model_key]
        d = getattr(res, "differenced_cols", set())
        irf_basis = _irf_to_level(irf.orth_irfs[:, IDX_BASIS, IDX_DEPEG], IDX_BASIS, d)
        irf_prem = _irf_to_level(irf.orth_irfs[:, IDX_PREMIUM, IDX_DEPEG], IDX_PREMIUM, d)
        fevd = res.fevd(periods=periods)
        fevd_basis = fevd.decomp[IDX_BASIS, :, IDX_DEPEG]
        fevd_prem = fevd.decomp[IDX_PREMIUM, :, IDX_DEPEG]
        return {
            "irf_depeg_to_basis": irf_basis,
            "irf_depeg_to_premium": irf_prem,
            "fevd_basis_from_depeg": fevd_basis,
            "fevd_premium_from_depeg": fevd_prem,
        }

    out = {
        "regime": regime,
        "models": models,
        "diff_cols": diff_cols,
        "M0": _extract("M0"),
        "M1": _extract("M1"),
        "M2": _extract("M2"),
        "M3": _extract("M3"),
    }
    return out


def save_nested_model_report(comparison: dict, out_path: Path | None = None) -> None:
    """嵌套模型对比报告：M0 vs M1 vs M2 vs M3 的 IRF 峰值与 FEVD"""
    cfg = get_config()
    path = out_path or cfg.get_output_dir() / "nested_model_comparison_report.txt"
    lines = [
        "=" * 70,
        "Nested Model Comparison: M0 vs M1 vs M2 vs M3",
        f"IAQF 2026 - Regime: {comparison['regime']}",
        "=" * 70,
        "",
        "[Models]",
        "  M0: Baseline VAR (no exog)",
        "  M1: M0 + Credit block (Basis_USDC_USDT_BN_bps)",
        "  M2: M0 + Institution block (gas + weekend)",
        "  M3: M0 + Credit + Institution",
        "",
    ]
    for key in ["M0", "M1", "M2", "M3"]:
        m = comparison.get(key)
        if m is None:
            lines.append(f"[{key}] skipped")
            continue
        irf_b = m["irf_depeg_to_basis"]
        irf_p = m["irf_depeg_to_premium"]
        fevd_b = m["fevd_basis_from_depeg"]
        fevd_p = m["fevd_premium_from_depeg"]
        peak_b = float(np.abs(irf_b).max())
        peak_p = float(np.abs(irf_p).max())
        fevd_final_b = float(fevd_b[-1]) * 100
        fevd_final_p = float(fevd_p[-1]) * 100
        lines.extend([
            f"[{key}]",
            f"  Depeg→Basis IRF |peak|: {peak_b:.4f} bps",
            f"  Depeg→Premium IRF |peak|: {peak_p:.4f} bps",
            f"  Basis FEVD from Depeg (final): {fevd_final_b:.2f}%",
            f"  Premium FEVD from Depeg (final): {fevd_final_p:.2f}%",
            "",
        ])
    lines.extend([
        "[Interpretation]",
        "  If M1/M2/M3 IRF attenuate vs M0: Credit/Institution blocks absorb part of Depeg→Basis/Premium transmission.",
        "  Residual direct effect = IRF remaining after adding both blocks (M3).",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] Nested model report: {path}")


def save_var_institutional_report(comparison: dict, out_path: Path | None = None) -> None:
    """制度摩擦对比报告：重合度、IRF、FEVD、系数解读"""
    cfg = get_config()
    path = out_path or cfg.get_output_dir() / "var_institutional_report.txt"
    ov = comparison.get("overlap", {})
    lines = [
        "=" * 70,
        "VARX Institutional Friction: Gas vs Gas+Weekend vs Gas+Pause",
        f"IAQF 2026 - Regime: {comparison['regime']}",
        "=" * 70,
        "",
        "[Dummy Overlap] corr(is_weekend, conversion_pause)",
        f"  corr = {ov.get('corr', 0):.4f}  |  n = {ov.get('n', 0):,}  |  {ov.get('rule', '')}",
        "",
        "[Models]",
        "  Baseline:     exog = gas (base_fee_gwei)",
        "  Main:         exog = [gas, is_weekend]",
        "  Robustness:   exog = [gas, conversion_pause]  (pause: 2023-03-10 21:00–03-13 09:00 ET)",
        "",
        "[Depeg → Premium IRF Peak |IRF|]",
        f"  Baseline:   {abs(comparison['depeg_to_premium_irf_baseline']).max():.4f} bps",
        f"  Main:       {abs(comparison['depeg_to_premium_irf_main']).max():.4f} bps",
        f"  Robustness: {abs(comparison['depeg_to_premium_irf_robustness']).max():.4f} bps",
        "",
        "[Premium FEVD from Depeg (final horizon)]",
        f"  Baseline:   {comparison['fevd_premium_baseline'][-1]*100:.2f}%",
        f"  Main:       {comparison['fevd_premium_main'][-1]*100:.2f}%",
        f"  Robustness: {comparison['fevd_premium_robustness'][-1]*100:.2f}%",
        "",
        "[Interpretation]",
        "  Cost friction (gas) vs institutional friction (weekend/pause) marginal contribution.",
        "  If weekend/pause significant: non-price institutional constraints matter.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] Institutional report: {path}")
