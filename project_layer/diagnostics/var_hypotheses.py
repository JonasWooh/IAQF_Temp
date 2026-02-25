"""VAR 假设检验补充：传导路径、regime 差异、切片动机、Cholesky 顺序敏感性。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import get_config
from ..models.var_runner import load_master, get_var_strategy


DEFAULT_VAR_COLS = ["Depeg_bps", "CMLSI", "Basis_bps", "Premium_bps"]
REDUCED_VAR_COLS = ["Depeg_bps", "Basis_bps", "Premium_bps"]
REGIMES = ["normal", "crisis", "recovery"]


def _interval_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _half_life_from_peak(arr: np.ndarray) -> int | None:
    """从峰值后首次跌回峰值一半的 horizon（分钟）."""
    if len(arr) == 0:
        return None
    peak_idx = int(np.nanargmax(np.abs(arr)))
    peak_abs = float(np.abs(arr[peak_idx]))
    if not np.isfinite(peak_abs) or peak_abs <= 1e-12:
        return None
    threshold = 0.5 * peak_abs
    for t in range(peak_idx + 1, len(arr)):
        if abs(arr[t]) <= threshold:
            return int(t - peak_idx)
    return None


def _series_metrics(arr: np.ndarray, lower: np.ndarray | None = None, upper: np.ndarray | None = None) -> dict:
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0 or np.all(~np.isfinite(arr)):
        return {}
    peak_idx = int(np.nanargmax(np.abs(arr)))
    peak_val = float(arr[peak_idx])
    peak_abs = float(abs(peak_val))
    out = {
        "peak_idx": peak_idx,
        "peak_val": peak_val,
        "peak_abs": peak_abs,
        "cum_signed": float(np.nansum(arr)),
        "cum_abs": float(np.nansum(np.abs(arr))),
        "half_life_from_peak": _half_life_from_peak(arr),
    }
    if lower is not None and upper is not None and len(lower) > peak_idx and len(upper) > peak_idx:
        lo = float(lower[peak_idx])
        hi = float(upper[peak_idx])
        out["peak_ci"] = [lo, hi]
        out["zero_excluded_at_peak"] = bool((lo > 0 and hi > 0) or (lo < 0 and hi < 0))
    return out


def _irf_level(results, irf_arr: np.ndarray, response_col: str) -> np.ndarray:
    diff_cols = getattr(results, "differenced_cols", set())
    return np.cumsum(irf_arr) if response_col in diff_cols else irf_arr


def _extract_irf_series(results, irf_obj, response_col: str, shock_col: str) -> np.ndarray:
    cols = list(getattr(results, "var_col_names", []))
    if response_col not in cols or shock_col not in cols:
        raise KeyError(f"{response_col}/{shock_col} not in VAR columns: {cols}")
    ridx = cols.index(response_col)
    sidx = cols.index(shock_col)
    arr = irf_obj.orth_irfs[:, ridx, sidx]
    return _irf_level(results, arr, response_col)


def _extract_ci_series(results, lower_arr: np.ndarray, upper_arr: np.ndarray, response_col: str, shock_col: str) -> tuple[np.ndarray, np.ndarray]:
    cols = list(getattr(results, "var_col_names", []))
    ridx = cols.index(response_col)
    sidx = cols.index(shock_col)
    lo = _irf_level(results, lower_arr[:, ridx, sidx], response_col)
    hi = _irf_level(results, upper_arr[:, ridx, sidx], response_col)
    return lo, hi


def _fit_var_on_data(var_data: pd.DataFrame):
    """复用现有策略的平稳性处理与拟合逻辑（不依赖 regime 切片）."""
    return get_var_strategy("crisis").fit(var_data)


def _fit_regime_model(df: pd.DataFrame, regime: str):
    strategy = get_var_strategy(regime)
    var_data = strategy.prepare_data(df)
    if len(var_data) < 100:
        raise ValueError(f"Insufficient samples for {regime}: {len(var_data)}")
    results = strategy.fit(var_data)
    return results


def _fit_regime_custom_cols(df: pd.DataFrame, regime: str, cols: list[str], same_sample_as_full: bool = False):
    strategy = get_var_strategy(regime)
    if same_sample_as_full:
        base = strategy.get_data_slice(df[DEFAULT_VAR_COLS].dropna())
        var_data = base[cols].copy()
    else:
        var_data = strategy.get_data_slice(df[cols].dropna())
    if len(var_data) < 100:
        raise ValueError(f"Insufficient samples for {regime}/{cols}: {len(var_data)}")
    return strategy.fit(var_data)


def _fit_full_sample_model(df: pd.DataFrame, cols: list[str]):
    cfg = get_config()
    start = cfg.get("dates.start")
    end = cfg.get("dates.end")
    var_data = df[cols].dropna().loc[start:end]
    if len(var_data) < 100:
        raise ValueError(f"Insufficient samples for full-sample {cols}: {len(var_data)}")
    return _fit_var_on_data(var_data)


def _irf_and_bands(results):
    cfg = get_config()
    periods = cfg.get("var.irf_periods", 60)
    repl = cfg.get("var.irf_ci_repl_triple", cfg.get("var.irf_ci_repl", 150))
    signif = cfg.get("var.irf_ci_signif", 0.05)
    seed = cfg.get("var.irf_ci_seed", 42)
    irf_obj = results.irf(periods=periods)
    lower, upper = results.irf_errband_mc(orth=True, repl=repl, steps=periods, signif=signif, seed=seed)
    return irf_obj, lower, upper


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) != len(b) or len(a) < 3:
        return None
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _fit_with_order(df: pd.DataFrame, regime: str, order_cols: list[str]):
    strategy = get_var_strategy(regime)
    base = strategy.prepare_data(df)
    var_data = base[order_cols].dropna()
    if len(var_data) < 100:
        raise ValueError(f"Insufficient samples for ordering {order_cols}: {len(var_data)}")
    return strategy.fit(var_data)


def run_var_hypothesis_tests(out_path: Path | None = None) -> dict:
    """
    输出：
      - transmission path（含/不含 CMLSI 对比）
      - regime IRF 定量比较
      - full-sample vs regime（切片动机）
      - Cholesky 顺序敏感性（轻量）
    """
    cfg = get_config()
    cfg.ensure_dirs()
    df = load_master()
    out_dir = cfg.get_output_dir()
    out_json = out_path or (out_dir / "var_hypotheses_metrics.json")

    periods = cfg.get("var.irf_periods", 60)
    metrics: dict[str, object] = {
        "meta": {
            "periods": periods,
            "default_var_cols": DEFAULT_VAR_COLS,
            "reduced_var_cols": REDUCED_VAR_COLS,
            "regimes": REGIMES,
            "heuristic_threshold_relative_change": 0.15,
        }
    }
    viz_data: dict[str, object] = {
        "transmission": {},
        "regime_series": {},
        "full_vs_regime_series": {},
    }

    # ------------------------------------------------------------------
    # A. 传导路径（危机期）: 显式化 H1/H2 的证据
    # ------------------------------------------------------------------
    crisis_full = _fit_regime_custom_cols(df, "crisis", DEFAULT_VAR_COLS, same_sample_as_full=True)
    crisis_red = _fit_regime_custom_cols(df, "crisis", REDUCED_VAR_COLS, same_sample_as_full=True)
    irf_full, lower_full, upper_full = _irf_and_bands(crisis_full)
    irf_red = crisis_red.irf(periods=periods)

    trans = {}
    # H1: Depeg -> CMLSI
    h1_arr = _extract_irf_series(crisis_full, irf_full, "CMLSI", "Depeg_bps")
    h1_lo, h1_hi = _extract_ci_series(crisis_full, lower_full, upper_full, "CMLSI", "Depeg_bps")
    viz_data["transmission"]["depeg_to_cmlsi"] = {
        "series": h1_arr.tolist(),
        "lower": h1_lo.tolist(),
        "upper": h1_hi.tolist(),
    }
    trans["depeg_to_cmlsi"] = _series_metrics(h1_arr, h1_lo, h1_hi)

    # H2: CMLSI -> Basis/Premium
    for response in ["Basis_bps", "Premium_bps"]:
        arr = _extract_irf_series(crisis_full, irf_full, response, "CMLSI")
        lo, hi = _extract_ci_series(crisis_full, lower_full, upper_full, response, "CMLSI")
        viz_data["transmission"][f"cmlsi_to_{response.lower()}"] = {
            "series": arr.tolist(),
            "lower": lo.tolist(),
            "upper": hi.tolist(),
        }
        trans[f"cmlsi_to_{response.lower()}"] = _series_metrics(arr, lo, hi)

    # 含/不含 CMLSI 的 Depeg 响应对比（同样本）
    compare = {}
    compare_map = {
        "Basis_bps": ("depeg_to_basis", "Basis_bps"),
        "Premium_bps": ("depeg_to_premium", "Premium_bps"),
    }
    for response_col, (key, response_name) in compare_map.items():
        arr_full = _extract_irf_series(crisis_full, irf_full, response_col, "Depeg_bps")
        arr_red = _extract_irf_series(crisis_red, irf_red, response_col, "Depeg_bps")
        m_full = _series_metrics(arr_full)
        m_red = _series_metrics(arr_red)
        peak_full = m_full.get("peak_abs", np.nan) or np.nan
        peak_red = m_red.get("peak_abs", np.nan) or np.nan
        cum_full = m_full.get("cum_abs", np.nan) or np.nan
        cum_red = m_red.get("cum_abs", np.nan) or np.nan
        peak_rel_change = None
        if np.isfinite(peak_full) and abs(peak_full) > 1e-12 and np.isfinite(peak_red):
            peak_rel_change = float((peak_red - peak_full) / abs(peak_full))
        cum_rel_change = None
        if np.isfinite(cum_full) and abs(cum_full) > 1e-12 and np.isfinite(cum_red):
            cum_rel_change = float((cum_red - cum_full) / abs(cum_full))
        viz_data["transmission"][f"compare_{key}"] = {
            "with_cmlsi": arr_full.tolist(),
            "without_cmlsi": arr_red.tolist(),
        }
        compare[key] = {
            "full_with_cmlsi": m_full,
            "reduced_no_cmlsi": m_red,
            "peak_abs_relative_change_reduced_vs_full": peak_rel_change,
            "cum_abs_relative_change_reduced_vs_full": cum_rel_change,
            "shape_corr_full_vs_reduced": _safe_corr(arr_full, arr_red),
        }

    fevd_full = crisis_full.fevd(periods=periods)
    cols_full = list(getattr(crisis_full, "var_col_names", []))
    idx = {c: cols_full.index(c) for c in cols_full}
    fevd_summary = {}
    for response in ["Basis_bps", "Premium_bps"]:
        if response in idx and "Depeg_bps" in idx and "CMLSI" in idx:
            fevd_summary[response] = {
                "from_depeg_final_pct": float(fevd_full.decomp[idx[response], :, idx["Depeg_bps"]][-1] * 100),
                "from_cmlsi_final_pct": float(fevd_full.decomp[idx[response], :, idx["CMLSI"]][-1] * 100),
            }

    # 传导路径“是否成立”的启发式判定（非正式中介检验）
    threshold = metrics["meta"]["heuristic_threshold_relative_change"]
    trans["heuristic_path_assessment"] = {
        "criteria": [
            "H1: Depeg->CMLSI peak CI excludes zero",
            "H2a: CMLSI->Basis/Premium peak CI excludes zero (at least one response)",
            f"H2b: Removing CMLSI changes Depeg response materially (|Δpeak|>{threshold:.0%} or |Δcum|>{threshold:.0%})",
        ],
        "h1_supported": bool(trans["depeg_to_cmlsi"].get("zero_excluded_at_peak", False)),
        "h2a_supported_any": bool(
            trans.get("cmlsi_to_basis_bps", {}).get("zero_excluded_at_peak", False)
            or trans.get("cmlsi_to_premium_bps", {}).get("zero_excluded_at_peak", False)
        ),
        "h2b_supported_any": bool(
            any(
                abs(v.get("peak_abs_relative_change_reduced_vs_full") or 0) > threshold
                or abs(v.get("cum_abs_relative_change_reduced_vs_full") or 0) > threshold
                for v in compare.values()
            )
        ),
        "note": "Heuristic evidence only; not a formal causal mediation test.",
    }
    metrics["transmission_path"] = {
        "crisis_full_model_cols": cols_full,
        "crisis_reduced_model_cols": list(getattr(crisis_red, "var_col_names", [])),
        "irf_metrics": trans,
        "with_vs_without_cmlsi": compare,
        "fevd_final_horizon": fevd_summary,
    }

    # ------------------------------------------------------------------
    # B. 三阶段传导差异：量化比较 + 简单 CI 重叠判断
    # ------------------------------------------------------------------
    regime_models = {}
    regime_irf = {}
    regime_bands = {}
    for regime in REGIMES:
        res = _fit_regime_model(df, regime)
        regime_models[regime] = res
        irf_obj, lower, upper = _irf_and_bands(res)
        regime_irf[regime] = irf_obj
        regime_bands[regime] = (lower, upper)

    regime_quant = {}
    for response in ["CMLSI", "Basis_bps", "Premium_bps"]:
        per_regime = {}
        for regime in REGIMES:
            arr = _extract_irf_series(regime_models[regime], regime_irf[regime], response, "Depeg_bps")
            lower, upper = regime_bands[regime]
            lo, hi = _extract_ci_series(regime_models[regime], lower, upper, response, "Depeg_bps")
            per_regime[regime] = {
                "series": arr.tolist(),
                "metrics": _series_metrics(arr, lo, hi),
                "ci_lower": lo.tolist(),
                "ci_upper": hi.tolist(),
            }
        # crisis vs others @ crisis peak horizon 的简单比较
        crisis_peak = per_regime["crisis"]["metrics"].get("peak_idx")
        pairwise = {}
        for other in ["normal", "recovery"]:
            if crisis_peak is None:
                continue
            c_lo = per_regime["crisis"]["ci_lower"][crisis_peak]
            c_hi = per_regime["crisis"]["ci_upper"][crisis_peak]
            o_lo = per_regime[other]["ci_lower"][crisis_peak]
            o_hi = per_regime[other]["ci_upper"][crisis_peak]
            pairwise[f"crisis_vs_{other}_at_crisis_peak"] = {
                "horizon": int(crisis_peak),
                "crisis_ci": [float(c_lo), float(c_hi)],
                "other_ci": [float(o_lo), float(o_hi)],
                "ci_overlap": _interval_overlap((float(c_lo), float(c_hi)), (float(o_lo), float(o_hi))),
                "peak_abs_ratio_crisis_over_other": (
                    float(
                        (per_regime["crisis"]["metrics"].get("peak_abs", np.nan) or np.nan)
                        / (per_regime[other]["metrics"].get("peak_abs", np.nan) or np.nan)
                    )
                    if (per_regime[other]["metrics"].get("peak_abs", 0) or 0) not in (0, None)
                    else None
                ),
                "cum_abs_ratio_crisis_over_other": (
                    float(
                        (per_regime["crisis"]["metrics"].get("cum_abs", np.nan) or np.nan)
                        / (per_regime[other]["metrics"].get("cum_abs", np.nan) or np.nan)
                    )
                    if (per_regime[other]["metrics"].get("cum_abs", 0) or 0) not in (0, None)
                    else None
                ),
            }
        regime_quant[response] = {
            "per_regime": {k: v["metrics"] for k, v in per_regime.items()},
            "pairwise_checks": pairwise,
        }
        viz_data["regime_series"][response] = {
            rg: {
                "series": per_regime[rg]["series"],
                "ci_lower": per_regime[rg]["ci_lower"],
                "ci_upper": per_regime[rg]["ci_upper"],
            }
            for rg in REGIMES
        }
    metrics["regime_quantitative_comparison"] = regime_quant

    # ------------------------------------------------------------------
    # C. 为什么做 regime 切片：全样本 vs 分阶段对比（平均化扭曲）
    # ------------------------------------------------------------------
    full_model = _fit_full_sample_model(df, DEFAULT_VAR_COLS)
    full_irf = full_model.irf(periods=periods)
    full_vs_regime = {}
    for response in ["CMLSI", "Basis_bps", "Premium_bps"]:
        arr_full = _extract_irf_series(full_model, full_irf, response, "Depeg_bps")
        out = {"full_sample": _series_metrics(arr_full)}
        viz_data["full_vs_regime_series"][response] = {
            "full_sample": arr_full.tolist(),
            "regimes": {},
        }
        for regime in REGIMES:
            arr_r = _extract_irf_series(regime_models[regime], regime_irf[regime], response, "Depeg_bps")
            m_r = regime_quant[response]["per_regime"][regime]
            viz_data["full_vs_regime_series"][response]["regimes"][regime] = arr_r.tolist()
            peak_full = out["full_sample"].get("peak_abs", np.nan) or np.nan
            peak_reg = m_r.get("peak_abs", np.nan) or np.nan
            attenuation = None
            if np.isfinite(peak_reg) and abs(peak_reg) > 1e-12 and np.isfinite(peak_full):
                attenuation = float(1 - (peak_full / peak_reg))
            out[regime] = {
                "metrics": m_r,
                "shape_corr_with_full": _safe_corr(arr_full, arr_r),
                "peak_abs_attenuation_full_vs_regime": attenuation,
            }
        full_vs_regime[response] = out

    # endog 协方差层面的简单 regime shift 证据（主线变量）
    cfg_dates = cfg.get("phases", {})
    endog_cov_shift = {}
    try:
        endog = df[DEFAULT_VAR_COLS].dropna()
        cov_reg = {}
        for regime in REGIMES:
            start = cfg.get(f"phases.{regime if regime != 'crisis' else 'stressed'}.start") if regime != "crisis" else cfg.get("phases.stressed.start")
            end = cfg.get(f"phases.{regime if regime != 'crisis' else 'stressed'}.end") if regime != "crisis" else cfg.get("phases.stressed.end")
            cov_reg[regime] = endog.loc[start:end].cov().values
        base_cov = cov_reg["normal"]
        for regime in ["crisis", "recovery"]:
            endog_cov_shift[f"fro_norm_vs_{regime}"] = float(np.linalg.norm(base_cov - cov_reg[regime], ord="fro"))
    except Exception:
        endog_cov_shift = {}
    metrics["regime_slicing_motivation"] = {
        "full_vs_regime_irf": full_vs_regime,
        "endog_covariance_shift_frobenius": endog_cov_shift,
        "note": "Full-sample VAR mixes distinct regimes and can attenuate crisis-specific responses.",
    }

    # ------------------------------------------------------------------
    # D. Cholesky 顺序敏感性（轻量）
    # ------------------------------------------------------------------
    ordering_specs = {
        "baseline": ["Depeg_bps", "CMLSI", "Basis_bps", "Premium_bps"],
        "alt_cmlsi_before_depeg": ["CMLSI", "Depeg_bps", "Basis_bps", "Premium_bps"],
        "alt_basis_before_cmlsi": ["Depeg_bps", "Basis_bps", "CMLSI", "Premium_bps"],
    }
    order_metrics = {}
    for name, order in ordering_specs.items():
        res = _fit_with_order(df, "crisis", order)
        irf_obj = res.irf(periods=periods)
        order_metrics[name] = {"order": order}
        for response in ["CMLSI", "Basis_bps", "Premium_bps"]:
            arr = _extract_irf_series(res, irf_obj, response, "Depeg_bps")
            order_metrics[name][f"depeg_to_{response.lower()}"] = _series_metrics(arr)
    metrics["ordering_sensitivity"] = order_metrics

    # JSON 输出
    out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [OK] VAR hypothesis metrics: {out_json}")
    _plot_var_hypotheses_figures(metrics, viz_data, out_dir)

    # 文本报告输出（便于论文写作引用）
    _write_var_hypotheses_report(metrics, out_dir / "var_hypotheses_report.txt")
    return metrics


def _fmt_metric(m: dict, key: str, default: str = "N/A") -> str:
    v = m.get(key)
    if v is None:
        return default
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4f}"
    return str(v)


def _plot_series_ci(ax, x, series, lower=None, upper=None, color="#3498DB", label=None, ylabel=None, title=None):
    if lower is not None and upper is not None:
        ax.fill_between(x, lower, upper, color=color, alpha=0.2)
    ax.plot(x, series, color=color, linewidth=2, label=label)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_xlabel("Minutes after shock")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend(fontsize=8)


def _plot_var_hypotheses_figures(metrics: dict, viz_data: dict, out_dir: Path) -> None:
    cfg = get_config()
    fig_dir = cfg.get_figures_dir()
    fig_dir.mkdir(parents=True, exist_ok=True)

    trans = viz_data.get("transmission", {})
    trans_metrics = metrics.get("transmission_path", {}).get("irf_metrics", {})
    periods = int(metrics.get("meta", {}).get("periods", 60))
    x = np.arange(periods + 1)

    # ------------------------------------------------------------------
    # Figure 21: Transmission path + with/without CMLSI
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # H1: Depeg -> CMLSI
    d2c = trans.get("depeg_to_cmlsi", {})
    _plot_series_ci(
        axes[0, 0],
        x,
        np.asarray(d2c.get("series", []), dtype=float),
        np.asarray(d2c.get("lower", []), dtype=float) if d2c else None,
        np.asarray(d2c.get("upper", []), dtype=float) if d2c else None,
        color="#2E86AB",
        label="Depeg → CMLSI",
        ylabel="CMLSI response",
        title="H1: Depeg → CMLSI (Crisis, with CI)",
    )

    # H2a: CMLSI -> Basis
    c2b = trans.get("cmlsi_to_basis_bps", {})
    _plot_series_ci(
        axes[0, 1],
        x,
        np.asarray(c2b.get("series", []), dtype=float),
        np.asarray(c2b.get("lower", []), dtype=float) if c2b else None,
        np.asarray(c2b.get("upper", []), dtype=float) if c2b else None,
        color="#E67E22",
        label="CMLSI → Basis",
        ylabel="Basis response (bps)",
        title="H2a: CMLSI → Basis_bps (Crisis, with CI)",
    )

    # H2b: CMLSI -> Premium
    c2p = trans.get("cmlsi_to_premium_bps", {})
    _plot_series_ci(
        axes[1, 0],
        x,
        np.asarray(c2p.get("series", []), dtype=float),
        np.asarray(c2p.get("lower", []), dtype=float) if c2p else None,
        np.asarray(c2p.get("upper", []), dtype=float) if c2p else None,
        color="#9B59B6",
        label="CMLSI → Premium",
        ylabel="Premium response (bps)",
        title="H2b: CMLSI → Premium_bps (Crisis, with CI)",
    )

    # Depeg -> Basis, with vs without CMLSI
    d2b = trans.get("compare_depeg_to_basis", {})
    y_full = np.asarray(d2b.get("with_cmlsi", []), dtype=float)
    y_red = np.asarray(d2b.get("without_cmlsi", []), dtype=float)
    ax = axes[1, 1]
    ax.plot(np.arange(len(y_full)), y_full, color="#E74C3C", linewidth=2, label="With CMLSI")
    ax.plot(np.arange(len(y_red)), y_red, color="#3498DB", linewidth=2, linestyle="--", label="Without CMLSI")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title("Depeg → Basis_bps (Crisis): With vs Without CMLSI")
    ax.set_xlabel("Minutes after shock")
    ax.set_ylabel("Basis response (bps)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Depeg -> Premium, with vs without CMLSI
    d2p = trans.get("compare_depeg_to_premium", {})
    y_full = np.asarray(d2p.get("with_cmlsi", []), dtype=float)
    y_red = np.asarray(d2p.get("without_cmlsi", []), dtype=float)
    ax = axes[2, 0]
    ax.plot(np.arange(len(y_full)), y_full, color="#E74C3C", linewidth=2, label="With CMLSI")
    ax.plot(np.arange(len(y_red)), y_red, color="#3498DB", linewidth=2, linestyle="--", label="Without CMLSI")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title("Depeg → Premium_bps (Crisis): With vs Without CMLSI")
    ax.set_xlabel("Minutes after shock")
    ax.set_ylabel("Premium response (bps)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Summary text panel
    ax = axes[2, 1]
    ax.axis("off")
    heur = trans_metrics.get("heuristic_path_assessment", {})
    b_cmp = metrics.get("transmission_path", {}).get("with_vs_without_cmlsi", {}).get("depeg_to_basis", {})
    p_cmp = metrics.get("transmission_path", {}).get("with_vs_without_cmlsi", {}).get("depeg_to_premium", {})
    txt = (
        "Transmission Path Summary (heuristic)\n\n"
        f"H1 Depeg→CMLSI peak CI excludes 0: {heur.get('h1_supported')}\n"
        f"H2 CMLSI→(Basis/Premium) peak CI excludes 0: {heur.get('h2a_supported_any')}\n"
        f"Removing CMLSI materially changes Depeg response: {heur.get('h2b_supported_any')}\n\n"
        "With vs Without CMLSI (Reduced vs Full)\n"
        f"Basis Δpeak_abs: {_fmt_metric(b_cmp, 'peak_abs_relative_change_reduced_vs_full')}\n"
        f"Basis Δcum_abs: {_fmt_metric(b_cmp, 'cum_abs_relative_change_reduced_vs_full')}\n"
        f"Premium Δpeak_abs: {_fmt_metric(p_cmp, 'peak_abs_relative_change_reduced_vs_full')}\n"
        f"Premium Δcum_abs: {_fmt_metric(p_cmp, 'cum_abs_relative_change_reduced_vs_full')}\n"
    )
    ax.text(0.02, 0.98, txt, va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle("VAR Hypothesis Tests (A): Transmission Path and With/Without CMLSI", fontsize=14, y=1.01)
    plt.tight_layout()
    out1 = fig_dir / "21_var_hypotheses_transmission.png"
    fig.savefig(out1, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out1.name}")

    # ------------------------------------------------------------------
    # Figure 22: Regime differences + full-sample slicing + ordering sensitivity
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    response_panels = [("CMLSI", axes2[0, 0]), ("Basis_bps", axes2[0, 1]), ("Premium_bps", axes2[1, 0])]
    colors = {
        "full_sample": "#222222",
        "normal": "#2ECC71",
        "crisis": "#E74C3C",
        "recovery": "#3498DB",
    }
    for response, ax in response_panels:
        blk = viz_data.get("full_vs_regime_series", {}).get(response, {})
        y_full = np.asarray(blk.get("full_sample", []), dtype=float)
        if len(y_full):
            ax.plot(np.arange(len(y_full)), y_full, color=colors["full_sample"], linewidth=2, label="Full sample")
        for rg in REGIMES:
            y = np.asarray(blk.get("regimes", {}).get(rg, []), dtype=float)
            if len(y):
                ax.plot(np.arange(len(y)), y, color=colors[rg], linewidth=2, label=rg.title())
        ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(f"Depeg → {response}: Full Sample vs Regimes")
        ax.set_xlabel("Minutes after shock")
        ax.set_ylabel("Response" + (" (bps)" if response != "CMLSI" else ""))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Ordering sensitivity grouped bars
    ax = axes2[1, 1]
    order_sens = metrics.get("ordering_sensitivity", {})
    order_names = list(order_sens.keys())
    labels = ["depeg_to_cmlsi", "depeg_to_basis_bps", "depeg_to_premium_bps"]
    x_pos = np.arange(len(order_names))
    width = 0.22
    bar_colors = ["#2E86AB", "#E67E22", "#9B59B6"]
    for i, key in enumerate(labels):
        vals = []
        for name in order_names:
            vals.append(float(order_sens.get(name, {}).get(key, {}).get("peak_abs", np.nan) or np.nan))
        ax.bar(x_pos + (i - 1) * width, vals, width=width, color=bar_colors[i], alpha=0.85, label=key)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(order_names, rotation=15, ha="right")
    ax.set_ylabel("Peak |IRF|")
    ax.set_title("Cholesky Ordering Sensitivity (Crisis)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    fig2.suptitle("VAR Hypothesis Tests (B): Regime Differences, Slicing Motivation, Ordering Robustness", fontsize=14, y=1.01)
    plt.tight_layout()
    out2 = fig_dir / "22_var_hypotheses_regime_slicing.png"
    fig2.savefig(out2, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    plt.close(fig2)
    print(f"  [OK] {out2.name}")


def _write_var_hypotheses_report(metrics: dict, path: Path) -> None:
    trans = metrics.get("transmission_path", {})
    trans_irf = trans.get("irf_metrics", {})
    with_without = trans.get("with_vs_without_cmlsi", {})
    regime = metrics.get("regime_quantitative_comparison", {})
    slicing = metrics.get("regime_slicing_motivation", {})
    order_sens = metrics.get("ordering_sensitivity", {})

    lines = [
        "=" * 70,
        "VAR Hypothesis Tests Supplement (Transmission / Regime / Slicing)",
        "IAQF 2026 - Project Layer",
        "=" * 70,
        "",
        "[A] Transmission Path (Crisis): Depeg -> CMLSI -> Basis/Premium (heuristic evidence)",
    ]
    d2c = trans_irf.get("depeg_to_cmlsi", {})
    lines.extend([
        f"  H1 Depeg->CMLSI: peak={_fmt_metric(d2c,'peak_val')} @ t={_fmt_metric(d2c,'peak_idx')}, "
        f"|peak|={_fmt_metric(d2c,'peak_abs')}, zero-excluded={d2c.get('zero_excluded_at_peak')}",
    ])
    for key in ["cmlsi_to_basis_bps", "cmlsi_to_premium_bps"]:
        m = trans_irf.get(key, {})
        lines.append(
            f"  H2 {key}: peak={_fmt_metric(m,'peak_val')} @ t={_fmt_metric(m,'peak_idx')}, "
            f"|peak|={_fmt_metric(m,'peak_abs')}, zero-excluded={m.get('zero_excluded_at_peak')}"
        )
    for k, v in with_without.items():
        lines.append(
            f"  With vs Without CMLSI [{k}]: Δpeak_abs(reduced-full)/full={_fmt_metric(v,'peak_abs_relative_change_reduced_vs_full')}, "
            f"Δcum_abs={_fmt_metric(v,'cum_abs_relative_change_reduced_vs_full')}, corr={_fmt_metric(v,'shape_corr_full_vs_reduced')}"
        )
    heur = trans_irf.get("heuristic_path_assessment", {})
    lines.extend([
        "",
        f"  Heuristic path assessment: H1={heur.get('h1_supported')}, H2a={heur.get('h2a_supported_any')}, H2b={heur.get('h2b_supported_any')}",
        f"  Note: {heur.get('note', '')}",
        "",
        "[B] Regime Differences (Quantified): Depeg response by regime",
    ])
    for response, block in regime.items():
        lines.append(f"  {response}:")
        per = block.get("per_regime", {})
        for rg in ["normal", "crisis", "recovery"]:
            m = per.get(rg, {})
            lines.append(
                f"    {rg:8s} |peak|={_fmt_metric(m,'peak_abs')} @t={_fmt_metric(m,'peak_idx')} "
                f"half-life={_fmt_metric(m,'half_life_from_peak')} cum_abs={_fmt_metric(m,'cum_abs')} "
                f"zero-excl-peak={m.get('zero_excluded_at_peak')}"
            )
        for pair, chk in block.get("pairwise_checks", {}).items():
            lines.append(
                f"    {pair}: horizon={chk.get('horizon')}, CI overlap={chk.get('ci_overlap')}, "
                f"peak ratio={_fmt_metric(chk,'peak_abs_ratio_crisis_over_other')}, "
                f"cum_abs ratio={_fmt_metric(chk,'cum_abs_ratio_crisis_over_other')}"
            )
    lines.extend([
        "",
        "[C] Why Regime Slicing (Full-sample VAR vs regime VAR)",
    ])
    full_vs = slicing.get("full_vs_regime_irf", {})
    for response, blk in full_vs.items():
        full_m = blk.get("full_sample", {})
        lines.append(
            f"  {response} full-sample: |peak|={_fmt_metric(full_m,'peak_abs')} @t={_fmt_metric(full_m,'peak_idx')} "
            f"cum_abs={_fmt_metric(full_m,'cum_abs')}"
        )
        for rg in ["normal", "crisis", "recovery"]:
            x = blk.get(rg, {})
            lines.append(
                f"    vs {rg}: corr={_fmt_metric(x,'shape_corr_with_full')}, "
                f"peak attenuation(full vs {rg})={_fmt_metric(x,'peak_abs_attenuation_full_vs_regime')}"
            )
    cov = slicing.get("endog_covariance_shift_frobenius", {})
    if cov:
        lines.append(
            "  Endog covariance shift (Frobenius): "
            + ", ".join(f"{k}={v:.4f}" for k, v in cov.items() if isinstance(v, (int, float)))
        )
    lines.extend([
        "",
        "[D] Cholesky Ordering Sensitivity (Crisis, lightweight robustness)",
    ])
    for name, blk in order_sens.items():
        order = blk.get("order", [])
        lines.append(f"  {name}: order={order}")
        for key in ["depeg_to_cmlsi", "depeg_to_basis_bps", "depeg_to_premium_bps"]:
            m = blk.get(key, {})
            lines.append(f"    {key}: |peak|={_fmt_metric(m,'peak_abs')} @t={_fmt_metric(m,'peak_idx')}")
    lines.extend([
        "",
        "[Interpretation Guide]",
        "  - 若 Crisis 的 |peak|/cum_abs/half-life 明显大于 Normal/Recovery，支持“危机期传导更强/更持久”。",
        "  - 若去掉 CMLSI 后 Depeg 响应显著改变，说明 CMLSI 至少是重要传导通道（启发式证据）。",
        "  - 若 full-sample 与 Crisis IRF 差异显著，说明混合估计会掩盖危机机制，regime 切片有必要。",
        "  - Cholesky 顺序敏感性若很高，应在论文中明确作为识别假设与局限。",
        "",
        "=" * 70,
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] VAR hypotheses report: {path}")
