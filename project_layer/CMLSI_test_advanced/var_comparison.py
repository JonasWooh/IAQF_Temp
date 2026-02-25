"""
Step 3: Compare Stable CMLSI vs Adaptive CMLSI in VAR
- IRF differences (Depeg → CMLSI, Basis, Premium)
- FEVD differences
"""
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.api import VAR

from ..config import get_config
from ..models.var_runner import load_master, get_var_strategy
from .rolling_pca import compute_adaptive_cmlsi, _get_features_and_data


IDX_DEPEG, IDX_CMLSI, IDX_BASIS, IDX_PREMIUM = 0, 1, 2, 3
VAR_COLS = ["Depeg_bps", "CMLSI", "Basis_bps", "Premium_bps"]


def _ensure_stationarity(var_data: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    """Return (stationary_data, differenced_cols)."""
    from statsmodels.tsa.stattools import adfuller
    cfg = get_config()
    signif = cfg.get("var.adf_signif", 0.05)
    out = var_data.copy()
    differenced = set()
    for col in var_data.columns:
        try:
            _, pvalue, *_ = adfuller(var_data[col].dropna(), autolag="AIC", regression="c")
            if pvalue >= signif:
                out[col] = var_data[col].diff()
                differenced.add(col)
        except Exception:
            pass
    return out.dropna(), differenced


def _irf_to_level(irf_arr: np.ndarray, col_idx: int, diff_cols: set) -> np.ndarray:
    """Cumsum IRF for differenced vars → level response."""
    col = VAR_COLS[col_idx] if col_idx < len(VAR_COLS) else None
    if col and col in diff_cols:
        return np.cumsum(irf_arr)
    return irf_arr


def run_var_with_cmlsi(df: pd.DataFrame, regime: str = "crisis") -> tuple:
    """Run VAR on regime slice, return (results, irf, differenced_cols)."""
    cfg = get_config()
    strategy = get_var_strategy(regime)
    var_data = strategy.get_data_slice(df)
    var_data = var_data[["Depeg_bps", "CMLSI", "Basis_bps", "Premium_bps"]].dropna()
    var_data, diff_cols = _ensure_stationarity(var_data)
    if len(var_data) < 100:
        raise ValueError(f"Insufficient samples: {len(var_data)}")
    model = VAR(var_data)
    lag_res = model.select_order(maxlags=20)
    lag = lag_res.aic
    results = model.fit(maxlags=lag, ic="aic")
    periods = cfg.get("var.irf_periods", 60)
    irf = results.irf(periods=periods)
    return results, irf, diff_cols


def run_stable_vs_adaptive_comparison(out_dir: Path | None = None) -> dict:
    """
    Run VAR with Stable CMLSI (original) vs Adaptive CMLSI.
    Return dict with IRF and FEVD for both.
    """
    cfg = get_config()
    df = load_master()
    features, _ = _get_features_and_data()
    cmlsi_adaptive = compute_adaptive_cmlsi(df, features, window_minutes=1440, n_pcs=3)

    # Align adaptive to df index
    cmlsi_adaptive = cmlsi_adaptive.reindex(df.index).ffill(limit=5).bfill(limit=5)

    df_stable = df.copy()
    df_adaptive = df.copy()
    df_adaptive["CMLSI"] = cmlsi_adaptive

    regime = "crisis"
    try:
        res_stable, irf_stable, diff_cols = run_var_with_cmlsi(df_stable, regime)
        res_adaptive, irf_adaptive, _ = run_var_with_cmlsi(df_adaptive, regime)
    except Exception as e:
        print(f"  [WARN] VAR comparison failed: {e}")
        return {}

    periods = cfg.get("var.irf_periods", 60)
    fevd_s = res_stable.fevd(periods=periods)
    fevd_a = res_adaptive.fevd(periods=periods)

    def _level_irf(irf_obj, idx):
        raw = irf_obj.orth_irfs[:, idx, IDX_DEPEG]
        return _irf_to_level(raw, idx, diff_cols)

    comparison = {
        "regime": regime,
        "differenced_cols": diff_cols,
        "irf_stable": {
            "depeg_cmlsi": _level_irf(irf_stable, IDX_CMLSI),
            "depeg_basis": _level_irf(irf_stable, IDX_BASIS),
            "depeg_premium": _level_irf(irf_stable, IDX_PREMIUM),
        },
        "irf_adaptive": {
            "depeg_cmlsi": _level_irf(irf_adaptive, IDX_CMLSI),
            "depeg_basis": _level_irf(irf_adaptive, IDX_BASIS),
            "depeg_premium": _level_irf(irf_adaptive, IDX_PREMIUM),
        },
        "fevd_stable": {
            "cmlsi": fevd_s.decomp[IDX_CMLSI, :, IDX_DEPEG],
            "basis": fevd_s.decomp[IDX_BASIS, :, IDX_DEPEG],
            "premium": fevd_s.decomp[IDX_PREMIUM, :, IDX_DEPEG],
        },
        "fevd_adaptive": {
            "cmlsi": fevd_a.decomp[IDX_CMLSI, :, IDX_DEPEG],
            "basis": fevd_a.decomp[IDX_BASIS, :, IDX_DEPEG],
            "premium": fevd_a.decomp[IDX_PREMIUM, :, IDX_DEPEG],
        },
    }

    out_dir = out_dir or cfg.get_output_dir() / "CMLSI_test_advanced"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    # Convert numpy arrays to lists for JSON
    out = {k: v for k, v in comparison.items() if k not in ("irf_stable", "irf_adaptive", "fevd_stable", "fevd_adaptive")}
    out["differenced_cols"] = list(diff_cols)
    out["irf_stable_peaks"] = {k: float(np.abs(v).max()) for k, v in comparison["irf_stable"].items()}
    out["irf_adaptive_peaks"] = {k: float(np.abs(v).max()) for k, v in comparison["irf_adaptive"].items()}
    out["fevd_stable_final"] = {k: float(v[-1]) for k, v in comparison["fevd_stable"].items()}
    out["fevd_adaptive_final"] = {k: float(v[-1]) for k, v in comparison["fevd_adaptive"].items()}
    with open(out_dir / "var_comparison_stable_vs_adaptive.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"  [OK] VAR comparison: {out_dir / 'var_comparison_stable_vs_adaptive.json'}")
    return comparison
