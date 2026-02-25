"""
DiagnosticsObserver - 监听 VAR 输出，生成报告与图表
"""
from pathlib import Path
from typing import Any

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox

from ..config import get_config
from ..models.var_runner import load_master, get_var_strategy


class DiagnosticsObserver:
    """Observer：监听模型，输出诊断报告和图表"""

    def __init__(self, results, regime: str = "crisis"):
        self.results = results
        self.regime = regime
        self.cfg = get_config()

    def generate_report(self) -> str:
        """生成 var_diagnostics.txt"""
        lines = [
            "=" * 70,
            "VAR Model Diagnostics (Stability + Residual White Noise)",
            f"IAQF 2026 - Regime: {self.regime}",
            "=" * 70,
            "",
        ]
        stable = self.results.is_stable()
        lines.append(f"Stability (roots in unit circle): {'✓ Yes' if stable else '✗ No'}")
        resid = self.results.resid
        lags_use = min(10, resid.shape[0] // 5)
        for col in resid.columns:
            df_lb = acorr_ljungbox(resid[col].dropna(), lags=[lags_use])
            pval = float(df_lb["lb_pvalue"].iloc[-1])
            ok = "✓" if pval > 0.05 else "✗"
            lines.append(f"  {col}: Ljung-Box(lag={lags_use}) p={pval:.4f} {ok}")
        return "\n".join(lines)

    def save_report(self, out_path: Path | None = None):
        path = out_path or self.cfg.get_output_dir() / "var_diagnostics.txt"
        path.write_text(self.generate_report(), encoding="utf-8")
        print(f"  [OK] Diagnostics: {path}")

    def run_all_regimes_report(self):
        """对 Normal / Crisis / Recovery 三期生成汇总报告"""
        cfg = self.cfg
        regimes = [
            ("normal", "Normal (Mar 1–9)"),
            ("crisis", "Crisis (Mar 10–13)"),
            ("recovery", "Recovery (Mar 14–21)"),
        ]
        lines = [
            "=" * 70,
            "VAR Model Diagnostics (All Regimes)",
            "IAQF 2026 - Project Layer",
            "=" * 70,
            "",
        ]
        df = load_master()
        for regime, label in regimes:
            try:
                strategy = get_var_strategy(regime)
                var_data = strategy.prepare_data(df)
                var_data, diff_cols = strategy._ensure_stationarity(var_data)
                if len(var_data) < 100:
                    lines.append(f"[{label}] Skipped: insufficient sample ({len(var_data)})")
                    continue
                results = strategy.fit(var_data, ensure_stationary=False)
                stable = results.is_stable()
                lines.append(f"[{label}] Stability: {'✓ Yes' if stable else '✗ No'}")
                if diff_cols:
                    lines.append(f"  Differenced (ADF non-stationary): {', '.join(sorted(diff_cols))}")
                lags_use = min(10, results.resid.shape[0] // 5)
                for col in results.resid.columns:
                    df_lb = acorr_ljungbox(results.resid[col].dropna(), lags=[lags_use])
                    pval = float(df_lb["lb_pvalue"].iloc[-1])
                    ok = "✓" if pval > 0.05 else "✗"
                    lines.append(f"  {col}: p={pval:.4f} {ok}")
                lines.append("")
            except Exception as e:
                lines.append(f"[{label}] Error: {e}")
                lines.append("")
        out = cfg.get_output_dir() / "var_diagnostics.txt"
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"  [OK] All-regime diagnostics: {out}")
