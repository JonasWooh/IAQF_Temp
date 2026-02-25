"""
CMLSI Advanced Robustness — Full Pipeline
Step 1: Prove structure changed
Step 2: Stable vs Adaptive CMLSI
Step 3: Compare in VAR (IRF/FEVD)
Step 4: Economic interpretation (report)
"""
from pathlib import Path

from ..config import get_config

from .structure_tests import run_structure_tests
from .rolling_pca import run_adaptive_cmlsi
from .var_comparison import run_stable_vs_adaptive_comparison
from .visualizations import (
    plot_structure_tests,
    plot_rolling_pca,
    plot_stable_vs_adaptive_cmlsi,
    plot_var_comparison,
    _get_features_and_data,
)


def run_full_cmlsi_robustness() -> None:
    """
    Run full CMLSI robustness pipeline:
    1. Structure tests (covariance, eigenvalue, subspace angle)
    2. Rolling PCA + Adaptive CMLSI
    3. VAR comparison (Stable vs Adaptive)
    4. All visualizations
    """
    cfg = get_config()
    cfg.ensure_dirs()
    out_dir = cfg.get_output_dir() / "CMLSI_test_advanced"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("CMLSI Advanced Robustness Tests")
    print("=" * 70)

    # Step 1: Structure tests
    print("\n--- Step 1: Structure Change Tests ---")
    results = run_structure_tests(out_dir)
    plot_structure_tests(results, out_dir)

    # Step 2: Rolling PCA + Adaptive CMLSI
    print("\n--- Step 2: Rolling PCA & Adaptive CMLSI ---")
    cmlsi_adaptive, rolling_metrics = run_adaptive_cmlsi(out_dir)
    _, df = _get_features_and_data()
    plot_rolling_pca(rolling_metrics, out_dir)
    plot_stable_vs_adaptive_cmlsi(df, cmlsi_adaptive, out_dir)

    # Step 3: VAR comparison
    print("\n--- Step 3: VAR Comparison (Stable vs Adaptive) ---")
    comparison = run_stable_vs_adaptive_comparison(out_dir)
    plot_var_comparison(comparison, out_dir)

    # Step 4: Economic interpretation report
    print("\n--- Step 4: Economic Interpretation Report ---")
    _write_interpretation_report(out_dir, results, comparison)

    print("\n" + "=" * 70)
    print("√ CMLSI Advanced Robustness complete.")
    print(f"  Output: {out_dir}")
    print("=" * 70)


def _write_interpretation_report(out_dir: Path, results: dict, comparison: dict) -> None:
    """Write economic interpretation report."""
    lines = [
        "=" * 70,
        "CMLSI Robustness: Economic Interpretation",
        "=" * 70,
        "",
        "Step 1 — Structure Change:",
        "  • Frobenius norm ||Σ_N - Σ_C||_F: Large → regime shift exists.",
        "  • Eigenvalue ratio: Crisis/Normal ≠ 1 → variance structure changed.",
        "  • Subspace angle: Large θ → principal direction rotated.",
        "",
        "Step 2 — Stable vs Adaptive CMLSI:",
        "  • Stable CMLSI: Normal-period fit, measures 'shock magnitude' (OOS).",
        "  • Adaptive CMLSI: Rolling PCA, measures 'structural collapse' (regime-adaptive).",
        "",
        "Step 3 — VAR Comparison:",
        "  • If IRF/FEVD differ: CMLSI construction matters for传导结论.",
        "  • If similar: Robust to index construction.",
        "",
    ]
    if comparison:
        import numpy as np
        irf_s = comparison.get("irf_stable", {})
        irf_a = comparison.get("irf_adaptive", {})
        peak_s = float(np.abs(irf_s.get("depeg_premium", np.array([0]))).max()) if irf_s else 0
        peak_a = float(np.abs(irf_a.get("depeg_premium", np.array([0]))).max()) if irf_a else 0
        lines.extend([
            "  IRF Peak |Depeg→Premium|:",
            f"    Stable:   {peak_s:.4f} bps",
            f"    Adaptive: {peak_a:.4f} bps",
            "",
        ])
    path = out_dir / "economic_interpretation_report.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] {path.name}")
