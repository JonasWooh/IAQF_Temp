"""运行 VAR 并生成全部图表 (06, 07, 11, 12, 14, 15, 16, 17, 20, 21)"""
from ..config import get_config
from .var_runner import (
    run_var_pipeline,
    run_var_gas_comparison,
    run_var_institutional_comparison,
    run_nested_model_comparison,
    save_nested_model_report,
)
from .var_figures import (
    plot_implied_micro_irf,
    plot_fragmentation_irf,
    plot_fevd,
    plot_basis_irf,
    plot_regime_comparison,
    plot_triple_regime_irf,
    plot_gas_comparison,
    plot_institutional_comparison,
    plot_nested_model_comparison,
)


def run_var_with_figures() -> None:
    cfg = get_config()
    cfg.ensure_dirs()
    results, irf = run_var_pipeline(regime="crisis")
    periods = cfg.get("var.irf_periods", 60)
    repl = cfg.get("var.irf_ci_repl", 1000)
    signif = cfg.get("var.irf_ci_signif", 0.05)
    seed = cfg.get("var.irf_ci_seed", 42)
    lower, upper = results.irf_errband_mc(
        orth=True, repl=repl, steps=periods, signif=signif, seed=seed
    )
    print("  [OK] Crisis VAR fitted")
    diff_cols = getattr(results, "differenced_cols", set())
    plot_implied_micro_irf(irf, lower, upper, differenced_cols=diff_cols)
    plot_fragmentation_irf(irf, lower, upper, differenced_cols=diff_cols)
    plot_fevd(results)
    plot_basis_irf(irf, lower, upper, differenced_cols=diff_cols)
    plot_regime_comparison(results)
    plot_triple_regime_irf(results)
    # Gas Fee 对比实验 (Baseline vs VARX)
    try:
        from .var_runner import save_var_gas_comparison_report
        comparison = run_var_gas_comparison(regime="crisis")
        plot_gas_comparison(comparison)
        save_var_gas_comparison_report(comparison)
    except FileNotFoundError as e:
        print(f"  [SKIP] Gas fee comparison: {e}")
    try:
        from .var_runner import save_var_institutional_report
        inst = run_var_institutional_comparison(regime="crisis")
        plot_institutional_comparison(inst)
        save_var_institutional_report(inst)
    except Exception as e:
        print(f"  [SKIP] Institutional friction comparison: {e}")
    # Nested model comparison (M0 vs M1 vs M2 vs M3)
    try:
        nested = run_nested_model_comparison(regime="crisis")
        plot_nested_model_comparison(nested)
        save_nested_model_report(nested)
    except Exception as e:
        print(f"  [SKIP] Nested model comparison: {e}")
