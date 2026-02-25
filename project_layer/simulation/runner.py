"""
GENIUS Act 模拟运行器 - 通过 strategy 参数切换情景
"""
import numpy as np
import matplotlib.pyplot as plt

from ..config import get_config
from ..models.var_runner import run_var_pipeline
from .strategies import MagnitudeScalingStrategy, TailTruncationStrategy, DurationDampStrategy


IDX_DEPEG, IDX_CMLSI, IDX_BASIS, IDX_PREMIUM = 0, 1, 2, 3
VAR_COLS = ["Depeg_bps", "CMLSI", "Basis_bps", "Premium_bps"]


def _irf_level(irf_arr: np.ndarray, response_idx: int, differenced_cols: set[str]) -> np.ndarray:
    col = VAR_COLS[response_idx] if response_idx < len(VAR_COLS) else None
    if col and col in differenced_cols:
        return np.cumsum(irf_arr)
    return irf_arr


def get_strategy(strategy_name: str, **kwargs):
    """根据字符串返回情景策略"""
    strategies = {
        "A": lambda: MagnitudeScalingStrategy(c=kwargs.get("c", 0.5)),
        "A_075": lambda: MagnitudeScalingStrategy(c=0.75),
        "A_050": lambda: MagnitudeScalingStrategy(c=0.50),
        "A_020": lambda: MagnitudeScalingStrategy(c=0.20),
        "B": lambda: TailTruncationStrategy(quantile=0.95),
        "C_5": lambda: DurationDampStrategy(tau=5),
        "C_10": lambda: DurationDampStrategy(tau=10),
        "C_20": lambda: DurationDampStrategy(tau=20),
    }
    if strategy_name not in strategies:
        raise ValueError(f"strategy must be one of {list(strategies.keys())}")
    return strategies[strategy_name]()


def run_simulation(strategy_name: str = "A", **kwargs) -> None:
    """
    运行 GENIUS Act 情景模拟
    strategy_name: "A", "A_075", "A_050", "A_020", "B", "C_5", "C_10", "C_20"
    """
    cfg = get_config()
    cfg.ensure_dirs()
    results, irf_obj = run_var_pipeline(regime="crisis")
    base_irf = irf_obj.orth_irfs
    diff_cols = getattr(results, "differenced_cols", set())
    strategy = get_strategy(strategy_name, results=results, **kwargs)
    if strategy_name.startswith("B"):
        strategy.results = results
    n_steps = base_irf.shape[0]
    x = np.arange(n_steps)
    fig_dir = cfg.get_output_dir() / "genius_act_simulation" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    titles = [
        ("Depeg → CMLSI", "CMLSI Response", IDX_CMLSI),
        ("Depeg → Basis", "Basis Response", IDX_BASIS),
        ("Depeg → Premium", "Premium Response (bps)", IDX_PREMIUM),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (title, ylabel, idx) in zip(axes, titles):
        base_line_raw = base_irf[:, idx, IDX_DEPEG]
        cf_line_raw = strategy.transform_irf(base_irf, idx, IDX_DEPEG, results=results)
        base_line = _irf_level(base_line_raw, idx, diff_cols)
        cf_line = _irf_level(cf_line_raw, idx, diff_cols)
        ax.plot(x, base_line, color="#E74C3C", linewidth=2, label="Factual")
        ax.plot(x, cf_line, color="#2ECC71", linewidth=2, label=f"Counterfactual ({strategy_name})")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel("Minutes after Shock")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"GENIUS Act: {strategy.name} (strategy={strategy_name})", fontsize=12, y=1.02)
    plt.tight_layout()
    out = fig_dir / f"genius_act_{strategy_name}.png"
    fig.savefig(out, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out}")
    plt.close()


def run_all_scenarios() -> None:
    """运行 A/B/C 三种情景，生成综合图"""
    cfg = get_config()
    cfg.ensure_dirs()
    results, irf_obj = run_var_pipeline(regime="crisis")
    base_irf = irf_obj.orth_irfs
    diff_cols = getattr(results, "differenced_cols", set())
    n_steps = base_irf.shape[0]
    x = np.arange(n_steps)
    fig_dir = cfg.get_output_dir() / "genius_act_simulation" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    titles = [
        ("Depeg → CMLSI", "CMLSI Response", IDX_CMLSI),
        ("Depeg → Basis", "Basis Response", IDX_BASIS),
        ("Depeg → Premium", "Premium Response (bps)", IDX_PREMIUM),
    ]
    scenarios_a = [(1.0, "#E74C3C"), (0.75, "#F39C12"), (0.5, "#3498DB"), (0.2, "#2ECC71")]
    strat_b = TailTruncationStrategy(results=results, quantile=0.95)
    tau_list = [(5, "#F39C12"), (10, "#3498DB"), (20, "#2ECC71")]
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    for col, (title, ylabel, idx) in enumerate(titles):
        base_line_raw = base_irf[:, idx, IDX_DEPEG]
        for c, color in scenarios_a:
            axes[0, col].plot(
                x,
                _irf_level(base_line_raw * c, idx, diff_cols),
                color=color,
                linewidth=2,
                label=f"c={c:.2f}",
            )
        axes[0, col].axhline(0, color="gray", linestyle="--", alpha=0.7)
        axes[0, col].set_title(title)
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, alpha=0.3)
        cf_b_raw = strat_b.transform_irf(base_irf, idx, IDX_DEPEG, results=results)
        base_line = _irf_level(base_line_raw, idx, diff_cols)
        cf_b = _irf_level(cf_b_raw, idx, diff_cols)
        axes[1, col].plot(x, base_line, color="#E74C3C", linewidth=2, label="Baseline")
        axes[1, col].plot(x, cf_b, color="#2ECC71", linewidth=2, label="95% winsorize")
        axes[1, col].axhline(0, color="gray", linestyle="--", alpha=0.7)
        axes[1, col].set_title(title)
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(True, alpha=0.3)
        axes[2, col].plot(x, base_line, color="#E74C3C", linewidth=2, label="Baseline")
        for tau, color in tau_list:
            strat_c = DurationDampStrategy(tau=tau)
            cf_c_raw = strat_c.transform_irf(base_irf, idx, IDX_DEPEG)
            cf_c = _irf_level(cf_c_raw, idx, diff_cols)
            axes[2, col].plot(x, cf_c, color=color, linewidth=2, label=f"τ={tau}")
        axes[2, col].axhline(0, color="gray", linestyle="--", alpha=0.7)
        axes[2, col].set_xlabel("Minutes after Shock")
        axes[2, col].set_title(title)
        axes[2, col].legend(fontsize=8)
        axes[2, col].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("A: Magnitude")
    axes[1, 0].set_ylabel("B: Tail trunc")
    axes[2, 0].set_ylabel("C: Duration")
    fig.suptitle("GENIUS Act: Three Scenarios | A: Magnitude | B: Winsorize 95% | C: Duration damp", fontsize=13, y=1.01)
    plt.tight_layout()
    out = fig_dir / "19_genius_act_three_scenarios.png"
    fig.savefig(out, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out}")
    plt.close()
    report = cfg.get_output_dir() / "genius_act_simulation" / "simulation_report.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "GENIUS Act Counterfactual Simulation\n"
        "A: Magnitude scaling (c=1,0.75,0.5,0.2)\n"
        "B: Tail truncation (95% winsorize)\n"
        "C: Duration damp (τ=5,10,20)\n",
        encoding="utf-8"
    )
    # 19a: 仅情景 A (Magnitude)
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (title, ylabel, idx) in zip(axes2, titles):
        base_line_raw = base_irf[:, idx, IDX_DEPEG]
        for c, color in scenarios_a:
            ax.plot(x, _irf_level(base_line_raw * c, idx, diff_cols), color=color, linewidth=2, label=f"c={c:.2f}")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel("Minutes after Shock")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig2.suptitle("Scenario A Only: Magnitude Scaling (c × IRF)", fontsize=12, y=1.02)
    plt.tight_layout()
    out19a = fig_dir / "19a_scenario_A_magnitude.png"
    fig2.savefig(out19a, dpi=cfg.get("visualization.figure_dpi", 150), bbox_inches="tight")
    print(f"  [OK] {out19a.name}")
    plt.close()
