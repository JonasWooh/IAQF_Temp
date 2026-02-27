import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

from project_layer.config import get_config
from project_layer.models.var_runner import load_master
from project_layer.features.utils import make_prefix


PAIR_DISPLAY = {
    "USD": "BTC/USD",
    "USDC": "BTC/USDC",
    "USDT": "BTC/USDT",
}


def _resolve_fragmentation_col(df: pd.DataFrame, exchange: str, pair: str, metric: str) -> str | None:
    btc_pair = f"BTC_{pair}"
    candidates = [
        f"{make_prefix(exchange, btc_pair)}_{metric}",
        f"{exchange}_{btc_pair}_{metric}",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None

def analyze_dimension_1_static_differences(out_dir: Path | None = None) -> dict:
    """
    Dimension 1: Static test of systematic liquidity differences during the normal period
    Use Kruskal-Wallis H test and Mann-Whitney U test to compare USD, USDC, USDT
    """
    cfg = get_config()
    df = load_master()
    if df is None:
        raise ValueError("Master table not found.")
        
    out_dir = out_dir or cfg.get_output_dir() / "fragmentation_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Extract data for the Normal phase
    pn = cfg.get("phases.normal.start", "2023-03-01")
    pe = cfg.get("phases.normal.end", "2023-03-09")
    df_normal = df.loc[pn:pe].copy()
    
    exchange = cfg.get("exchanges.btc.exchanges", ["BINANCEUS"])[0]
    metrics = ["rel_spread", "log_depth"]
    pairs = ["USD", "USDC", "USDT"]
    results = {}
    
    report_lines = [
        "==================================================",
        "Part1: Static Liquidity Differences (Normal Regime)",
        f"Period: {pn} to {pe}",
        "==================================================\n"
    ]

    for metric in metrics:
        report_lines.append(f"--- Analyzing Metric: {metric.upper()} ---")
        
        # Construct column names and extract non-null data
        cols = {pair: _resolve_fragmentation_col(df_normal, exchange, pair, metric) for pair in pairs}
        missing_pairs = [pair for pair in pairs if cols[pair] is None]
        if missing_pairs:
            report_lines.append(f"  [WARN] Missing columns for {metric}: {missing_pairs}. Skipping metric.\n")
            continue
        data = {pair: df_normal[cols[pair]].dropna().values for pair in pairs}
        
        # Ensure all three groups have data
        if any(len(v) < 50 for v in data.values()):
            report_lines.append("  [WARN] Insufficient data for one or more pairs.\n")
            continue
            
        # Record median (more resistant to extreme outliers than relative mean)
        medians = {pair: np.median(data[pair]) for pair in pairs}
        for pair in pairs:
            report_lines.append(f"  {PAIR_DISPLAY[pair]} Median {metric}: {medians[pair]:.6f}")
            
        # 2. Global Kruskal-Wallis test
        stat_kw, p_kw = stats.kruskal(data["USD"], data["USDC"], data["USDT"])
        report_lines.append(f"  [Global] Kruskal-Wallis P-value: {p_kw:.4e}")
        if p_kw < 0.05:
            report_lines.append("  -> Conclusion: Significant differences exist across the three pairs.")
        else:
            report_lines.append("  -> Conclusion: NO significant differences across the pairs.")
            
        # 3. Post-hoc pairwise test (Mann-Whitney U)
        report_lines.append("  [Pairwise] Mann-Whitney U P-values:")
        pw_usd_usdc = stats.mannwhitneyu(data["USD"], data["USDC"], alternative='two-sided')[1]
        pw_usdc_usdt = stats.mannwhitneyu(data["USDC"], data["USDT"], alternative='two-sided')[1]
        pw_usd_usdt = stats.mannwhitneyu(data["USD"], data["USDT"], alternative='two-sided')[1]
        
        report_lines.append(f"     BTC/USD vs BTC/USDC: {pw_usd_usdc:.4e}")
        report_lines.append(f"     BTC/USDC vs BTC/USDT: {pw_usdc_usdt:.4e}")
        report_lines.append(f"     BTC/USD vs BTC/USDT: {pw_usd_usdt:.4e}\n")
        
        results[metric] = {
            "medians": medians,
            "p_kw": p_kw,
            "p_pairwise": {"usd_usdc": pw_usd_usdc, "usdc_usdt": pw_usdc_usdt, "usd_usdt": pw_usd_usdt}
        }
        
        # 4. Draw boxplot to visually show differences
        _plot_static_boxplot(df_normal, cols, metric, out_dir, dpi=cfg.get("visualization.figure_dpi", 150))
        
    # Save text report
    report_path = out_dir / "dim1_static_differences_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  [OK] Dimension 1 Report saved to: {report_path}")
    
    return results

def _plot_static_boxplot(df_normal: pd.DataFrame, cols: dict, metric: str, out_dir: Path, dpi: int):
    """Draw normal boxplot, removing extreme outliers to ensure visualization effect"""
    plot_data = pd.DataFrame({PAIR_DISPLAY[pair]: df_normal[col] for pair, col in cols.items()})
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=plot_data, showfliers=False, ax=ax, palette="Set2")
    
    ax.set_title(f"Baseline Distribution of {metric.upper()} in Normal Regime")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("Quote Currency")
    
    # Add grid lines for easy comparison
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    out_path = out_dir / f"dim1_boxplot_{metric}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
def analyze_dimension_2_dynamic_fragmentation(out_dir: Path | None = None) -> dict:
    """
    Dimension 2: Dynamic evolution of liquidity fragmentation during the crisis period (DiD logic)
    Compare Normal and Crisis regimes to calculate liquidity deterioration magnitude (Delta) for each pair
    """
    cfg = get_config()
    df = load_master()
    if df is None:
        raise ValueError("Master table not found.")
        
    out_dir = out_dir or cfg.get_output_dir() / "fragmentation_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Extract data for the Normal and Crisis regimes
    pn = cfg.get("phases.normal.start", "2023-03-01")
    pe_n = cfg.get("phases.normal.end", "2023-03-09")
    ps = cfg.get("phases.stressed.start", "2023-03-10")
    pe_s = cfg.get("phases.stressed.end", "2023-03-13")
    
    df_normal = df.loc[pn:pe_n]
    df_crisis = df.loc[ps:pe_s]
    
    exchange = cfg.get("exchanges.btc.exchanges", ["BINANCEUS"])[0]
    metrics = ["rel_spread", "log_depth"]
    pairs = ["USD", "USDC", "USDT"]
    results = {}
    
    report_lines = [
        "==================================================",
        "Part2: Dynamic Liquidity Fragmentation (DiD)",
        f"Normal: {pn} to {pe_n}",
        f"Crisis: {ps} to {pe_s}",
        "==================================================\n"
    ]
    
    for metric in metrics:
        report_lines.append(f"--- Analyzing Metric: {metric.upper()} ---")
        metric_results = {}
        
        for pair in pairs:
            col = _resolve_fragmentation_col(df_normal, exchange, pair, metric)
            if col is None:
                report_lines.append(f"  [WARN] Missing column for {pair} {metric}; skipping pair.\n")
                continue
            
            # Use median values to compute Delta, reducing sensitivity to extreme spikes
            norm_vals = df_normal[col].dropna()
            cris_vals = df_crisis[col].dropna()
            
            med_norm = norm_vals.median() if not norm_vals.empty else np.nan
            med_cris = cris_vals.median() if not cris_vals.empty else np.nan
            delta = med_cris - med_norm
            
            metric_results[pair] = {
                "normal_median": med_norm,
                "crisis_median": med_cris,
                "delta": delta
            }
            
            report_lines.append(f"  {PAIR_DISPLAY[pair]}:")
            report_lines.append(f"    Normal Median : {med_norm:.6f}")
            report_lines.append(f"    Crisis Median : {med_cris:.6f}")
            report_lines.append(f"    Delta (C - N) : {delta:.6f}\n")
            
        results[metric] = metric_results
        
        # 2. Plot DiD shock-magnitude bar chart
        if metric_results:
            _plot_did_barchart(metric_results, metric, out_dir, dpi=cfg.get("visualization.figure_dpi", 150))
        else:
            report_lines.append(f"  [WARN] No valid data for {metric}; skipping plot.\n")
        
    # 3. Compute relative deterioration multiple (how many times USDC deterioration exceeds USDT)
    if "rel_spread" in results and "USDC" in results["rel_spread"] and "USDT" in results["rel_spread"]:
        delta_usdc = results["rel_spread"]["USDC"]["delta"]
        delta_usdt = results["rel_spread"]["USDT"]["delta"]
        if delta_usdt > 0:
            ratio = delta_usdc / delta_usdt
            report_lines.append(f"--- Cross-Sectional Fragmentation Ratio ---")
            report_lines.append(f"  BTC/USDC Spread Deterioration is {ratio:.2f}x compared to BTC/USDT.")
        
    report_path = out_dir / "dim2_dynamic_fragmentation_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  [OK] Dimension 2 Report saved to: {report_path}")
    
    return results

def _plot_did_barchart(metric_results: dict, metric: str, out_dir: Path, dpi: int):
    """Plot shock-magnitude bar chart, highlighting affected vs safe-haven channels"""
    pairs = list(metric_results.keys())
    deltas = [metric_results[p]["delta"] for p in pairs]
    pair_labels = [PAIR_DISPLAY.get(p, p) for p in pairs]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Color coding: highly affected USD/USDC in red, USDT in green
    colors = ['#E74C3C' if p in ['USD', 'USDC'] else '#2ECC71' for p in pairs] 
    bars = ax.bar(pair_labels, deltas, color=colors, alpha=0.8)
    
    # Titles and labels
    title_metric = "Relative Spread" if metric == "rel_spread" else "Log Depth"
    ax.set_title(f"Liquidity Shock Magnitude ($\Delta$ {title_metric}) \n Crisis vs Normal Regime")
    ax.set_ylabel(f"$\Delta$ {title_metric} (Crisis Median - Normal Median)")
    ax.set_xlabel("Quote Currency")
    
    # Add value labels on bars
    ax.bar_label(bars, fmt='%.4f', padding=3)
    ax.axhline(0, color='black', linewidth=1)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    out_path = out_dir / f"dim2_did_barchart_{metric}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
def analyze_dimension_3_correlation_breakdown(out_dir: Path | None = None) -> dict:
    """
    Dimension 3: Cross-market liquidity comovement (correlation breakdown)
    Compare Spearman rank-correlation matrices between Normal and Crisis regimes
    """
    cfg = get_config()
    df = load_master()
    if df is None:
        raise ValueError("Master table not found.")
        
    out_dir = out_dir or cfg.get_output_dir() / "fragmentation_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Extract data for the Normal and Crisis regimes
    pn = cfg.get("phases.normal.start", "2023-03-01")
    pe_n = cfg.get("phases.normal.end", "2023-03-09")
    ps = cfg.get("phases.stressed.start", "2023-03-10")
    pe_s = cfg.get("phases.stressed.end", "2023-03-13")
    
    df_normal = df.loc[pn:pe_n]
    df_crisis = df.loc[ps:pe_s]
    
    exchange = cfg.get("exchanges.btc.exchanges", ["BINANCEUS"])[0]
    metrics = ["rel_spread", "log_depth"]
    pairs = ["USD", "USDC", "USDT"]
    results = {}
    
    report_lines = [
        "==================================================",
        "Dimension 3: Cross-Market Correlation Breakdown",
        f"Normal: {pn} to {pe_n}",
        f"Crisis: {ps} to {pe_s}",
        "==================================================\n"
    ]
    
    for metric in metrics:
        report_lines.append(f"--- Analyzing Metric: {metric.upper()} ---")
        
        # Extract the three columns corresponding to the current metric
        cols = [_resolve_fragmentation_col(df_normal, exchange, pair, metric) for pair in pairs]
        if any(c is None for c in cols):
            missing_pairs = [pair for pair, col in zip(pairs, cols) if col is None]
            report_lines.append(f"  [WARN] Missing columns for {metric}: {missing_pairs}. Skipping metric.\n")
            continue
        
        # Use Spearman rank correlation (robust to non-normality and extreme spikes)
        # use .diff() to focus on comovement of changes
        corr_normal = df_normal[cols].diff().dropna().corr(method='spearman')
        corr_crisis = df_crisis[cols].diff().dropna().corr(method='spearman')
        
        # Simplify column names for cleaner report output
        clean_cols = [PAIR_DISPLAY[p] for p in pairs]
        corr_normal.columns = clean_cols
        corr_normal.index = clean_cols
        corr_crisis.columns = clean_cols
        corr_crisis.index = clean_cols
        
        report_lines.append("  [Normal Regime Spearman Correlation]")
        report_lines.append(corr_normal.to_string())
        report_lines.append("\n  [Crisis Regime Spearman Correlation]")
        report_lines.append(corr_crisis.to_string())
        
        # Compute core correlation decay (especially between stressed USDC and safer USDT)
        if "BTC/USDC" in clean_cols and "BTC/USDT" in clean_cols:
            corr_n_usdc_usdt = corr_normal.loc["BTC/USDC", "BTC/USDT"]
            corr_c_usdc_usdt = corr_crisis.loc["BTC/USDC", "BTC/USDT"]
            drop_magnitude = corr_n_usdc_usdt - corr_c_usdc_usdt
            report_lines.append("\n  [Key Finding: BTC/USDC-BTC/USDT Comovement]")
            report_lines.append(f"    Normal Correlation : {corr_n_usdc_usdt:.4f}")
            report_lines.append(f"    Crisis Correlation : {corr_c_usdc_usdt:.4f}")
            report_lines.append(f"    Correlation Drop   : {drop_magnitude:.4f}\n")
            
        results[metric] = {
            "normal": corr_normal.to_dict(),
            "crisis": corr_crisis.to_dict()
        }
        
        # 2. Plot side-by-side heatmaps for visual comparison
        _plot_correlation_heatmap(corr_normal, corr_crisis, metric, out_dir, dpi=cfg.get("visualization.figure_dpi", 150))
        
    report_path = out_dir / "dim3_correlation_breakdown_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  [OK] Dimension 3 Report saved to: {report_path}")
    
    return results

def _plot_correlation_heatmap(corr_normal: pd.DataFrame, corr_crisis: pd.DataFrame, metric: str, out_dir: Path, dpi: int):
    """Plot side-by-side correlation heatmaps for Normal and Crisis regimes"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Use a unified color scale (0 to 1) to ensure comparability
    vmin, vmax = 0, 1
    cmap = "YlGnBu" # Blue-green palette; darker color indicates stronger correlation
    
    sns.heatmap(corr_normal, annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax, 
                ax=axes[0], cbar=False, square=True)
    axes[0].set_title("Normal Regime Comovement")
    
    sns.heatmap(corr_crisis, annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax, 
                ax=axes[1], cbar=True, square=True)
    axes[1].set_title("Crisis Regime Comovement")
    
    title_metric = "Relative Spread" if metric == "rel_spread" else "Log Depth"
    fig.suptitle(f"Cross-Market Correlation Breakdown: {title_metric}", fontsize=14)
    
    plt.tight_layout()
    out_path = out_dir / f"dim3_correlation_heatmap_{metric}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()