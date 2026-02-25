"""
Step 1: Structure Change Tests
- Covariance matrix Frobenius norm: ||Σ_N - Σ_C||_F
- Eigenvalue distribution comparison
- PCA subspace angle: θ = arccos(|v_N^T v_C|)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ..config import get_config
from ..models.var_runner import load_master


def _get_features_and_data() -> tuple[list[str], pd.DataFrame]:
    """Load features from cmlsi_meta and master data."""
    cfg = get_config()
    meta_path = cfg.get_output_dir() / cfg.get("paths.cmlsi_meta", "cmlsi_meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"cmlsi_meta.json not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    features = meta.get("features", [])
    df = load_master()
    if df is None or not features:
        raise ValueError("Master or features not found")
    return features, df


def covariance_frobenius_norm(
    df: pd.DataFrame,
    features: list[str],
    normal_start: str,
    normal_end: str,
    crisis_start: str,
    crisis_end: str,
) -> dict:
    """
    ||Σ_N - Σ_C||_F: Frobenius norm of covariance matrix difference.
    Large value → regime shift exists.
    """
    X_n = df.loc[normal_start:normal_end][features].dropna()
    X_c = df.loc[crisis_start:crisis_end][features].dropna()
    if len(X_n) < 50 or len(X_c) < 50:
        return {"frobenius_norm": np.nan, "n_normal": len(X_n), "n_crisis": len(X_c)}
    cov_n = X_n.cov().values
    cov_c = X_c.cov().values
    diff = cov_n - cov_c
    frob = np.sqrt(np.sum(diff ** 2))
    return {
        "frobenius_norm": float(frob),
        "n_normal": len(X_n),
        "n_crisis": len(X_c),
        "cov_norm_normal": float(np.sqrt(np.sum(cov_n ** 2))),
        "cov_norm_crisis": float(np.sqrt(np.sum(cov_c ** 2))),
    }


def eigenvalue_comparison(
    df: pd.DataFrame,
    features: list[str],
    normal_start: str,
    normal_end: str,
    crisis_start: str,
    crisis_end: str,
) -> dict:
    """
    Compare eigenvalue distributions between Normal and Crisis.
    eigenvalue_ratio_collapse: λ1_C / λ1_N, etc.
    """
    X_n = df.loc[normal_start:normal_end][features].dropna()
    X_c = df.loc[crisis_start:crisis_end][features].dropna()
    if len(X_n) < 50 or len(X_c) < 50:
        return {}
    scaler_n = StandardScaler()
    scaler_c = StandardScaler()
    Z_n = scaler_n.fit_transform(X_n)
    Z_c = scaler_c.fit_transform(X_c)
    pca_n = PCA(n_components=None).fit(Z_n)
    pca_c = PCA(n_components=None).fit(Z_c)
    ev_n = pca_n.explained_variance_
    ev_c = pca_c.explained_variance_
    n_comp = min(len(ev_n), len(ev_c), 7)
    ratios = [float(ev_c[i] / ev_n[i]) if ev_n[i] > 1e-10 else np.nan for i in range(n_comp)]
    return {
        "eigenvalue_ratio_crisis_over_normal": ratios,
        "evr_normal": pca_n.explained_variance_ratio_.tolist(),
        "evr_crisis": pca_c.explained_variance_ratio_.tolist(),
        "pc1_evr_normal": float(pca_n.explained_variance_ratio_[0]),
        "pc1_evr_crisis": float(pca_c.explained_variance_ratio_[0]),
    }


def rmt_marchenko_pastur_analysis(
    df: pd.DataFrame,
    features: list[str],
    normal_start: str,
    normal_end: str,
    sigma_sq: float = 1.0,
) -> dict:
    """
    Random Matrix Theory (RMT): Marchenko-Pastur 噪声上界。
    λ_max^noise = σ² × (1 + √q)², q = N/T.
    标准化后 σ²=1。λ > λ_max^noise → 信号；λ ≤ λ_max^noise → 噪声。
    """
    X = df.loc[normal_start:normal_end][features].dropna()
    N = len(features)
    T = len(X)
    if T < 50:
        return {}
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    pca = PCA(n_components=None).fit(Z)
    eigenvalues = pca.explained_variance_.tolist()
    q = N / T
    lambda_max_noise = sigma_sq * (1 + np.sqrt(q)) ** 2
    signal_count = sum(1 for ev in eigenvalues if ev > lambda_max_noise)
    judgments = [
        {"pc": i + 1, "lambda": float(ev), "is_signal": bool(ev > lambda_max_noise)}
        for i, ev in enumerate(eigenvalues)
    ]
    return {
        "N": N,
        "T": T,
        "q": float(q),
        "sigma_sq": sigma_sq,
        "lambda_max_noise": float(lambda_max_noise),
        "eigenvalues": [float(ev) for ev in eigenvalues],
        "signal_count": signal_count,
        "judgments": judgments,
    }


def pca_subspace_angle(
    df: pd.DataFrame,
    features: list[str],
    normal_start: str,
    normal_end: str,
    crisis_start: str,
    crisis_end: str,
    n_components: int = 3,
) -> dict:
    """
    θ = arccos(|v_N^T v_C|) for first principal component.
    Large angle → principal direction rotated (structure changed).
    """
    X_n = df.loc[normal_start:normal_end][features].dropna()
    X_c = df.loc[crisis_start:crisis_end][features].dropna()
    if len(X_n) < 50 or len(X_c) < 50:
        return {}
    scaler = StandardScaler()
    Z_n = scaler.fit_transform(X_n)
    Z_c = scaler.transform(X_c)
    pca_n = PCA(n_components=n_components).fit(Z_n)
    pca_c = PCA(n_components=n_components).fit(Z_c)
    angles_deg = []
    for i in range(n_components):
        v_n = pca_n.components_[i]
        v_c = pca_c.components_[i]
        cos_sim = np.abs(np.dot(v_n, v_c) / (np.linalg.norm(v_n) * np.linalg.norm(v_c) + 1e-12))
        cos_sim = np.clip(cos_sim, 0, 1)
        angle_rad = np.arccos(cos_sim)
        angles_deg.append(float(np.degrees(angle_rad)))
    return {
        "subspace_angle_deg": angles_deg,
        "pc1_angle_deg": angles_deg[0],
        "loadings_normal_pc1": pca_n.components_[0].tolist(),
        "loadings_crisis_pc1": pca_c.components_[0].tolist(),
    }


def run_structure_tests(out_dir: Path | None = None) -> dict:
    """Run all structure tests and save report + figures."""
    cfg = get_config()
    features, df = _get_features_and_data()
    pn = cfg.get("phases.normal.start", "2023-03-01")
    pe_n = cfg.get("phases.normal.end", "2023-03-09")
    ps = cfg.get("phases.stressed.start", "2023-03-10")
    pe_s = cfg.get("phases.stressed.end", "2023-03-13")

    results = {}
    results["frobenius"] = covariance_frobenius_norm(df, features, pn, pe_n, ps, pe_s)
    results["eigenvalue"] = eigenvalue_comparison(df, features, pn, pe_n, ps, pe_s)
    results["subspace_angle"] = pca_subspace_angle(df, features, pn, pe_n, ps, pe_s)
    results["rmt"] = rmt_marchenko_pastur_analysis(df, features, pn, pe_n)

    out_dir = out_dir or cfg.get_output_dir() / "CMLSI_test_advanced"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "structure_tests_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Text report
    lines = [
        "=" * 70,
        "CMLSI Structure Change Tests",
        "=" * 70,
        "",
        "[1] Covariance Frobenius Norm ||Σ_N - Σ_C||_F",
        f"  Frobenius norm: {results['frobenius'].get('frobenius_norm', 'N/A')}",
        f"  n_normal={results['frobenius'].get('n_normal')}, n_crisis={results['frobenius'].get('n_crisis')}",
        "  → Large value indicates regime shift.",
        "",
        "[2] Eigenvalue Ratio (Crisis / Normal)",
        f"  PC1 EVR Normal: {results['eigenvalue'].get('pc1_evr_normal', 'N/A'):.4f}",
        f"  PC1 EVR Crisis: {results['eigenvalue'].get('pc1_evr_crisis', 'N/A'):.4f}",
        f"  Eigenvalue ratios: {results['eigenvalue'].get('eigenvalue_ratio_crisis_over_normal', [])}",
        "",
        "[3] PCA Subspace Angle (degrees)",
        f"  PC1 angle: {results['subspace_angle'].get('pc1_angle_deg', 'N/A'):.2f}°",
        f"  All PCs: {results['subspace_angle'].get('subspace_angle_deg', [])}",
        "  → Large angle indicates principal direction rotated.",
        "",
        "[4] RMT (Marchenko-Pastur) Signal vs Noise",
    ]
    rmt = results.get("rmt", {})
    if rmt:
        lines.extend([
            f"  N={rmt.get('N')}, T={rmt.get('T'):,}, q=N/T={rmt.get('q', 0):.6f}",
            f"  λ_max^noise = σ²(1+√q)² = {rmt.get('lambda_max_noise', 0):.4f}",
            f"  Signal PCs (λ > λ_max^noise): {rmt.get('signal_count', 0)}",
            "",
        ])
        for j in rmt.get("judgments", []):
            tag = "信号" if j.get("is_signal") else "噪声"
            lines.append(f"  PC{j.get('pc')}: λ={j.get('lambda', 0):.3f}  {'>' if j.get('is_signal') else '≤'} {rmt.get('lambda_max_noise', 0):.4f}  → {tag}")
        lines.append("")
    else:
        lines.append("  (RMT skipped)")
        lines.append("")
    txt_path = out_dir / "structure_tests_report.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] Structure tests: {txt_path}")

    # RMT 详细报告
    if results.get("rmt"):
        _write_rmt_report(out_dir, results["rmt"])
        print(f"  [OK] RMT report: {out_dir / 'pca_rmt_report.txt'}")

    return results


def _write_rmt_report(out_dir: Path, rmt: dict) -> None:
    """写入 RMT 详细说明报告。"""
    lam_max = rmt.get("lambda_max_noise", 0)
    lines = [
        "=" * 70,
        "PCA RMT (Random Matrix Theory) — Marchenko-Pastur 分析",
        "=" * 70,
        "",
        "一、核心问题",
        "  PCA 得到的主成分中，哪些是真实信号，哪些只是随机噪声？",
        "  RMT 用来给「纯噪声」的特征值设定一个上界，超过此上界的成分更可能是信号。",
        "",
        "二、Marchenko-Pastur 定理",
        "  在 N 个变量、T 个观测的样本相关矩阵中，若数据是纯白噪声，",
        "  则特征值最大上界：λ_max^noise = σ²(1+√q)²,  q = N/T",
        "  标准化后 σ²=1。",
        "",
        "三、参数与计算",
        f"  N (变量数): {rmt.get('N')}",
        f"  T (观测数): {rmt.get('T'):,}",
        f"  q = N/T: {rmt.get('q', 0):.6f}",
        f"  σ² (标准化后): {rmt.get('sigma_sq', 1)}",
        f"  λ_max^noise = (1+√q)² = {lam_max:.4f}",
        "",
        "四、观测特征值 vs 噪声上界",
        "",
        "  PC    观测 λ    是否 > λ_max^noise    判定",
        "  " + "-" * 50,
    ]
    for j in rmt.get("judgments", []):
        ev = j.get("lambda", 0)
        sig = j.get("is_signal", False)
        tag = "信号" if sig else "噪声"
        cmp = "✓" if sig else "✗"
        lines.append(f"  PC{j.get('pc')}    {ev:.3f}      {cmp}                    {tag}")
    lines.extend([
        "",
        "五、解读",
        f"  • 信号 PC 数: {rmt.get('signal_count', 0)} 个 (λ > {lam_max:.3f})",
        "  • RMT 支持至少保留上述数量的主成分。",
        "  • 结合累计方差、子样本稳定性，3 PC 为 RMT 与实务的折中。",
        "",
        "=" * 70,
    ])
    (out_dir / "pca_rmt_report.txt").write_text("\n".join(lines), encoding="utf-8")
