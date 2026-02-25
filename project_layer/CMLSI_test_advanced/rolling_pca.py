"""
Step 2: Rolling / Dynamic PCA → Adaptive CMLSI
- Rolling window PCA: loadings drift, explained variance over time
- Adaptive CMLSI: CMLSI computed with rolling PCA (each window fit + transform center)
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


def compute_rolling_pca_metrics(
    df: pd.DataFrame,
    features: list[str],
    window_minutes: int = 1440,  # 24h
    step_minutes: int = 60,
) -> pd.DataFrame:
    """
    Rolling PCA: at each step, fit PCA on [t-window, t], compute:
    - PC1 explained variance ratio
    - PC1 loading (first component) for each feature
    Returns DataFrame with index=time, columns=metrics.
    """
    X = df[features].dropna(how="any")
    if len(X) < window_minutes:
        return pd.DataFrame()
    idx = X.index
    n = len(X)
    window = window_minutes
    step = step_minutes
    rows = []
    for i in range(window, n, step):
        sub = X.iloc[i - window : i]
        if len(sub) < window // 2:
            continue
        scaler = StandardScaler()
        Z = scaler.fit_transform(sub)
        pca = PCA(n_components=min(3, len(features)))
        pca.fit(Z)
        evr1 = pca.explained_variance_ratio_[0]
        load1 = pca.components_[0]
        row = {"pc1_evr": evr1, "time": idx[i]}
        for j, f in enumerate(features):
            row[f"load_{f}"] = load1[j] if j < len(load1) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).set_index("time")
    return out


def compute_adaptive_cmlsi(
    df: pd.DataFrame,
    features: list[str],
    window_minutes: int = 1440,
    n_pcs: int = 3,
) -> pd.Series:
    """
    Adaptive CMLSI: at each t, fit PCA on [t-window, t], transform point t (or center).
    Uses expanding window until window is full, then rolling.
    """
    X = df[features].dropna(how="any")
    if len(X) < window_minutes:
        return pd.Series(dtype=float)
    idx = X.index
    n = len(X)
    window = window_minutes
    cmlsi = np.full(n, np.nan)
    for i in range(window, n):
        sub = X.iloc[i - window : i]
        if len(sub) < window // 2:
            continue
        scaler = StandardScaler()
        Z = scaler.fit_transform(sub)
        pca = PCA(n_components=min(n_pcs, len(features)))
        pca.fit(Z)
        weights = pca.explained_variance_ratio_ / pca.explained_variance_ratio_.sum()
        # Transform the last point (or center) - use point at i-1 to avoid look-ahead
        z_last = scaler.transform(X.iloc[i : i + 1])
        scores = pca.transform(z_last)[0]
        cmlsi[i] = np.dot(scores[: len(weights)], weights)
    # Align with Normal/Crisis mean for sign (Crisis stress = higher CMLSI)
    pn = get_config().get("phases.normal.start", "2023-03-01")
    pe_n = get_config().get("phases.normal.end", "2023-03-09")
    ps = get_config().get("phases.stressed.start", "2023-03-10")
    pe_s = get_config().get("phases.stressed.end", "2023-03-13")
    mask_n = (idx >= pn) & (idx <= pe_n)
    mask_c = (idx >= ps) & (idx <= pe_s)
    mean_n = np.nanmean(cmlsi[mask_n])
    mean_c = np.nanmean(cmlsi[mask_c])
    if not (np.isnan(mean_n) or np.isnan(mean_c)) and mean_c < mean_n:
        cmlsi = -cmlsi
    return pd.Series(cmlsi, index=idx).ffill(limit=5).bfill(limit=5)


def run_adaptive_cmlsi(out_dir: Path | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """
    Compute Adaptive CMLSI and rolling metrics.
    Returns (cmlsi_adaptive_series, rolling_metrics_df).
    """
    cfg = get_config()
    features, df = _get_features_and_data()
    rolling_metrics = compute_rolling_pca_metrics(df, features, window_minutes=1440, step_minutes=60)
    cmlsi_adaptive = compute_adaptive_cmlsi(df, features, window_minutes=1440, n_pcs=3)

    out_dir = out_dir or cfg.get_output_dir() / "CMLSI_test_advanced"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not cmlsi_adaptive.empty:
        cmlsi_adaptive.to_csv(out_dir / "cmlsi_adaptive.csv")
    if not rolling_metrics.empty:
        rolling_metrics.to_csv(out_dir / "rolling_pca_metrics.csv")
    print(f"  [OK] Adaptive CMLSI: {out_dir / 'cmlsi_adaptive.csv'}")
    return cmlsi_adaptive, rolling_metrics
