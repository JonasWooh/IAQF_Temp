"""
CMLSI PCA - Stateful: fit() 仅限 Normal 期，transform() 全样本
消除 Look-ahead Bias
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ..config import get_config


class CMLSIPCA:
    """PCA 类：fit 仅限 Normal 期，transform 全期"""

    def __init__(self, features: list[str], n_pcs: int = 3):
        self.features = features
        self.n_pcs = n_pcs
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.loadings_: np.ndarray | None = None
        self.meta_: dict | None = None

    def fit(self, data: pd.DataFrame, phase_start: str, phase_end: str) -> "CMLSIPCA":
        """仅用 Normal 期拟合"""
        cfg = get_config()
        normal = data.loc[phase_start:phase_end][self.features].dropna()
        if len(normal) < 50:
            raise ValueError(f"Insufficient normal-period rows: {len(normal)}")
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(normal)
        self.pca = PCA(n_components=min(self.n_pcs, len(self.features)))
        self.pca.fit(X)
        evr = self.pca.explained_variance_ratio_
        weights = evr / evr.sum()
        self.loadings_ = np.asarray((self.pca.components_.T @ weights).flatten())
        return self

    def transform(
        self,
        data: pd.DataFrame,
        regime_windows: list[tuple[str, str]],
    ) -> pd.Series:
        """对多个时期分别 transform（OOS）"""
        if self.scaler is None or self.pca is None:
            raise RuntimeError("Call fit() first")
        weights = self.pca.explained_variance_ratio_ / self.pca.explained_variance_ratio_.sum()
        pca_data = data[self.features].copy()
        valid_mask = pca_data.notna().all(axis=1)
        cmlsi_series = pd.Series(index=data.index, dtype=float)
        for start, end in regime_windows:
            subset = pca_data.loc[valid_mask].loc[start:end]
            if len(subset) == 0:
                continue
            X = self.scaler.transform(subset)
            pc_scores = self.pca.transform(X)
            cmlsi_scores = (pc_scores * weights).sum(axis=1)
            cmlsi_series.loc[subset.index] = np.asarray(cmlsi_scores)
        cmlsi_all = cmlsi_series.values
        phase_stressed_start = get_config().get("phases.stressed.start")
        phase_stressed_end = get_config().get("phases.stressed.end")
        phase_normal_start = get_config().get("phases.normal.start")
        phase_normal_end = get_config().get("phases.normal.end")
        mask_crisis = (data.index >= phase_stressed_start) & (data.index <= phase_stressed_end)
        mask_normal = (data.index >= phase_normal_start) & (data.index <= phase_normal_end)
        crisis_mean = np.nanmean(cmlsi_all[mask_crisis]) if np.any(mask_crisis) else np.nan
        normal_mean = np.nanmean(cmlsi_all[mask_normal]) if np.any(mask_normal) else np.nan
        if not (np.isnan(crisis_mean) or np.isnan(normal_mean)) and crisis_mean < normal_mean:
            cmlsi_all = -cmlsi_all
            self.loadings_ = -(self.pca.components_.T @ weights).flatten()
        return pd.Series(cmlsi_all, index=data.index).ffill(limit=5).bfill(limit=5)

    def get_meta(self, data: pd.DataFrame, regime_windows: list[tuple[str, str]]) -> dict:
        """生成 meta 信息"""
        phase_start, phase_end = regime_windows[0][0], regime_windows[0][1]
        normal = data.loc[phase_start:phase_end][self.features].dropna()
        X = self.scaler.transform(normal)
        pca_full = PCA(n_components=None).fit(X)
        evr = self.pca.explained_variance_ratio_
        cumevr = sum(evr[: self.pca.n_components_])
        return {
            "features": self.features,
            "loadings_dict": dict(zip(self.features, self.loadings_.tolist())),
            "loadings_array": self.loadings_.tolist(),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "cmlsi_explained_variance_ratio": float(cumevr),
            "explained_variance_ratio": pca_full.explained_variance_ratio_.tolist(),
            "oos_fit_period": f"{phase_start} to {phase_end}",
        }
