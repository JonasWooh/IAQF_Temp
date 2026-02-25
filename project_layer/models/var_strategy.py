"""
VAR Model Strategy - 策略模式
支持 VARX：Gas 极端拥堵哑变量 I(base_fee > threshold) 作为 exog
β 系数直接代表“因链上熔断，跨所价差被额外推高了多少 bps”
"""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from ..config import get_config


class VARModelStrategy(ABC):
    """VAR 估计策略基类"""

    @property
    @abstractmethod
    def regime(self) -> str:
        pass

    @abstractmethod
    def get_data_slice(self, df: pd.DataFrame) -> pd.DataFrame:
        """获取该 regime 的数据切片"""
        pass

    def prepare_data(
        self,
        df: pd.DataFrame,
        include_gas: bool = False,
        institutional: str | None = None,
        include_credit: bool = False,
    ):
        """
        加载并切片 VAR 数据。
        include_gas=True 或 include_credit=True 时返回 (var_data, exog)。
        institutional: None | "weekend" | "pause" — 制度摩擦 dummy，与 gas 一起作为 exog。
        include_credit: 是否加入 Credit block (Basis_USDC_USDT_BN_bps) 作为 exog。
        """
        cfg = get_config()
        cols = cfg.get("var.cols", ["Depeg_bps", "CMLSI", "Basis_bps", "Premium_bps"])
        for c in cols:
            if c not in df.columns:
                raise ValueError(f"Missing column {c}")
        var_data = df[cols].dropna()
        var_data = self.get_data_slice(var_data)
        if not include_gas and not include_credit:
            return var_data

        exog_parts = []
        attrs = {}

        if include_gas:
            from ..data.gas_preprocessing import (
                load_gas_fee,
                build_gas_dummy_for_varx,
                preprocess_gas_level_for_varx,
                preprocess_gas_log_level_for_varx,
                adf_test_gas_level_for_regime,
                adf_test_log_gas_for_regime,
            )
            gas_df = load_gas_fee()
            mode = cfg.get("var.gas_exog_mode", "level")
            adf_signif = cfg.get("var.gas_adf_signif") or cfg.get("var.adf_signif", 0.05)
            gas_exog = None
            if mode == "level":
                adf_res = adf_test_gas_level_for_regime(gas_df, var_data.index, signif=adf_signif)
                if adf_res["is_stationary"]:
                    gas_exog = preprocess_gas_level_for_varx(gas_df, var_data.index)
                    gas_exog.attrs["adf_pvalue"] = adf_res["pvalue"]
                else:
                    mode = "dummy"
            if mode == "log_level" and gas_exog is None:
                adf_res = adf_test_log_gas_for_regime(gas_df, var_data.index, signif=adf_signif)
                if adf_res["is_stationary"]:
                    gas_exog = preprocess_gas_log_level_for_varx(gas_df, var_data.index)
                    gas_exog.attrs["adf_pvalue"] = adf_res["pvalue"]
                else:
                    mode = "dummy"
            if mode == "dummy" and gas_exog is None:
                thresh = cfg.get("var.gas_dummy_threshold_gwei")
                gas_exog = build_gas_dummy_for_varx(
                    gas_df, var_data.index,
                    threshold_gwei=thresh if thresh is not None else None,
                )
            if gas_exog is None:
                raise ValueError(f"Unknown var.gas_exog_mode: {mode}")
            exog_parts.append(("gas", gas_exog))
            attrs["gas_attrs"] = getattr(gas_exog, "attrs", {})

            if institutional in ("weekend", "pause"):
                from ..data.institutional_friction import (
                    build_weekend_dummy,
                    build_conversion_pause_dummy,
                )
                inst = (
                    build_weekend_dummy(var_data.index)
                    if institutional == "weekend"
                    else build_conversion_pause_dummy(var_data.index)
                )
                exog_parts.append(("inst", inst))
                attrs["institutional"] = institutional

        if include_credit:
            from ..data.credit_block import build_credit_block
            credit_exog = build_credit_block(df, var_data.index)
            exog_parts.append(("credit", credit_exog))

        exog_df = pd.DataFrame({k: v for k, v in exog_parts}, index=var_data.index)
        exog_df.attrs.update(attrs)
        return var_data, exog_df

    def fit(
        self,
        var_data: pd.DataFrame,
        ensure_stationary: bool = True,
        fixed_lag: int | None = None,
        exog: pd.Series | np.ndarray | None = None,
    ):
        """
        拟合 VAR 或 VARX。
        exog: 外生变量（如 gas_congestion_dummy 0/1），与 var_data 行对齐；若提供则为 VARX。
        """
        if ensure_stationary:
            var_data, differenced_cols = self._ensure_stationarity(var_data)
        else:
            differenced_cols = set()
        exog_aligned = None
        if exog is not None:
            if isinstance(exog, pd.DataFrame):
                exog_reindexed = exog.reindex(var_data.index)
                mask = ~exog_reindexed.isna().any(axis=1)
                var_data = var_data.loc[mask]
                exog_aligned = exog_reindexed.loc[mask].values
            elif isinstance(exog, pd.Series):
                exog_reindexed = exog.reindex(var_data.index)
                mask = ~exog_reindexed.isna()
                var_data = var_data.loc[mask]
                exog_aligned = exog_reindexed.loc[mask].values.reshape(-1, 1)
            else:
                exog_aligned = np.asarray(exog)
                if exog_aligned.ndim == 1:
                    exog_aligned = exog_aligned.reshape(-1, 1)
                n = min(len(var_data), len(exog_aligned))
                var_data = var_data.iloc[:n]
                exog_aligned = exog_aligned[:n]
        model = VAR(var_data, exog=exog_aligned) if exog_aligned is not None else VAR(var_data)
        if fixed_lag is not None:
            lag = fixed_lag
            results = model.fit(maxlags=lag)
        else:
            lag_res = model.select_order(maxlags=20)
            lag = lag_res.aic
            results = model.fit(maxlags=lag, ic="aic")
        results.differenced_cols = differenced_cols
        results.var_col_names = list(var_data.columns)
        return results

    def _ensure_stationarity(self, var_data: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
        """Return (stationary_data, differenced_cols). differenced_cols = columns that were Δ'd."""
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

    @staticmethod
    def irf_to_level(irf_arr: np.ndarray, col_names: list[str], differenced_cols: set[str], col_idx: int) -> np.ndarray:
        """Cumsum IRF for differenced vars → level response. col_idx = response variable index."""
        col = col_names[col_idx] if col_idx < len(col_names) else None
        if col and col in differenced_cols:
            return np.cumsum(irf_arr)
        return irf_arr


class NormalVARStrategy(VARModelStrategy):
    regime = "normal"

    def get_data_slice(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = get_config()
        start = cfg.get("phases.normal.start")
        end = cfg.get("phases.normal.end")
        return df.loc[start:end]


class CrisisVARStrategy(VARModelStrategy):
    regime = "crisis"

    def get_data_slice(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = get_config()
        start = cfg.get("phases.stressed.start")
        end = cfg.get("phases.stressed.end")
        return df.loc[start:end]


class RecoveryVARStrategy(VARModelStrategy):
    regime = "recovery"

    def get_data_slice(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = get_config()
        start = cfg.get("phases.recovery.start")
        end = cfg.get("phases.recovery.end")
        return df.loc[start:end]
