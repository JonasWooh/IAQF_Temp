"""列名、合并工具"""
from pathlib import Path

import pandas as pd


def pair_to_suffix(pair: str) -> str:
    return pair.replace("_", "")


def make_prefix(exchange: str, pair: str) -> str:
    return f"{exchange}_{pair_to_suffix(pair)}"


def load_and_prefix_parquet(fp: Path, exchange: str, pair: str) -> pd.DataFrame:
    df = pd.read_parquet(fp)
    df["time_exchange"] = pd.to_datetime(df["time_exchange"])
    prefix = make_prefix(exchange, pair)
    rename_map = {c: f"{prefix}_{c}" for c in df.columns if c != "time_exchange"}
    df = df.rename(columns=rename_map)
    return df


def merge_1min_tables(
    dfs: list[pd.DataFrame],
    start_date: str,
    end_date: str,
    ffill_limit: int = 5,
) -> pd.DataFrame:
    master = dfs[0]
    for df in dfs[1:]:
        master = pd.merge(master, df, on="time_exchange", how="outer")
    master = master.sort_values("time_exchange").reset_index(drop=True)
    full_range = pd.date_range(
        start=start_date + " 00:00:00",
        end=end_date + " 23:59:00",
        freq="1min",
    )
    master = master.set_index("time_exchange")
    master = master.reindex(full_range)
    master.index.name = "time_exchange"
    master = master.reset_index()
    feature_cols = [c for c in master.columns if c != "time_exchange"]
    master[feature_cols] = master[feature_cols].ffill(limit=ffill_limit)
    return master.set_index("time_exchange")
