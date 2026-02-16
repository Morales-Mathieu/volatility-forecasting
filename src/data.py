from __future__ import annotations

from dataclasses import asdict
from typing import Tuple
import json

import numpy as np
import pandas as pd
import yfinance as yf

from config import ProjectConfig
from utils import ensure_dir


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def download_prices(cfg: ProjectConfig) -> pd.DataFrame:
    """
    Download daily OHLCV data from Yahoo Finance via yfinance.
    """
    df = yf.download(
        cfg.ticker,
        start=cfg.start,
        end=cfg.end,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for ticker={cfg.ticker}. "
            f"Check symbol, dates, or network connection."
        )

    df = _flatten_columns(df)
    df.index = pd.to_datetime(df.index)

    # make index stable/clean
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    return df


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Log returns:
        r_t = log(P_t / P_{t-1})
    """
    r = np.log(prices / prices.shift(1))
    return r.dropna()


def realized_vol_from_returns(r: pd.Series, h: int) -> pd.Series:
    """
    Realized volatility proxy over horizon h (days):
        sigma_real(t,h) = sqrt( sum_{i=0}^{h-1} r_{t-i}^2 )
    """
    rv = np.sqrt((r**2).rolling(window=h).sum())
    return rv.dropna()


def build_dataset(cfg: ProjectConfig, save: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build master dataset with:
    - adj_close
    - log returns
    - rolling vol baselines (annualized)
    - realized vol targets (annualized)
    """
    ensure_dir(cfg.data_raw_dir)
    ensure_dir(cfg.data_processed_dir)

    prices = download_prices(cfg)

    if "Adj Close" in prices.columns:
        px = prices["Adj Close"].rename("adj_close")
    elif "Close" in prices.columns:
        px = prices["Close"].rename("adj_close")
    else:
        raise KeyError("Neither 'Adj Close' nor 'Close' found in downloaded prices.")

    r = compute_log_returns(px).rename("r")
    df = pd.DataFrame({"adj_close": px}).join(r, how="inner")

    # Rolling volatility baselines (annualized)
    for w in cfg.roll_windows:
        df[f"roll_vol_{w}d"] = df["r"].rolling(w).std() * np.sqrt(cfg.trading_days)

    # Realized vol targets (annualized)
    for h in cfg.horizons:
        df[f"realized_vol_{h}d"] = realized_vol_from_returns(df["r"], h) * np.sqrt(cfg.trading_days / h)

    df = df.dropna()

    if save:
        raw_path = cfg.data_raw_dir / f"{cfg.ticker}_ohlcv.csv"
        proc_path = cfg.data_processed_dir / f"{cfg.ticker}_dataset.csv"
        cfg_path = cfg.data_processed_dir / "config_used.json"

        prices.to_csv(raw_path)
        df.to_csv(proc_path)

        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2, default=str)

    return df, df["r"]
