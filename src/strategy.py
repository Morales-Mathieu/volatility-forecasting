from __future__ import annotations

import numpy as np
import pandas as pd

from config import ProjectConfig


def vol_target_weights(vol_forecast_ann: pd.Series, cfg: ProjectConfig) -> pd.Series:
    """
    Volatility targeting weight:
        w_t = clip( sigma_target / sigma_hat_t, w_min, w_max )

    Inputs are annualized vols.
    """
    w = cfg.vol_target_annual / vol_forecast_ann
    w = w.clip(lower=cfg.w_min, upper=cfg.w_max)
    return w


def backtest_vol_targeting(
    df: pd.DataFrame,
    vol_forecast_ann: pd.Series,
    cfg: ProjectConfig,
    name: str = "vol_target",
) -> pd.DataFrame:
    """
    Backtest (no look-ahead):
      - Use forecast volatility at time t-1 to size exposure for return at time t
      - rp_t = w_{t-1} * r_t  (r_t are log returns)

    Returns DataFrame with columns: w, r, rp, equity
    """
    r = df["r"].copy()

    common = r.index.intersection(vol_forecast_ann.index)
    r = r.loc[common]
    vol_forecast_ann = vol_forecast_ann.loc[common]

    w = vol_target_weights(vol_forecast_ann, cfg)
    w_lag = w.shift(1)

    # Avoid backfill (can introduce look-ahead). Use neutral exposure for first obs.
    w_lag = w_lag.fillna(1.0).clip(lower=cfg.w_min, upper=cfg.w_max)

    rp = (w_lag * r).rename("rp")

    # r and rp are log returns -> equity = exp(cumsum(log_returns))
    equity = np.exp(rp.cumsum()).rename("equity")

    out = pd.DataFrame({"w": w_lag, "r": r, "rp": rp, "equity": equity})
    out.attrs["name"] = name
    return out


def backtest_buy_hold(df: pd.DataFrame) -> pd.DataFrame:
    r = df["r"].copy()
    equity = np.exp(r.cumsum()).rename("equity")
    out = pd.DataFrame({"r": r, "rp": r, "equity": equity})
    out.attrs["name"] = "buy_hold"
    return out


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def perf_stats(rp: pd.Series, trading_days: int = 252) -> dict:
    """
    Performance stats from log returns rp.
    - Equity: exp(cumsum(rp))
    - CAGR computed from equity endpoints
    - Sharpe uses mean/std of daily log returns (ok for comparison)
    """
    rp = rp.dropna()
    if len(rp) < 10:
        return {}

    equity = np.exp(rp.cumsum())
    total_years = len(rp) / trading_days

    cagr = float(equity.iloc[-1] ** (1 / total_years) - 1) if total_years > 0 else np.nan

    mean_d = float(rp.mean())
    vol_d = float(rp.std())

    ann_return = mean_d * trading_days
    ann_vol = vol_d * np.sqrt(trading_days)
    sharpe = (ann_return / ann_vol) if ann_vol > 0 else np.nan

    mdd = max_drawdown(equity)

    return {
        "cagr": cagr,
        "ann_return_log": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(mdd),
        "n_obs": int(len(rp)),
    }
