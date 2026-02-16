from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from arch import arch_model

from config import ProjectConfig



# Helpers


def annualize_vol(daily_vol: pd.Series, trading_days: int = 252) -> pd.Series:
    return daily_vol * np.sqrt(trading_days)


def forecast_rolling_vol(df: pd.DataFrame, window: int, trading_days: int = 252) -> pd.Series:
    """
    Rolling volatility forecast = rolling std of returns (daily) annualized.
    """
    vol = df["r"].rolling(window).std()
    return annualize_vol(vol, trading_days=trading_days).dropna().rename(f"roll{window}")



# EWMA (RiskMetrics)


def ewma_variance(r: pd.Series, lam: float) -> pd.Series:
    """
    EWMA variance recursion (1-step ahead):
        sigma_t^2 = lam * sigma_{t-1}^2 + (1-lam) * r_{t-1}^2

    Returns series sigma_t^2 aligned so that sigma_t^2 is the forecast variance for time t
    using information up to t-1.
    """
    r2 = r**2
    var = pd.Series(index=r.index, dtype=float)

    # init
    var.iloc[0] = r2.iloc[:20].mean() if len(r2) >= 20 else r2.mean()

    for i in range(1, len(r2)):
        var.iloc[i] = lam * var.iloc[i - 1] + (1 - lam) * r2.iloc[i - 1]

    return var


def forecast_ewma_vol(df: pd.DataFrame, lam: float, trading_days: int = 252) -> pd.Series:
    """
    1-step ahead EWMA volatility forecast (annualized).
    """
    var = ewma_variance(df["r"], lam=lam)
    vol_daily = np.sqrt(var)
    vol_ann = annualize_vol(vol_daily, trading_days=trading_days)
    vol_ann.name = "ewma"
    return vol_ann.dropna()



# GARCH(1,1) Walk-forward


def _fit_garch11(train_r: pd.Series):
    scale = 100.0
    r_scaled = train_r * scale
    am = arch_model(r_scaled, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    return res, scale


def forecast_garch_vol_walkforward(df: pd.DataFrame, cfg: ProjectConfig) -> pd.Series:
    """
    Walk-forward 1-step ahead annualized volatility forecast from GARCH(1,1).
    Robust to occasional fit failures: keeps last successful fit.
    """
    r = df["r"].copy()
    out = pd.Series(index=r.index, dtype=float, name="garch")

    last_fit_i: Optional[int] = None
    res = None
    scale = 100.0

    for i in range(cfg.train_window, len(r)):
        need_refit = (last_fit_i is None) or ((i - last_fit_i) >= cfg.refit_every)

        if need_refit:
            train = r.iloc[i - cfg.train_window : i]
            try:
                res, scale = _fit_garch11(train)
                last_fit_i = i
            except Exception:
                pass

        if res is None:
            continue

        try:
            f = res.forecast(horizon=1, reindex=False)
            var_scaled = float(f.variance.values[-1, 0])
            if not np.isfinite(var_scaled) or var_scaled <= 0:
                continue
            vol = np.sqrt(var_scaled) / scale
            out.iloc[i] = vol
        except Exception:
            continue

    out = out.dropna()
    return annualize_vol(out, trading_days=cfg.trading_days)



# Multi-horizon: make horizons actually different (simple smoothing)

def horizon_transform(vol_1d_ann: pd.Series, h: int) -> pd.Series:
    """
    Simple, defensible way to create horizon-dependent forecasts (for 3D surfaces + evaluation):
    - use a rolling mean of the 1-day forecast over h days
    This yields distinct surfaces without introducing complex multi-step GARCH math.
    """
    if h <= 1:
        return vol_1d_ann.copy()
    return vol_1d_ann.rolling(h, min_periods=1).mean()


def build_forecast_panel(df: pd.DataFrame, cfg: ProjectConfig, include_baselines: bool = True) -> pd.DataFrame:
    panel = pd.DataFrame(index=df.index)

    ewma_1d = forecast_ewma_vol(df, lam=cfg.ewma_lambda, trading_days=cfg.trading_days)
    garch_1d = forecast_garch_vol_walkforward(df, cfg)

    roll_map = {}
    if include_baselines:
        for w in cfg.roll_windows:
            roll_map[w] = forecast_rolling_vol(df, window=w, trading_days=cfg.trading_days)

    for h in cfg.horizons:
        panel[f"ewma_h={h}"] = horizon_transform(ewma_1d, h=h)
        panel[f"garch_h={h}"] = horizon_transform(garch_1d, h=h)

        if include_baselines:
            for w, s in roll_map.items():
                panel[f"roll{w}_h={h}"] = horizon_transform(s, h=h)

    panel = panel.dropna()
    return panel
