from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.graph_objects as go

from config import ProjectConfig
from utils import ensure_dir


# Core save helper


def savefig(cfg: ProjectConfig, filename: str) -> Path:
    ensure_dir(cfg.reports_figures_dir)
    path = cfg.reports_figures_dir / filename
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()  # avoid figure stacking / memory issues
    return path


# Stylized facts (2D)


def plot_price(cfg: ProjectConfig, df: pd.DataFrame) -> Path:
    plt.figure()
    df["adj_close"].plot()
    plt.title(f"{cfg.ticker} Adjusted Close")
    plt.xlabel("Date")
    plt.ylabel("Price")
    return savefig(cfg, "01_price.png")


def plot_returns(cfg: ProjectConfig, df: pd.DataFrame) -> Path:
    plt.figure()
    df["r"].plot()
    plt.title(f"{cfg.ticker} Log Returns")
    plt.xlabel("Date")
    plt.ylabel("Log return")
    return savefig(cfg, "02_returns.png")


def plot_rolling_vol(cfg: ProjectConfig, df: pd.DataFrame) -> Path:
    plt.figure()
    for w in cfg.roll_windows:
        col = f"roll_vol_{w}d"
        df[col].plot(label=col)
    plt.title(f"{cfg.ticker} Rolling Volatility (Annualized)")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    return savefig(cfg, "03_rolling_vol.png")


def plot_hist_returns(cfg: ProjectConfig, df: pd.DataFrame) -> Path:
    plt.figure()
    df["r"].hist(bins=80)
    plt.title(f"{cfg.ticker} Returns Distribution")
    plt.xlabel("Log return")
    plt.ylabel("Count")
    return savefig(cfg, "04_returns_hist.png")


def plot_acf_returns(cfg: ProjectConfig, df: pd.DataFrame, lags: int = 40) -> Path:
    _ = sm.graphics.tsa.plot_acf(df["r"], lags=lags)
    plt.title(f"{cfg.ticker} ACF of Returns")
    return savefig(cfg, "05_acf_returns.png")


def plot_acf_squared_returns(cfg: ProjectConfig, df: pd.DataFrame, lags: int = 40) -> Path:
    _ = sm.graphics.tsa.plot_acf(df["r"] ** 2, lags=lags)
    plt.title(f"{cfg.ticker} ACF of Squared Returns (Vol Clustering)")
    return savefig(cfg, "06_acf_sq_returns.png")


def plot_realized_targets(cfg: ProjectConfig, df: pd.DataFrame) -> Path:
    plt.figure()
    for h in cfg.horizons:
        col = f"realized_vol_{h}d"
        df[col].plot(label=col)
    plt.title(f"{cfg.ticker} Realized Volatility Targets (Annualized)")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    return savefig(cfg, "07_realized_vol_targets.png")


# 3D surfaces (Plotly HTML)


def plot_vol_surface_3d_research(
    cfg: ProjectConfig,
    vol_df: pd.DataFrame,
    title: str,
    filename_html: str,
    smooth_window: int = 5,
) -> Path:
    """
    Research-grade interactive 3D surface.

    vol_df:
      - index = dates
      - columns like "h=1","h=5","h=10"
      - values = annualized vol

    Features:
      - light smoothing along time for a cleaner "nappe"
      - numeric time axis (better geometry)
      - contours + clean white template
      - hover shows real dates
    """
    ensure_dir(cfg.reports_figures_dir)

    df_ = vol_df.copy().sort_index()

    cols_sorted = sorted(df_.columns, key=lambda c: int(c.split("=")[1]))
    df_ = df_[cols_sorted]

    if smooth_window and smooth_window > 1:
        df_ = df_.rolling(smooth_window, min_periods=1).mean()

    x = np.arange(len(df_))
    x_text = df_.index.strftime("%Y-%m-%d").to_numpy()
    y = np.array([int(c.split("=")[1]) for c in df_.columns])
    z = df_.values.T  # (len(y), len(x))

    fig = go.Figure(
        data=[
            go.Surface(
                z=z,
                x=x,
                y=y,
                customdata=np.tile(x_text, (len(y), 1)),
                hovertemplate="Date=%{customdata}<br>Horizon=%{y}d<br>Vol=%{z:.3f}<extra></extra>",
                contours={
                    "z": {"show": True, "usecolormap": True, "highlightcolor": "black", "project_z": True}
                },
            )
        ]
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(title="Time (index)", showgrid=False, zeroline=False),
            yaxis=dict(title="Horizon (days)", showgrid=False, zeroline=False),
            zaxis=dict(title="Annualized Vol", showgrid=False, zeroline=False),
            camera=dict(eye=dict(x=1.4, y=1.6, z=0.9)),
        ),
    )

    path = cfg.reports_figures_dir / filename_html
    fig.write_html(path, include_plotlyjs="cdn")
    
    png_path = cfg.reports_figures_dir / filename_html.replace(".html", ".png")
    fig.write_image(png_path, scale=2)

    return path


def plot_error_surface_3d(
    cfg: ProjectConfig,
    realized_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    title: str,
    filename_html: str,
    smooth_window: int = 5,
) -> Path:
    """
    3D surface of forecast error (forecast - realized), annualized scale.

    Inputs:
      - realized_df columns like "h=1","h=5","h=10"
      - forecast_df columns like "h=1","h=5","h=10"
    """
    ensure_dir(cfg.reports_figures_dir)

    common = realized_df.index.intersection(forecast_df.index)
    real = realized_df.loc[common].sort_index()
    fcst = forecast_df.loc[common].sort_index()

    cols_sorted = sorted(real.columns, key=lambda c: int(c.split("=")[1]))
    real = real[cols_sorted]
    fcst = fcst[cols_sorted]

    err = (fcst - real)

    if smooth_window and smooth_window > 1:
        err = err.rolling(smooth_window, min_periods=1).mean()

    x = np.arange(len(err))
    x_text = err.index.strftime("%Y-%m-%d").to_numpy()
    y = np.array([int(c.split("=")[1]) for c in err.columns])
    z = err.values.T

    fig = go.Figure(
        data=[
            go.Surface(
                z=z,
                x=x,
                y=y,
                customdata=np.tile(x_text, (len(y), 1)),
                hovertemplate="Date=%{customdata}<br>Horizon=%{y}d<br>Error=%{z:.3f}<extra></extra>",
                contours={"z": {"show": True, "usecolormap": True, "project_z": True}},
            )
        ]
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(title="Time (index)", showgrid=False, zeroline=False),
            yaxis=dict(title="Horizon (days)", showgrid=False, zeroline=False),
            zaxis=dict(title="Forecast Error (ann.)", showgrid=False, zeroline=False),
            camera=dict(eye=dict(x=1.4, y=1.6, z=0.9)),
        ),
    )

    path = cfg.reports_figures_dir / filename_html
    fig.write_html(path, include_plotlyjs="cdn")
    
    png_path = cfg.reports_figures_dir / filename_html.replace(".html", ".png")
    fig.write_image(png_path, scale=2)

    return path



# Strategy visuals


def plot_equity_curves(cfg: ProjectConfig, curves: dict, filename: str = "08_equity_curves.png") -> Path:
    """
    curves: dict[str, pd.Series] name -> equity series
    """
    plt.figure()
    for name, eq in curves.items():
        eq.plot(label=name)
    plt.title("Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Equity (start=1)")
    plt.legend()
    return savefig(cfg, filename)


def plot_drawdowns(cfg: ProjectConfig, curves: dict, filename: str = "09_drawdowns.png") -> Path:
    plt.figure()
    for name, eq in curves.items():
        peak = eq.cummax()
        dd = eq / peak - 1.0
        dd.plot(label=name)
    plt.title("Drawdowns")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    return savefig(cfg, filename)



# Forecast evaluation visuals (2D)


def plot_forecast_vs_realized(
    cfg: ProjectConfig,
    df: pd.DataFrame,
    forecast_panel: pd.DataFrame,
    model: str,
    h: int,
    start_date: str | None = None,
) -> Path:
    """
    Line plot: forecast vol vs realized vol for horizon h.
    model in {"ewma","garch","roll20","roll60"}
    """
    y_true = df[f"realized_vol_{h}d"]
    y_pred = forecast_panel[f"{model}_h={h}"]

    common = y_true.index.intersection(y_pred.index)
    y_true = y_true.loc[common]
    y_pred = y_pred.loc[common]

    if start_date is not None:
        y_true = y_true.loc[y_true.index >= start_date]
        y_pred = y_pred.loc[y_pred.index >= start_date]

    plt.figure()
    y_true.plot(label="realized", alpha=0.9)
    y_pred.plot(label=f"forecast ({model})", alpha=0.9)
    plt.title(f"Forecast vs Realized Vol — {cfg.ticker} — horizon {h}d")
    plt.xlabel("Date")
    plt.ylabel("Annualized Vol")
    plt.legend()

    suffix = f"_from_{start_date}" if start_date else ""
    return savefig(cfg, f"10_fcast_vs_real_{model}_h{h}{suffix}.png")


def plot_forecast_scatter(
    cfg: ProjectConfig,
    df: pd.DataFrame,
    forecast_panel: pd.DataFrame,
    model: str,
    h: int,
) -> Path:
    """
    Scatter plot: forecast vs realized (with y=x line).
    """
    y_true = df[f"realized_vol_{h}d"]
    y_pred = forecast_panel[f"{model}_h={h}"]

    common = y_true.index.intersection(y_pred.index)
    y_true = y_true.loc[common]
    y_pred = y_pred.loc[common]

    plt.figure()
    plt.scatter(y_pred, y_true, s=10, alpha=0.35, edgecolors="none")
    lo = float(min(y_pred.min(), y_true.min()))
    hi = float(max(y_pred.max(), y_true.max()))
    plt.plot([lo, hi], [lo, hi])
    plt.grid(True, alpha=0.2)

    plt.title(f"Scatter — Forecast vs Realized — {model} — h={h}d")
    plt.xlabel("Forecast vol (ann.)")
    plt.ylabel("Realized vol (ann.)")

    return savefig(cfg, f"11_scatter_{model}_h{h}.png")


def plot_error_distribution(
    cfg: ProjectConfig,
    df: pd.DataFrame,
    forecast_panel: pd.DataFrame,
    model: str,
    h: int,
) -> Path:
    """
    Histogram of forecast error (forecast - realized).
    """
    y_true = df[f"realized_vol_{h}d"]
    y_pred = forecast_panel[f"{model}_h={h}"]

    common = y_true.index.intersection(y_pred.index)
    err = (y_pred.loc[common] - y_true.loc[common]).dropna()

    plt.figure()
    err.hist(bins=60)
    plt.title(f"Error distribution — {model} — h={h}d (forecast - realized)")
    plt.xlabel("Error (ann. vol)")
    plt.ylabel("Count")
    return savefig(cfg, f"12_error_hist_{model}_h{h}.png")
