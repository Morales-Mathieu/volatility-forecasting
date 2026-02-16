from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from config import ProjectConfig
from utils import ensure_dir


def _align(y_true: pd.Series, y_pred: pd.Series) -> tuple[pd.Series, pd.Series]:
    yt = y_true.dropna()
    yp = y_pred.dropna()
    common = yt.index.intersection(yp.index)
    return yt.loc[common], yp.loc[common]



# Loss functions (vol forecasting)

def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    yt, yp = _align(y_true, y_pred)
    return float((yt - yp).abs().mean())


def mse(y_true: pd.Series, y_pred: pd.Series) -> float:
    yt, yp = _align(y_true, y_pred)
    return float(((yt - yp) ** 2).mean())


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def qlike(y_true: pd.Series, y_pred: pd.Series, eps: float = 1e-12) -> float:
    """
    QLIKE loss (standard in volatility forecasting):
        QLIKE = mean( log(pred^2) + true^2 / pred^2 )

    Assumes y_true and y_pred are on the same scale (here: annualized vol).
    """
    yt, yp = _align(y_true, y_pred)
    yp = yp.clip(lower=eps)

    loss = np.log((yp ** 2) + eps) + (yt ** 2) / ((yp ** 2) + eps)
    return float(loss.mean())


# Evaluation table builder


def evaluate_forecasts(
    df: pd.DataFrame,
    forecast_panel: pd.DataFrame,
    cfg: ProjectConfig,
    models: List[str] = ("roll20", "roll60", "ewma", "garch"),
) -> pd.DataFrame:
    """
    Compare forecasts vs realized targets for each horizon.

    Expected:
      - df: realized_vol_{h}d
      - forecast_panel: {model}_h={h} columns (annualized)
    """
    rows = []

    for h in cfg.horizons:
        y_true = df[f"realized_vol_{h}d"]

        for m in models:
            col = f"{m}_h={h}"
            if col not in forecast_panel.columns:
                continue

            y_pred = forecast_panel[col]
            yt, yp = _align(y_true, y_pred)

            rows.append(
                {
                    "model": m,
                    "horizon_days": h,
                    "MAE": mae(yt, yp),
                    "RMSE": rmse(yt, yp),
                    "MSE": mse(yt, yp),
                    "QLIKE": qlike(yt, yp),
                    "n_obs": int(len(yt)),
                }
            )

    out = pd.DataFrame(rows).sort_values(["horizon_days", "QLIKE", "RMSE"])
    return out


def save_metrics_table(cfg: ProjectConfig, metrics_df: pd.DataFrame, filename: str = "metrics_table.csv") -> Path:
    ensure_dir(cfg.reports_tables_dir)
    path = cfg.reports_tables_dir / filename
    metrics_df.to_csv(path, index=False)
    return path


def save_metrics_markdown(cfg: ProjectConfig, metrics_df: pd.DataFrame, filename: str = "metrics_table.md") -> Path:
    """
    README-friendly table export.
    """
    ensure_dir(cfg.reports_tables_dir)
    path = cfg.reports_tables_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(metrics_df.to_markdown(index=False))
        f.write("\n")
    return path
