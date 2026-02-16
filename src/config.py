from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, List


@dataclass(frozen=True)
class ProjectConfig:

    # Paths

    project_root: Path = Path(__file__).resolve().parents[1]
    data_raw_dir: Path = project_root / "data" / "raw"
    data_processed_dir: Path = project_root / "data" / "processed"
    reports_figures_dir: Path = project_root / "reports" / "figures"
    reports_tables_dir: Path = project_root / "reports" / "tables"


    # Asset & sample period

    ticker: str = "SPY"
    start: str = "2005-01-01"
    end: str = "2025-01-01"


    # Trading constants

    trading_days: int = 252
    random_seed: int = 42


    # Forecast setup

    horizons: Tuple[int, ...] = (1, 5, 10)

    # Walk-forward parameters
    train_window: int = 1000
    refit_every: int = 1

    # Baseline rolling windows
    roll_windows: Tuple[int, ...] = (20, 60)

    # EWMA
    ewma_lambda: float = 0.94


    # Economic test: volatility targeting

    vol_target_annual: float = 0.15
    w_min: float = 0.0
    w_max: float = 2.0


    # Derived properties

    def n_horizons(self) -> int:
        return len(self.horizons)

    @property
    def model_names(self) -> List[str]:
        return ["roll20", "roll60", "ewma", "garch"]

    def as_dict(self):
        return asdict(self)


cfg = ProjectConfig()
