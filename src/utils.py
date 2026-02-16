from __future__ import annotations

from pathlib import Path
from typing import Union
import numpy as np
import random


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Create directory if it does not exist.
    Accepts str or Path.
    Returns Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)


def print_header(title: str) -> None:
    """
    Simple formatted header for notebook sections.
    """
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}\n")
