"""Risk scoring per zone.

Density levels follow common crowd-safety guidance (e.g. Fruin; Still, 2014):
    < 2  persons/m2  normal
    2-4  busy, still free to move
    4-6  high, movement restricted - watch closely
    >= 6 critical, crush risk

The twin is uncertainty-aware: it also scores the *upper* estimate
(density + 1 standard deviation). A zone whose best estimate is fine but whose
upper estimate is dangerous is marked UNVERIFIED - it needs eyes on it.
"""
from __future__ import annotations

import numpy as np

LEVELS = ["normal", "busy", "high", "critical"]
THRESHOLDS = (2.0, 4.0, 6.0)
CRITICAL = THRESHOLDS[2]
PRESSURE_HIGH = 0.6   # density x speed variance, calibrated on the simulation


def level_of(density: np.ndarray) -> np.ndarray:
    return np.digitize(density, THRESHOLDS)


def score(density: np.ndarray, std: np.ndarray, pressure: np.ndarray | None = None) -> dict:
    lvl = level_of(density)
    upper = density + std
    lvl_up = level_of(upper)
    if pressure is not None:
        turbulent = (pressure > PRESSURE_HIGH) & (density >= 3.0)
        lvl = np.where(turbulent, np.minimum(lvl + 1, 3), lvl)
    unverified = (lvl_up > lvl) & (lvl_up >= 2)
    return {"level": lvl, "level_upper": lvl_up, "unverified": unverified, "upper": upper}
