"""Short-term prediction: which zones will turn critical in the next few minutes.

Method: track the rate of change of fused density per zone with exponential
smoothing, then extrapolate to the critical density. A warning is raised only
when the zone is already dense (>= 4.5 persons/m2), still rising, expected to
hit critical within the horizon, and this has held for 3 ticks in a row.

Settings were chosen on seed 0 as a balance between lead time and false alarms
(see experiments/results/results.md for the trade-off). A simulation-based
forecast (run the crowd model forward from the live state) is future work.
"""
from __future__ import annotations

import numpy as np

from .risk import CRITICAL


class RateTracker:
    def __init__(self, n_zones: int, tau: float = 20.0, horizon: float = 120.0,
                 min_density: float = 4.5, min_rate: float = 0.002, persist: int = 3):
        self.tau, self.horizon = tau, horizon
        self.min_density, self.min_rate, self.persist = min_density, min_rate, persist
        self.prev = None
        self.rate = np.zeros(n_zones)
        self.run = np.zeros(n_zones, dtype=int)
        self._ttc = np.full(n_zones, np.inf)

    def update(self, density: np.ndarray, dt: float):
        if self.prev is not None and dt > 0:
            inst = (density - self.prev) / dt
            a = min(1.0, dt / self.tau)
            self.rate = (1 - a) * self.rate + a * inst
        self.prev = density.copy()
        raw = np.full(len(density), np.inf)
        rising = (self.rate > self.min_rate) & (density >= self.min_density) & (density < CRITICAL)
        raw[rising] = (CRITICAL - density[rising]) / self.rate[rising]
        raw[raw > self.horizon] = np.inf
        self.run = np.where(np.isfinite(raw), self.run + 1, 0)
        self._ttc = np.where(self.run >= self.persist, raw, np.inf)

    def forecast(self, density: np.ndarray, seconds: float) -> np.ndarray:
        return np.maximum(0.0, density + self.rate * seconds)

    def time_to_critical(self, density: np.ndarray | None = None) -> np.ndarray:
        """Seconds until each zone is expected to reach critical density (inf = no warning)."""
        return self._ttc.copy()
