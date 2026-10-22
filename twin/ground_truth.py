"""Turn a simulated snapshot into per-zone truth: count, density, flow, pressure."""
from __future__ import annotations

import numpy as np

from simulation.venue import Venue


def zone_truth(venue: Venue, pos: np.ndarray, vel: np.ndarray) -> dict:
    n = venue.n_zones
    if len(pos) == 0:
        z = np.zeros(n)
        return {"count": z, "density": z, "vx": z, "vy": z, "speed_var": z, "pressure": z}
    idx = venue.zone_index(pos[:, 0], pos[:, 1])
    count = np.bincount(idx, minlength=n).astype(float)
    sx = np.bincount(idx, weights=vel[:, 0], minlength=n)
    sy = np.bincount(idx, weights=vel[:, 1], minlength=n)
    safe = np.maximum(count, 1)
    vx, vy = sx / safe, sy / safe
    # velocity variance per zone (used for crowd pressure, Helbing et al. 2007)
    dvx = vel[:, 0] - vx[idx]
    dvy = vel[:, 1] - vy[idx]
    var = np.bincount(idx, weights=dvx ** 2 + dvy ** 2, minlength=n) / safe
    density = count / venue.zone_area
    return {
        "count": count,
        "density": density,
        "vx": vx,
        "vy": vy,
        "speed_var": var,
        "pressure": density * var,
    }
