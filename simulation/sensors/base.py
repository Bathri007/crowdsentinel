"""Common measurement format used by every data source.

Every source - real or simulated - is converted into Measurement records.
Each record says: "the sum of people over these zones is `value`, with this
variance". A camera gives one record per zone it sees; a cell tower gives one
record for its whole sector. The fusion engine only ever sees these records,
so a simulated feed can be swapped for a real one without touching fusion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Measurement:
    source: str              # e.g. "CAM1", "DRONE", "TOWER-W", "WIFI-G2"
    kind: str                # "cctv" | "drone" | "tower" | "wifi"
    zones: np.ndarray        # zone indices covered by this record
    value: float             # estimated number of people in those zones
    var: float               # variance of `value` (people^2)
    t: float                 # time the observation refers to (s)
    flow: tuple[float, float, float] | None = None  # (vx, vy, speed variance) if the sensor sees motion


@dataclass
class SourceStatus:
    name: str
    kind: str
    online: bool = True
    last_seen: float = -1.0
    real_data: bool = False  # True when fed by a real feed, False when simulated
    coverage: list[int] = field(default_factory=list)

    def to_dict(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "online": self.online,
            "last_seen": self.last_seen,
            "real": self.real_data,
            "coverage": self.coverage,
        }


def zones_in_rect(venue, r0, r1, c0, c1) -> np.ndarray:
    """Zone indices for rows r0..r1 and cols c0..c1 (inclusive), walkable only."""
    out = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if 0 <= r < venue.rows and 0 <= c < venue.cols and venue.zone_walkable[r, c]:
                out.append(r * venue.cols + c)
    return np.array(out, dtype=int)


def zones_in_circle(venue, x, y, radius) -> np.ndarray:
    centers = venue.zone_centers()
    d = np.hypot(centers[:, 0] - x, centers[:, 1] - y)
    walk = venue.zone_walkable.ravel()
    return np.flatnonzero((d <= radius) & walk)
