"""Venue model: geometry, gates, walls and the zone grid used by the digital twin.

All coordinates are in metres. Origin is the bottom-left corner of the ground,
x grows to the right (east) and y grows towards the stage (north).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Gate:
    name: str
    # Gate opening as a line segment (x1, y1, x2, y2) lying on the outer wall.
    seg: tuple[float, float, float, float]
    is_open: bool = True

    @property
    def width(self) -> float:
        x1, y1, x2, y2 = self.seg
        return float(np.hypot(x2 - x1, y2 - y1))

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.seg
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class Venue:
    name: str
    width: float
    height: float
    cell: float                       # zone size in metres (square zones)
    walls: list[tuple[float, float, float, float]]
    gates: list[Gate]
    stage: tuple[float, float, float, float]          # x0, y0, x1, y1 (blocked)
    watch_area: tuple[float, float, float, float]     # where people want to stand
    front_line_y: float                               # barrier line in front of stage
    zone_walkable: np.ndarray = field(init=False)     # (rows, cols) bool

    def __post_init__(self):
        self.cols = int(round(self.width / self.cell))
        self.rows = int(round(self.height / self.cell))
        walk = np.ones((self.rows, self.cols), dtype=bool)
        x0, y0, x1, y1 = self.stage
        for r in range(self.rows):
            for c in range(self.cols):
                cx, cy = (c + 0.5) * self.cell, (r + 0.5) * self.cell
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    walk[r, c] = False
        self.zone_walkable = walk

    # ---- zone helpers -------------------------------------------------
    @property
    def n_zones(self) -> int:
        return self.rows * self.cols

    @property
    def zone_area(self) -> float:
        return self.cell * self.cell

    def zone_index(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        c = np.clip((x / self.cell).astype(int), 0, self.cols - 1)
        r = np.clip((y / self.cell).astype(int), 0, self.rows - 1)
        return r * self.cols + c

    def zone_centers(self) -> np.ndarray:
        r, c = np.divmod(np.arange(self.n_zones), self.cols)
        return np.stack([(c + 0.5) * self.cell, (r + 0.5) * self.cell], axis=1)

    def zone_name(self, idx: int) -> str:
        r, c = divmod(int(idx), self.cols)
        return f"{chr(ord('A') + r)}{c + 1}"

    def all_wall_segments(self) -> list[tuple[float, float, float, float]]:
        """Walls including closed gates (a closed gate behaves like a wall)."""
        segs = list(self.walls)
        segs += [g.seg for g in self.gates if not g.is_open]
        x0, y0, x1, y1 = self.stage
        segs += [(x0, y0, x1, y0), (x0, y0, x0, y1), (x1, y0, x1, y1)]
        return segs

    def open_gates(self) -> list[Gate]:
        return [g for g in self.gates if g.is_open]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "cell": self.cell,
            "rows": self.rows,
            "cols": self.cols,
            "walls": [list(s) for s in self.walls],
            "gates": [
                {"name": g.name, "seg": list(g.seg), "open": g.is_open, "width": g.width}
                for g in self.gates
            ],
            "stage": list(self.stage),
            "watch_area": list(self.watch_area),
            "front_line_y": self.front_line_y,
            "walkable": self.zone_walkable.astype(int).ravel().tolist(),
        }


def _outer_walls_with_gaps(w: float, h: float, gates: list[Gate]):
    """Build the outer boundary as segments, leaving holes where gates are."""
    walls = []
    sides = {
        "S": ((0, 0), (w, 0)),
        "N": ((0, h), (w, h)),
        "W": ((0, 0), (0, h)),
        "E": ((w, 0), (w, h)),
    }
    for side, ((ax, ay), (bx, by)) in sides.items():
        horizontal = ay == by
        cuts = []
        for g in gates:
            x1, y1, x2, y2 = g.seg
            if horizontal and y1 == y2 == ay:
                cuts.append((min(x1, x2), max(x1, x2)))
            elif not horizontal and x1 == x2 == ax:
                cuts.append((min(y1, y2), max(y1, y2)))
        cuts.sort()
        start = ax if horizontal else ay
        end = bx if horizontal else by
        cur = start
        for a, b in cuts:
            if a > cur:
                walls.append((cur, ay, a, ay) if horizontal else (ax, cur, ax, a))
            cur = b
        if cur < end:
            walls.append((cur, ay, end, ay) if horizontal else (ax, cur, ax, end))
    return walls


def rally_ground(gate4_open: bool = False) -> Venue:
    """A 100 m x 70 m open-air rally ground.

    - Stage on the north side, a front barrier 4 m in front of it.
    - Three entry/exit gates on the south wall (G2 in the middle is narrow).
    - A side emergency gate G4 on the east wall, closed by default.
    - An internal fence at y = 25 m with a 20 m gap in the middle: everybody
      has to pass through this gap, which is the designed bottleneck.
    """
    w, h = 100.0, 70.0
    gates = [
        Gate("G1", (10.0, 0.0, 16.0, 0.0)),
        Gate("G2", (47.0, 0.0, 51.0, 0.0)),
        Gate("G3", (84.0, 0.0, 90.0, 0.0)),
        Gate("G4", (100.0, 40.0, 100.0, 46.0), is_open=gate4_open),
    ]
    walls = _outer_walls_with_gaps(w, h, gates)
    # internal fence with a gap between x = 40 and x = 60
    walls += [(0.0, 25.0, 40.0, 25.0), (60.0, 25.0, 100.0, 25.0)]
    # front barrier in front of the stage (people press against it)
    walls += [(25.0, 60.0, 75.0, 60.0)]
    return Venue(
        name="Rally ground (synthetic)",
        width=w,
        height=h,
        cell=5.0,
        walls=walls,
        gates=gates,
        stage=(35.0, 64.0, 65.0, 70.0),
        watch_area=(20.0, 30.0, 80.0, 60.0),
        front_line_y=60.0,
    )
