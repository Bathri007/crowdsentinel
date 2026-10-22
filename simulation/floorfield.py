"""Floor fields: for every point of the ground, which direction leads to a target.

We rasterise the venue on a fine grid (0.5 m), mark walls/stage as blocked and run
a multi-source Dijkstra from the target cells. Each free cell then points to its
lowest-distance neighbour. Agents read their desired walking direction from it.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .venue import Venue

RES = 0.5  # metres per floor-field cell


def _rasterise_blocked(venue: Venue) -> np.ndarray:
    nx, ny = int(venue.width / RES), int(venue.height / RES)
    blocked = np.zeros((ny, nx), dtype=bool)
    xs = (np.arange(nx) + 0.5) * RES
    ys = (np.arange(ny) + 0.5) * RES
    X, Y = np.meshgrid(xs, ys)
    for (x1, y1, x2, y2) in venue.all_wall_segments():
        d = _dist_to_segment(X, Y, x1, y1, x2, y2)
        blocked |= d < 0.45
    x0, y0, x1, y1 = venue.stage
    blocked |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    return blocked


def _dist_to_segment(X, Y, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    t = np.clip(((X - x1) * dx + (Y - y1) * dy) / max(L2, 1e-9), 0, 1)
    px, py = x1 + t * dx, y1 + t * dy
    return np.hypot(X - px, Y - py)


class FloorField:
    """Direction field towards a set of target cells."""

    def __init__(self, venue: Venue, target_mask: np.ndarray):
        self.venue = venue
        blocked = _rasterise_blocked(venue)
        ny, nx = blocked.shape
        self.nx, self.ny = nx, ny
        idx = np.arange(nx * ny).reshape(ny, nx)
        rows, cols, w = [], [], []
        for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            a = idx[max(0, -dy):ny - max(0, dy), max(0, -dx):nx - max(0, dx)]
            b = idx[max(0, dy):ny + min(0, dy), max(0, dx):nx + min(0, dx)]
            ok = ~blocked.ravel()[a.ravel()] & ~blocked.ravel()[b.ravel()]
            rows.append(a.ravel()[ok])
            cols.append(b.ravel()[ok])
            w.append(np.full(ok.sum(), np.hypot(dx, dy) * RES))
        g = coo_matrix(
            (np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
            shape=(nx * ny, nx * ny),
        ).tocsr()
        targets = np.flatnonzero(target_mask.ravel() & ~blocked.ravel())
        if len(targets) == 0:
            raise ValueError("floor field has no reachable target cells")
        dist = dijkstra(g, directed=False, indices=targets, min_only=True)
        dist = dist.reshape(ny, nx)
        self.dist = dist

        # direction = towards the neighbour with the smallest distance
        big = np.where(np.isfinite(dist), dist, 1e9)
        pad = np.pad(big, 1, constant_values=1e9)
        best = big.copy()
        dirx = np.zeros_like(big)
        diry = np.zeros_like(big)
        for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            nb = pad[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]
            better = nb < best - 1e-9
            best = np.where(better, nb, best)
            n = np.hypot(dx, dy)
            dirx = np.where(better, dx / n, dirx)
            diry = np.where(better, dy / n, diry)
        # smooth the direction field a little so agents do not zig-zag
        from scipy.ndimage import uniform_filter

        sx = uniform_filter(dirx, 3)
        sy = uniform_filter(diry, 3)
        norm = np.hypot(sx, sy)
        norm[norm == 0] = 1
        self.dirx = sx / norm
        self.diry = sy / norm
        self.blocked = blocked

    def direction(self, pos: np.ndarray) -> np.ndarray:
        ix = np.clip((pos[:, 0] / RES).astype(int), 0, self.nx - 1)
        iy = np.clip((pos[:, 1] / RES).astype(int), 0, self.ny - 1)
        return np.stack([self.dirx[iy, ix], self.diry[iy, ix]], axis=1)

    def distance(self, pos: np.ndarray) -> np.ndarray:
        ix = np.clip((pos[:, 0] / RES).astype(int), 0, self.nx - 1)
        iy = np.clip((pos[:, 1] / RES).astype(int), 0, self.ny - 1)
        return self.dist[iy, ix]


def front_field(venue: Venue) -> FloorField:
    """Everyone wants to get close to the stage front barrier."""
    nx, ny = int(venue.width / RES), int(venue.height / RES)
    xs = (np.arange(nx) + 0.5) * RES
    ys = (np.arange(ny) + 0.5) * RES
    X, Y = np.meshgrid(xs, ys)
    x0, _, x1, _ = venue.stage
    mask = (Y > venue.front_line_y - 1.5) & (Y < venue.front_line_y - 0.5) & (X > x0 - 5) & (X < x1 + 5)
    return FloorField(venue, mask)


def exit_field(venue: Venue) -> FloorField:
    """Shortest walking direction to the nearest open gate."""
    nx, ny = int(venue.width / RES), int(venue.height / RES)
    mask = np.zeros((ny, nx), dtype=bool)
    for g in venue.open_gates():
        x1, y1, x2, y2 = g.seg
        if y1 == y2:  # gate on a horizontal wall
            ix0, ix1 = int(min(x1, x2) / RES), int(max(x1, x2) / RES)
            iy = 0 if y1 == 0 else ny - 1
            mask[iy, ix0:ix1] = True
        else:
            iy0, iy1 = int(min(y1, y2) / RES), int(max(y1, y2) / RES)
            ix = 0 if x1 == 0 else nx - 1
            mask[iy0:iy1, ix] = True
    return FloorField(venue, mask)
