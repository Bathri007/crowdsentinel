"""Map camera pixels onto the ground plane of the venue (metres).

Calibrate once per fixed camera: pick at least 4 points you can see in the
image AND locate on the venue map (barrier corners, gate posts, light poles),
write them in a JSON file and load it here.

    {
      "image_points": [[102, 655], [1180, 640], [880, 210], [330, 215]],
      "ground_points": [[20, 30], [60, 30], [60, 55], [20, 55]]
    }
"""
from __future__ import annotations

import json

import cv2
import numpy as np


class GroundMapper:
    def __init__(self, image_points, ground_points):
        src = np.asarray(image_points, dtype=np.float64)
        dst = np.asarray(ground_points, dtype=np.float64)
        if len(src) < 4 or len(src) != len(dst):
            raise ValueError("need at least 4 matching image/ground points")
        self.H, _ = cv2.findHomography(src, dst, 0)
        if self.H is None:
            raise ValueError("homography could not be computed - are the points collinear?")

    @classmethod
    def from_json(cls, path: str) -> "GroundMapper":
        with open(path) as f:
            d = json.load(f)
        return cls(d["image_points"], d["ground_points"])

    def to_ground(self, pts_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def reprojection_error(self, image_points, ground_points) -> float:
        g = self.to_ground(np.asarray(image_points))
        return float(np.linalg.norm(g - np.asarray(ground_points), axis=1).mean())

    def density_to_zones(self, dmap: np.ndarray, frame_shape, venue) -> np.ndarray:
        """Sum a density map (people per map cell) into venue zones -> people per zone.

        Counts are additive, so we just move each cell's people to the zone its
        centre lands in. Cells that land outside the venue are dropped.
        """
        ch, cw = frame_shape[0] / dmap.shape[0], frame_shape[1] / dmap.shape[1]
        rr, cc = np.mgrid[0:dmap.shape[0], 0:dmap.shape[1]]
        centres = np.stack([(cc.ravel() + 0.5) * cw, (rr.ravel() + 0.5) * ch], axis=1)
        g = self.to_ground(centres)
        ok = (g[:, 0] >= 0) & (g[:, 0] < venue.width) & (g[:, 1] >= 0) & (g[:, 1] < venue.height)
        zones = venue.zone_index(g[ok, 0], g[ok, 1])
        return np.bincount(zones, weights=dmap.ravel()[ok], minlength=venue.n_zones)

    def visible_zones(self, frame_shape, venue, step: int = 16) -> np.ndarray:
        """Zones the camera can see (at least a few sample pixels land in them)."""
        h, w = frame_shape[:2]
        ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
        g = self.to_ground(np.stack([xs.ravel(), ys.ravel()], axis=1))
        ok = (g[:, 0] >= 0) & (g[:, 0] < venue.width) & (g[:, 1] >= 0) & (g[:, 1] < venue.height)
        z, n = np.unique(venue.zone_index(g[ok, 0], g[ok, 1]), return_counts=True)
        walk = venue.zone_walkable.ravel()
        return z[(n >= 4) & walk[z]]
