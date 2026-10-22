"""Crowd movement from video: dense optical flow (Farneback) per grid cell.

Gives, for every cell of a grid over the image: mean motion (dx, dy) in pixels
per second, and the variance of motion inside the cell. Crowd pressure
(Helbing et al., 2007) = density x velocity variance; high values mean
turbulent, pushing movement, which appears before crushes.
"""
from __future__ import annotations

import cv2
import numpy as np


class FlowEstimator:
    def __init__(self, grid: tuple[int, int] = (12, 16), work_width: int = 480):
        self.grid = grid              # (rows, cols) of the analysis grid
        self.work_width = work_width  # frames are downscaled to this width for speed
        self.prev = None
        self.prev_t = None

    def _gray(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        s = self.work_width / w
        small = cv2.resize(frame_bgr, (self.work_width, int(h * s)), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), s

    def update(self, frame_bgr: np.ndarray, t: float):
        """Returns per-cell (mean_dx, mean_dy, var) in ORIGINAL pixels/second, or None for the first frame."""
        gray, s = self._gray(frame_bgr)
        if self.prev is None or t <= self.prev_t:
            self.prev, self.prev_t = gray, t
            return None
        flow = cv2.calcOpticalFlowFarneback(self.prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dt = t - self.prev_t
        self.prev, self.prev_t = gray, t
        flow = flow / (s * dt)       # -> original pixels per second
        return cell_stats(flow, self.grid)


def cell_stats(flow: np.ndarray, grid: tuple[int, int]):
    h, w = flow.shape[:2]
    R, C = grid
    mx = np.zeros((R, C))
    my = np.zeros((R, C))
    var = np.zeros((R, C))
    for r in range(R):
        for c in range(C):
            blk = flow[r * h // R:(r + 1) * h // R, c * w // C:(c + 1) * w // C].reshape(-1, 2)
            m = blk.mean(0)
            mx[r, c], my[r, c] = m
            var[r, c] = ((blk - m) ** 2).sum(1).mean()
    return mx, my, var


def crowd_pressure(density: np.ndarray, velocity_var: np.ndarray) -> np.ndarray:
    """Helbing crowd pressure = local density x local velocity variance."""
    return density * velocity_var
