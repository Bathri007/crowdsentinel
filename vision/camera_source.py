"""Real CCTV / drone video -> the twin's Measurement records.

This is the ingestion adapter for real cameras. It reads a video file or an
RTSP stream, runs density estimation + optical flow about twice per second,
projects the result onto the venue zones with the camera's homography, and
emits the same Measurement records the simulated cameras produce. The fusion
engine cannot tell the difference.

    src = VideoCameraSource("CAM1", "data/raw/front.mp4", "data/calib/cam1.json", venue, estimator)
    for t, measurements in src.stream():
        ...
"""
from __future__ import annotations

import cv2
import numpy as np

from simulation.sensors.base import Measurement, SourceStatus

from .flow import FlowEstimator
from .homography import GroundMapper


class VideoCameraSource:
    def __init__(self, name, url_or_path, calib_json, venue, estimator, rate_hz=2.0, rel_error=0.10):
        self.name, self.url = name, url_or_path
        self.venue, self.est = venue, estimator
        self.mapper = calib_json if isinstance(calib_json, GroundMapper) else GroundMapper.from_json(calib_json)
        self.rate_hz, self.rel_error = rate_hz, rel_error
        self.flow = FlowEstimator()
        self.status = SourceStatus(name, "cctv", real_data=True)
        self.zones = None

    def process_frame(self, frame, t) -> list[Measurement]:
        if self.zones is None:
            self.zones = self.mapper.visible_zones(frame.shape, self.venue)
            self.status.coverage = self.zones.tolist()
        dmap = self.est.predict(frame)
        per_zone = self.mapper.density_to_zones(dmap, frame.shape, self.venue)
        fl = self.flow.update(frame, t)
        out = []
        for z in self.zones:
            val = float(per_zone[z])
            var = (self.rel_error * max(val, 2.0)) ** 2
            out.append(Measurement(self.name, "cctv", np.array([z]), val, var, t, flow=None))
        self.status.online, self.status.last_seen = True, t
        if fl is not None:
            # image-plane motion -> ground-plane velocity per analysis cell, assigned to zones
            mx, my, var = fl
            R, C = mx.shape
            h, w = frame.shape[:2]
            cy, cx = np.mgrid[0:R, 0:C]
            px = np.stack([(cx.ravel() + 0.5) * w / C, (cy.ravel() + 0.5) * h / R], axis=1)
            g0 = self.mapper.to_ground(px)
            g1 = self.mapper.to_ground(px + np.stack([mx.ravel(), my.ravel()], axis=1))
            v = g1 - g0                           # metres per second on the ground
            ok = (g0[:, 0] >= 0) & (g0[:, 0] < self.venue.width) & (g0[:, 1] >= 0) & (g0[:, 1] < self.venue.height)
            zid = self.venue.zone_index(g0[ok, 0], g0[ok, 1])
            # pixel variance -> rough metric variance using the local scale of the homography
            scale = np.linalg.norm(v[ok], axis=1) / np.maximum(np.hypot(mx.ravel()[ok], my.ravel()[ok]), 1e-6)
            mvar = var.ravel()[ok] * scale ** 2
            by_zone = {}
            for z, vv, s2 in zip(zid, v[ok], mvar):
                by_zone.setdefault(int(z), []).append((vv[0], vv[1], s2))
            for m in out:
                vals = by_zone.get(int(m.zones[0]))
                if vals:
                    a = np.array(vals)
                    m.flow = (float(a[:, 0].mean()), float(a[:, 1].mean()), float(a[:, 2].mean()))
        return out

    def stream(self, max_seconds: float | None = None):
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            self.status.online = False
            raise IOError(f"cannot open video source {self.url}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        every = max(1, int(round(fps / self.rate_hz)))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            if max_seconds is not None and t > max_seconds:
                break
            if i % every == 0:
                yield t, self.process_frame(frame, t)
            i += 1
        cap.release()
        self.status.online = False
