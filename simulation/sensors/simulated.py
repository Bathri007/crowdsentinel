"""Simulated sensors that observe the ground-truth crowd with realistic errors.

Error models (all documented so the evaluation is reproducible):

CCTV   - density-model style count per zone, 8 % random error, and a systematic
         undercount in very dense crowds (occlusion): -4 % per person/m2 above 3.
         Cameras occasionally drop out for 60 s.
Drone  - top-down view of a 25 m x 25 m footprint, 12 % error, moves at 6 m/s.
Tower  - number of phones in a sector (85 % of people carry a phone), plus
         background phones of passers-by, reported every 10 s with a 10 s delay.
WiFi   - devices connected to a hotspot (30 % of people connect), radius 15 m,
         reported every 6 s.
"""
from __future__ import annotations

import numpy as np

from twin.ground_truth import zone_truth

from .base import Measurement, SourceStatus, zones_in_circle, zones_in_rect


class CCTVCamera:
    def __init__(self, name, venue, zones, rng, outage_prob=0.002, outage_len=60.0):
        self.name, self.venue, self.zones, self.rng = name, venue, zones, rng
        self.outage_prob, self.outage_len = outage_prob, outage_len
        self.down_until = -1.0
        self.status = SourceStatus(name, "cctv", coverage=zones.tolist())

    def observe(self, t, truth) -> list[Measurement]:
        if t < self.down_until:
            self.status.online = False
            return []
        if self.rng.random() < self.outage_prob:
            self.down_until = t + self.outage_len
            self.status.online = False
            return []
        self.status.online = True
        self.status.last_seen = t
        out = []
        for z in self.zones:
            true = truth["count"][z]
            dens = truth["density"][z]
            occl = 0.04 * max(0.0, dens - 3.0)
            val = max(0.0, true * (1 + self.rng.normal(0, 0.08)) * (1 - occl))
            var = (0.10 * max(val, 2.0)) ** 2
            out.append(Measurement(self.name, "cctv", np.array([z]), val, var, t,
                                   flow=(truth["vx"][z], truth["vy"][z], truth["speed_var"][z])))
        return out


class Drone:
    """A drone whose camera footprint can be steered by a guidance policy."""

    def __init__(self, venue, rng, start=None, speed=6.0, footprint=25.0):
        self.venue, self.rng = venue, rng
        self.pos = np.array(start if start is not None else (venue.width / 2, venue.height / 2), float)
        self.target = self.pos.copy()
        self.speed, self.footprint = speed, footprint
        self.status = SourceStatus("DRONE", "drone")

    def move(self, dt):
        d = self.target - self.pos
        dist = np.linalg.norm(d)
        step = self.speed * dt
        self.pos = self.target.copy() if dist <= step else self.pos + d / dist * step

    def footprint_zones(self, pos=None) -> np.ndarray:
        x, y = self.pos if pos is None else pos
        h = self.footprint / 2
        centers = self.venue.zone_centers()
        inside = (np.abs(centers[:, 0] - x) < h) & (np.abs(centers[:, 1] - y) < h)
        return np.flatnonzero(inside & self.venue.zone_walkable.ravel())

    def observe(self, t, truth) -> list[Measurement]:
        zones = self.footprint_zones()
        self.status.last_seen = t
        self.status.coverage = zones.tolist()
        out = []
        for z in zones:
            true = truth["count"][z]
            val = max(0.0, true * (1 + self.rng.normal(0, 0.12)))
            var = (0.12 * max(val, 2.0)) ** 2
            out.append(Measurement("DRONE", "drone", np.array([z]), val, var, t,
                                   flow=(truth["vx"][z], truth["vy"][z], truth["speed_var"][z])))
        return out


class CellTower:
    def __init__(self, name, zones, rng, period=10.0, delay=10.0, phone_rate=0.85, background=40):
        self.name, self.zones, self.rng = name, zones, rng
        self.period, self.delay = period, delay
        self.p, self.bg = phone_rate, background
        self.next_t = 0.0
        self.status = SourceStatus(name, "tower", coverage=zones.tolist())

    def observe(self, t, truth_at) -> list[Measurement]:
        if t + 1e-9 < self.next_t:
            return []
        self.next_t = t + self.period
        t_obs = max(0.0, t - self.delay)
        truth = truth_at(t_obs)
        n = truth["count"][self.zones].sum()
        phones = self.rng.binomial(int(n), self.p) + self.rng.poisson(self.bg)
        val = max(0.0, (phones - self.bg) / self.p * (1 + self.rng.normal(0, 0.03)))
        var = n * (1 - self.p) / self.p + self.bg / self.p ** 2 + (0.05 * max(n, 10)) ** 2
        self.status.last_seen = t
        return [Measurement(self.name, "tower", self.zones, val, var, t_obs)]


class WifiHotspot:
    def __init__(self, name, zones, rng, period=6.0, connect_rate=0.3):
        self.name, self.zones, self.rng = name, zones, rng
        self.period, self.p = period, connect_rate
        self.next_t = 0.0
        self.status = SourceStatus(name, "wifi", coverage=zones.tolist())

    def observe(self, t, truth) -> list[Measurement]:
        if t + 1e-9 < self.next_t:
            return []
        self.next_t = t + self.period
        n = truth["count"][self.zones].sum()
        devices = self.rng.binomial(int(n), self.p)
        val = devices / self.p
        var = n * (1 - self.p) / self.p + (0.10 * max(n, 10)) ** 2 + 4
        self.status.last_seen = t
        return [Measurement(self.name, "wifi", self.zones, val, var, t)]


def default_layout(venue, seed=0):
    """Sensor layout for the rally ground.

    The cameras cover the stage centre/right, the fence gap and the middle gate.
    The front-left of the stage has no camera - a realistic blind spot, which is
    what the drone guidance has to discover.
    """
    rng = np.random.default_rng(seed + 1000)
    cams = [
        CCTVCamera("CAM1", venue, zones_in_rect(venue, 9, 11, 7, 12), rng),   # front centre
        CCTVCamera("CAM2", venue, zones_in_rect(venue, 9, 11, 13, 17), rng),  # front right
        CCTVCamera("CAM3", venue, zones_in_rect(venue, 4, 6, 7, 12), rng),    # fence gap
        CCTVCamera("CAM4", venue, zones_in_rect(venue, 0, 2, 8, 11), rng),    # gate G2
    ]
    third = venue.cols // 3
    towers = [
        CellTower("TOWER-W", zones_in_rect(venue, 0, venue.rows - 1, 0, third - 1), rng),
        CellTower("TOWER-C", zones_in_rect(venue, 0, venue.rows - 1, third, 2 * third), rng),
        CellTower("TOWER-E", zones_in_rect(venue, 0, venue.rows - 1, 2 * third + 1, venue.cols - 1), rng),
    ]
    wifi = [
        WifiHotspot("WIFI-G1", zones_in_circle(venue, 13, 6, 15), rng),
        WifiHotspot("WIFI-G2", zones_in_circle(venue, 49, 6, 15), rng),
        WifiHotspot("WIFI-G3", zones_in_circle(venue, 87, 6, 15), rng),
        WifiHotspot("WIFI-STAGE", zones_in_circle(venue, 50, 55, 15), rng),
    ]
    drone = Drone(venue, rng)
    return cams, drone, towers, wifi


class SensorSuite:
    """All simulated sources, observing one trajectory over time."""

    def __init__(self, trajectory, seed=0, use=("cctv", "drone", "tower", "wifi")):
        self.tr = trajectory
        self.venue = trajectory.venue
        self.cams, self.drone, self.towers, self.wifi = default_layout(self.venue, seed)
        self.use = set(use)
        self._truth_cache: dict[int, dict] = {}

    def truth(self, k: int) -> dict:
        if k not in self._truth_cache:
            self._truth_cache[k] = zone_truth(self.venue, self.tr.pos[k], self.tr.vel[k])
        return self._truth_cache[k]

    def truth_at(self, t: float) -> dict:
        times = self.tr.times
        k = int(np.clip(np.searchsorted(times, t + 1e-9) - 1, 0, len(times) - 1))
        return self.truth(k)

    def statuses(self) -> list[SourceStatus]:
        out = []
        if "cctv" in self.use:
            out += [c.status for c in self.cams]
        if "drone" in self.use:
            out.append(self.drone.status)
        if "tower" in self.use:
            out += [t.status for t in self.towers]
        if "wifi" in self.use:
            out += [w.status for w in self.wifi]
        return out

    def observe(self, k: int, dt: float) -> list[Measurement]:
        t = self.tr.times[k]
        truth = self.truth(k)
        ms: list[Measurement] = []
        if "cctv" in self.use:
            for c in self.cams:
                ms += c.observe(t, truth)
        if "drone" in self.use:
            self.drone.move(dt)
            ms += self.drone.observe(t, truth)
        if "tower" in self.use:
            for tw in self.towers:
                ms += tw.observe(t, self.truth_at)
        if "wifi" in self.use:
            for w in self.wifi:
                ms += w.observe(t, truth)
        return ms

    def cctv_coverage(self) -> np.ndarray:
        if not self.cams:
            return np.array([], dtype=int)
        return np.unique(np.concatenate([c.zones for c in self.cams]))
