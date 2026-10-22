"""Blind-spot detection and drone guidance (the core contribution).

The Kalman fusion keeps an uncertainty for every zone. Zones nobody has looked
at for a while become uncertain. The active policy sends the drone to the spot
where (uncertainty x possible risk) is highest, discounted by travel time.

Baselines for comparison: a fixed lawn-mower patrol, random waypoints, and a
drone hovering at the centre.
"""
from __future__ import annotations

import numpy as np

BLIND_STD = 0.3   # persons/m2: above this a zone counts as a blind spot


def blind_spots(std_density: np.ndarray, walkable: np.ndarray) -> np.ndarray:
    return (std_density > BLIND_STD) & walkable


class _Policy:
    replan_every = 10.0

    def __init__(self, venue, drone, rng):
        self.venue, self.drone, self.rng = venue, drone, rng
        self.next_plan = 0.0
        c = venue.cell
        # candidate drone positions: centres of 5x5-zone footprints, on a zone grid
        xs = np.arange(2.5 * c, venue.width - 2.5 * c + 1e-6, c)
        ys = np.arange(2.5 * c, venue.height - 2.5 * c + 1e-6, c)
        self.cands = np.array([(x, y) for y in ys for x in xs])
        self.reason = ""

    def plan(self, t, fusion, risk_upper):
        raise NotImplementedError

    def maybe_plan(self, t, fusion, risk_upper):
        if t + 1e-9 >= self.next_plan:
            self.next_plan = t + self.replan_every
            self.plan(t, fusion, risk_upper)


class ActiveGuidance(_Policy):
    name = "active"

    def __init__(self, venue, drone, rng, risk_weight=2.0, travel_scale=30.0):
        super().__init__(venue, drone, rng)
        self.risk_weight, self.travel_scale = risk_weight, travel_scale
        self._fp = [self.drone.footprint_zones(p) for p in self.cands]

    def plan(self, t, fusion, risk_upper):
        var = fusion.density_std() ** 2
        weight = 1.0 + self.risk_weight * np.clip(risk_upper / 6.0, 0, 2)
        value = var * weight
        scores = np.array([value[z].sum() for z in self._fp])
        travel = np.linalg.norm(self.cands - self.drone.pos, axis=1) / self.drone.speed
        scores = scores / (1.0 + travel / self.travel_scale)
        best = int(np.argmax(scores))
        self.drone.target = self.cands[best].copy()
        zone = self.venue.zone_name(self.venue.zone_index(np.array([self.cands[best][0]]), np.array([self.cands[best][1]]))[0])
        self.reason = f"largest blind spot near {zone}"


class FixedPatrol(_Policy):
    name = "patrol"

    def __init__(self, venue, drone, rng):
        super().__init__(venue, drone, rng)
        c = venue.cell
        xs = np.arange(2.5 * c, venue.width - 2.5 * c + 1e-6, 5 * c)
        ys = np.arange(2.5 * c, venue.height - 2.5 * c + 1e-6, 5 * c)
        pts = []
        for i, y in enumerate(ys):
            row = xs if i % 2 == 0 else xs[::-1]
            pts += [(x, y) for x in row]
        self.waypoints = np.array(pts)
        self.i = 0

    def plan(self, t, fusion, risk_upper):
        if np.linalg.norm(self.drone.pos - self.drone.target) < 1.0:
            self.i = (self.i + 1) % len(self.waypoints)
        self.drone.target = self.waypoints[self.i].copy()
        self.reason = "fixed patrol route"


class RandomWaypoint(_Policy):
    name = "random"

    def plan(self, t, fusion, risk_upper):
        if np.linalg.norm(self.drone.pos - self.drone.target) < 1.0:
            self.drone.target = self.cands[self.rng.integers(len(self.cands))].copy()
        self.reason = "random waypoint"


class Hover(_Policy):
    name = "hover"

    def plan(self, t, fusion, risk_upper):
        self.drone.target = np.array([self.venue.width / 2, self.venue.height / 2])
        self.reason = "hovering at centre"


POLICIES = {"active": ActiveGuidance, "patrol": FixedPatrol, "random": RandomWaypoint, "hover": Hover}
