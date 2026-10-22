"""TwinEngine: one tick = sensors -> fusion -> risk -> prediction -> drone -> evacuation.

The engine is used both by the live backend (one tick every 1-2 s) and by the
experiments (run through a whole scenario as fast as possible).
"""
from __future__ import annotations

import time

import numpy as np

from simulation.sensors.simulated import SensorSuite

from .blindspot import POLICIES, blind_spots
from .evacuation import Evacuation
from .fusion import ENGINES, KalmanFusion
from .predict import RateTracker
from .risk import CRITICAL, LEVELS, score


class TwinEngine:
    def __init__(self, trajectory, fusion="kalman", drone_policy="active", seed=0,
                 sources=("cctv", "drone", "tower", "wifi"), evacuation=True):
        self.tr = trajectory
        self.venue = trajectory.venue
        self.sensors = SensorSuite(trajectory, seed=seed, use=sources)
        self.fusion = ENGINES[fusion](self.venue)
        rng = np.random.default_rng(seed + 7)
        self.policy = POLICIES[drone_policy](self.venue, self.sensors.drone, rng) if "drone" in sources else None
        self.rates = RateTracker(self.venue.n_zones)
        self.evac = Evacuation(self.venue) if evacuation else None
        self.k = 0
        self.walk = self.venue.zone_walkable.ravel()
        self.last_ms = []
        self.tick_ms = 0.0

    @property
    def t(self) -> float:
        return self.tr.times[self.k]

    def reset(self, **kw):
        self.__init__(self.tr, **kw)

    def step(self) -> dict:
        """Advance to the next snapshot and return the full twin state."""
        t0 = time.perf_counter()
        dt = self.tr.times[self.k] - self.tr.times[self.k - 1] if self.k > 0 else 0.0
        ms = self.sensors.observe(self.k, dt)
        self.fusion.step(ms, dt)
        dens = self.fusion.density()
        std = self.fusion.density_std()
        self.rates.update(dens, dt)
        pressure = dens * self.fusion.flow[:, 2]
        risk = score(dens, std, pressure)
        if self.policy is not None and isinstance(self.fusion, KalmanFusion):
            self.policy.maybe_plan(self.t, self.fusion, risk["upper"])
        elif self.policy is not None:
            self.policy.maybe_plan(self.t, _NoStd(self.fusion), risk["upper"])
        ttc = self.rates.time_to_critical(dens)
        self.last_ms = ms
        self.state = {
            "t": self.t,
            "density": dens,
            "std": std,
            "pressure": pressure,
            "risk": risk,
            "ttc": ttc,
            "blind": blind_spots(std, self.walk),
        }
        self.tick_ms = (time.perf_counter() - t0) * 1000
        self.k += 1
        return self.state

    def done(self) -> bool:
        return self.k >= len(self.tr)

    # -- payload for the dashboard ----------------------------------------
    def snapshot(self, with_truth: bool = True) -> dict:
        s = self.state
        v = self.venue
        counts = self.fusion.counts()
        crit = [int(z) for z in np.flatnonzero((s["density"] >= CRITICAL) & self.walk)]
        pred = [int(z) for z in np.flatnonzero(np.isfinite(s["ttc"]) & (s["ttc"] > 0) & self.walk)]
        alerts = []
        for z in crit:
            alerts.append({"level": "critical", "zone": v.zone_name(z), "zone_id": z,
                           "text": f"{v.zone_name(z)} is critical: {s['density'][z]:.1f} persons/m²"})
        for z in sorted(pred, key=lambda z: s["ttc"][z]):
            alerts.append({"level": "warning", "zone": v.zone_name(z), "zone_id": z,
                           "text": f"{v.zone_name(z)} will turn critical in {_eta(s['ttc'][z])} "
                                   f"({s['density'][z]:.1f} persons/m² now)"})
        for z in np.flatnonzero(s["risk"]["unverified"]):
            alerts.append({"level": "unverified", "zone": v.zone_name(z), "zone_id": int(z),
                           "text": f"{v.zone_name(z)} could be {s['risk']['upper'][z]:.1f} persons/m², send the drone"})
        evac = None
        if self.evac is not None:
            gates_now = {g.name: g.is_open for g in v.gates}
            plan = self.evac.recommend(counts, s["density"], crit, pred)
            evac = {
                "next": {int(k): (int(n) if n is not None else None) for k, n in plan["next"].items()},
                "load": {k: round(x) for k, x in plan["load"].items()},
                "clearance_min": {k: round(x / 60, 1) for k, x in plan["clearance_s"].items()},
                "actions": plan["actions"],
                "gates_open": gates_now,
            }
        drone = self.sensors.drone
        out = {
            "t": round(s["t"], 1),
            "phase": self.tr.scenario.phase(s["t"]),
            "estimated_total": int(round(counts.sum())),
            "density": np.round(s["density"], 2).tolist(),
            "std": np.round(s["std"], 2).tolist(),
            "confidence": np.round(np.exp(-s["std"] / 0.5), 2).tolist(),
            "level": s["risk"]["level"].astype(int).tolist(),
            "blind": s["blind"].astype(int).tolist(),
            "flow": np.round(self.fusion.flow[:, :2], 2).tolist(),
            "ttc": [None if not np.isfinite(x) else round(float(x)) for x in s["ttc"]],
            "alerts": alerts[:30],
            "drone": {"pos": np.round(drone.pos, 1).tolist(), "target": np.round(drone.target, 1).tolist(),
                      "reason": getattr(self.policy, "reason", ""),
                      "footprint": drone.footprint_zones().tolist()},
            "sources": [st.to_dict() for st in self.sensors.statuses()],
            "evacuation": evac,
            "tick_ms": round(self.tick_ms, 1),
            "levels": LEVELS,
        }
        if with_truth:
            truth = self.sensors.truth(self.k - 1)
            out["truth_density"] = np.round(truth["density"], 2).tolist()
            out["true_total"] = int(truth["count"].sum())
        return out


def _eta(seconds: float) -> str:
    return f"about {seconds:.0f} s" if seconds < 90 else f"about {seconds / 60:.0f} min"


class _NoStd:
    """Engines without uncertainty: the active drone policy just sees zero variance."""

    def __init__(self, f):
        self.f = f

    def density_std(self):
        return np.zeros(self.f.venue.n_zones)
