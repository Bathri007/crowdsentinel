"""Fusion engines: turn many noisy, partial measurements into one live estimate.

Three engines share one interface, so experiments can compare them fairly:

- CCTVOnly        : what a typical camera dashboard does. Zones without a camera
                    keep their last known value.
- WeightedFusion  : simple confidence-weighted average (roadmap "version 1").
- KalmanFusion    : one Kalman filter over all zones (roadmap "version 2").
                    Its uncertainty grows wherever nobody is looking, which is
                    what the blind-spot module uses to steer the drone.

All engines estimate people COUNT per zone; density = count / zone area.
"""
from __future__ import annotations

import numpy as np

from simulation.sensors.base import Measurement
from simulation.venue import Venue


class _Base:
    name = "base"

    def __init__(self, venue: Venue):
        self.venue = venue
        self.walk = np.flatnonzero(venue.zone_walkable.ravel())       # zone ids of walkable zones
        self.pos_of = -np.ones(venue.n_zones, dtype=int)
        self.pos_of[self.walk] = np.arange(len(self.walk))
        self.n = len(self.walk)
        self.x = np.zeros(self.n)
        # latest flow seen per zone: vx, vy, speed variance, time seen
        self.flow = np.zeros((venue.n_zones, 3))
        self.flow_t = np.full(venue.n_zones, -1e9)

    # -- helpers ------------------------------------------------------
    def _idx(self, zones):
        p = self.pos_of[zones]
        return p[p >= 0]

    def _update_flow(self, ms: list[Measurement]):
        for m in ms:
            if m.flow is not None and len(m.zones) == 1:
                z = m.zones[0]
                self.flow[z] = m.flow
                self.flow_t[z] = m.t

    # -- outputs ------------------------------------------------------
    def counts(self) -> np.ndarray:
        out = np.zeros(self.venue.n_zones)
        out[self.walk] = np.maximum(self.x, 0)
        return out

    def density(self) -> np.ndarray:
        return self.counts() / self.venue.zone_area

    def density_std(self) -> np.ndarray:
        return np.zeros(self.venue.n_zones)

    def step(self, ms: list[Measurement], dt: float):
        raise NotImplementedError


class CCTVOnly(_Base):
    name = "cctv_only"

    def step(self, ms, dt):
        self._update_flow(ms)
        for m in ms:
            if m.kind == "cctv":
                p = self._idx(m.zones)
                if len(p) == 1:
                    self.x[p[0]] = m.value


class WeightedFusion(_Base):
    """Inverse-variance average of direct measurements; regional counts
    (tower, WiFi) are spread evenly over the zones nobody measured directly."""
    name = "weighted"

    def step(self, ms, dt):
        self._update_flow(ms)
        num = np.zeros(self.n)
        den = np.zeros(self.n)
        for m in ms:
            p = self._idx(m.zones)
            if len(p) == 1:
                num[p[0]] += m.value / m.var
                den[p[0]] += 1.0 / m.var
        direct = den > 0
        new = self.x.copy()
        new[direct] = num[direct] / den[direct]
        share_num = np.zeros(self.n)
        share_cnt = np.zeros(self.n)
        for m in ms:
            p = self._idx(m.zones)
            if len(p) > 1:
                known = direct[p]
                rest = p[~known]
                if len(rest) == 0:
                    continue
                remainder = max(0.0, m.value - new[p[known]].sum())
                share_num[rest] += remainder / len(rest)
                share_cnt[rest] += 1
        reg = (share_cnt > 0) & ~direct
        new[reg] = 0.5 * self.x[reg] + 0.5 * share_num[reg] / share_cnt[reg]
        self.x = new


class KalmanFusion(_Base):
    """Linear Kalman filter over people counts in every walkable zone.

    Process model : random walk with spatially correlated noise (neighbouring
                    zones tend to change together), noise grows with crowd size.
    Measurements  : each record says sum(x[zones]) = value + noise(var).
    """
    name = "kalman"

    def __init__(self, venue, q_base=0.7, q_rel=0.03, corr_len=5.0, init_var=1.0):
        super().__init__(venue)
        self.q_base, self.q_rel = q_base, q_rel
        self.P = np.eye(self.n) * init_var
        # Spatially correlated process noise: C = G G^T is positive semi-definite
        # by construction (a truncated Gaussian matrix is not, and makes the filter blow up).
        c = venue.zone_centers()[self.walk]
        d2 = ((c[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        G = np.exp(-d2 / corr_len ** 2)
        C = G @ G.T
        dg = np.sqrt(np.diag(C))
        self.C = C / dg[:, None] / dg[None, :]

    def predict(self, dt):
        s = np.sqrt(self.q_base ** 2 + (self.q_rel * np.maximum(self.x, 0)) ** 2)
        Q = (s[:, None] * self.C * s[None, :]) * dt
        self.P = self.P + Q

    def update(self, ms: list[Measurement]):
        rows, vals, rvar = [], [], []
        for m in ms:
            p = self._idx(m.zones)
            if len(p) == 0:
                continue
            rows.append(p)
            vals.append(m.value)
            # delayed reports are less certain about "now"
            lag = max(0.0, self._now - m.t)
            rvar.append(m.var + len(p) * self.q_base ** 2 * lag)
        if not rows:
            return
        H = np.zeros((len(rows), self.n))
        for i, p in enumerate(rows):
            H[i, p] = 1.0
        z = np.array(vals)
        R = np.diag(rvar)
        PHt = self.P @ H.T
        S = H @ PHt + R
        K = np.linalg.solve(S, PHt.T).T
        self.x = self.x + K @ (z - H @ self.x)
        # Joseph form keeps P symmetric positive-definite
        IKH = np.eye(self.n) - K @ H
        self.P = IKH @ self.P @ IKH.T + (K * np.asarray(rvar)) @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        # note: x is NOT clipped at zero here - clipping inside the filter adds
        # people every tick and biases the estimate upwards. counts() clips the output.

    def step(self, ms, dt):
        self._now = max([m.t for m in ms], default=getattr(self, "_now", 0.0))
        self._update_flow(ms)
        self.predict(dt)
        self.update(ms)

    def density_std(self) -> np.ndarray:
        out = np.zeros(self.venue.n_zones)
        out[self.walk] = np.sqrt(np.maximum(np.diag(self.P), 0)) / self.venue.zone_area
        return out

    def confidence(self) -> np.ndarray:
        """0..1 per zone. 1 = we know the density well, 0 = we are guessing."""
        return np.exp(-self.density_std() / 0.5)


ENGINES = {"cctv_only": CCTVOnly, "weighted": WeightedFusion, "kalman": KalmanFusion}
