"""Social-force crowd simulation of a rally, used as the ground truth for the twin.

Because we simulate every person, we know the true density everywhere at every
moment. The sensor generators then "observe" this crowd with realistic noise,
gaps and delays, and we can measure how close the twin gets to the truth.

Model: Helbing & Molnar social force (driving force + exponential repulsion
+ body compression), with floor fields for route choice.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .floorfield import FloorField, exit_field, front_field
from .venue import Venue, rally_ground


@dataclass
class Scenario:
    """Timeline of a rally. Times are in seconds from gates opening."""
    n_people: int = 4000
    arrival_end: float = 360.0      # people keep arriving until this time
    surge_start: float = 480.0      # crowd pushes towards the stage
    surge_end: float = 600.0
    surge_ramp: float = 60.0        # seconds for the surge push to build up
    dispersal: float = 690.0        # event ends, everyone heads to the exits
    duration: float = 870.0
    gate4_open: bool = False
    settle_speed: float = 0.25      # gentle forward push while watching (m/s)
    surge_speed: float = 1.5        # forward push during the surge (m/s)

    def phase(self, t: float) -> str:
        if t < self.arrival_end:
            return "arrival"
        if self.surge_start <= t < self.surge_end:
            return "surge"
        if t >= self.dispersal:
            return "dispersal"
        return "watching"


@dataclass
class Trajectory:
    """Recorded snapshots: every `record_every` seconds, positions and velocities."""
    venue: Venue
    scenario: Scenario
    times: list[float] = field(default_factory=list)
    pos: list[np.ndarray] = field(default_factory=list)
    vel: list[np.ndarray] = field(default_factory=list)

    def save(self, path: str):
        counts = np.array([len(p) for p in self.pos])
        np.savez_compressed(
            path,
            times=np.array(self.times),
            counts=counts,
            pos=np.concatenate(self.pos) if self.pos else np.zeros((0, 2)),
            vel=np.concatenate(self.vel) if self.vel else np.zeros((0, 2)),
            scenario=np.array([repr(self.scenario.__dict__)]),
        )

    @classmethod
    def load(cls, path: str, venue: Venue, scenario: Scenario) -> "Trajectory":
        d = np.load(path)
        tr = cls(venue, scenario)
        offs = np.concatenate([[0], np.cumsum(d["counts"])])
        pos, vel = d["pos"], d["vel"]          # read each array once
        for i, t in enumerate(d["times"]):
            tr.times.append(float(t))
            tr.pos.append(pos[offs[i]:offs[i + 1]])
            tr.vel.append(vel[offs[i]:offs[i + 1]])
        return tr

    def __len__(self):
        return len(self.times)


def _seg_arrays(segs):
    a = np.array(segs, dtype=float)
    return a[:, 0:2], a[:, 2:4]


class CrowdSim:
    # social-force parameters (per unit mass)
    TAU = 0.5          # relaxation time (s)
    A = 25.0           # repulsion strength (m/s^2)  ~ 2000 N / 80 kg
    B = 0.08           # repulsion range (m)
    K_BODY = 750.0     # body compression (1/s^2)
    DAMP = 4.0         # contact damping (1/s), keeps dense crowds numerically stable
    A_WALL = 25.0
    MAX_SPEED = 2.5
    DT = 0.05
    FLOW_TAU = 1.0     # recorded velocities are smoothed over ~1 s, like optical flow

    def __init__(self, scenario: Scenario | None = None, seed: int = 0, venue: Venue | None = None):
        self.sc = scenario or Scenario()
        self.venue = venue or rally_ground(self.sc.gate4_open)
        self.rng = np.random.default_rng(seed)
        self.ff_front: FloorField = front_field(self.venue)
        self.ff_exit: FloorField = exit_field(self.venue)
        self.wall_a, self.wall_b = _seg_arrays(self.venue.all_wall_segments())
        self.pos = np.zeros((0, 2))
        self.vel = np.zeros((0, 2))
        self.vel_s = np.zeros((0, 2))
        self.r = np.zeros(0)
        self.v0 = np.zeros(0)
        self.spawned = 0
        self.t = 0.0

    # ---- spawning -----------------------------------------------------
    def _spawn(self):
        sc = self.sc
        if self.t >= sc.arrival_end or self.spawned >= sc.n_people:
            return
        entry_gates = [g for g in self.venue.gates if g.is_open and g.seg[1] == 0.0]
        target_total = min(sc.n_people, int(np.ceil(sc.n_people * (self.t + self.DT) / sc.arrival_end)))
        n_new = target_total - self.spawned
        if n_new <= 0:
            return
        gi = self.rng.integers(0, len(entry_gates), n_new)
        new = []
        for k in range(n_new):
            g = entry_gates[gi[k]]
            x1, _, x2, _ = g.seg
            x = self.rng.uniform(min(x1, x2) + 0.4, max(x1, x2) - 0.4)
            y = self.rng.uniform(1.0, 2.5)
            new.append((x, y))
        new = np.array(new)
        self.pos = np.vstack([self.pos, new])
        self.vel = np.vstack([self.vel, np.zeros_like(new)])
        self.vel_s = np.vstack([self.vel_s, np.zeros_like(new)])
        self.r = np.concatenate([self.r, self.rng.uniform(0.19, 0.23, n_new)])
        self.v0 = np.concatenate([self.v0, np.clip(self.rng.normal(1.3, 0.2, n_new), 0.8, 1.8)])
        self.spawned += n_new

    # ---- forces -------------------------------------------------------
    def _desired(self) -> np.ndarray:
        sc = self.sc
        phase = sc.phase(self.t)
        if phase == "dispersal":
            return self.ff_exit.direction(self.pos) * self.v0[:, None]
        e = self.ff_front.direction(self.pos)
        speed = self.v0.copy()
        wx0, wy0, wx1, wy1 = self.venue.watch_area
        in_watch = (
            (self.pos[:, 0] > wx0) & (self.pos[:, 0] < wx1) & (self.pos[:, 1] > wy0) & (self.pos[:, 1] < wy1)
        )
        # once in the watch area, people only push gently - except during a surge
        if phase == "surge":
            ramp = min(1.0, (self.t - sc.surge_start) / sc.surge_ramp)
            push = sc.settle_speed + (sc.surge_speed - sc.settle_speed) * ramp
        else:
            push = sc.settle_speed
        near_front = self.ff_front.distance(self.pos) < 25.0
        settled = in_watch & near_front
        speed[settled] = np.minimum(speed[settled], push)
        return e * speed[:, None]

    def _agent_forces(self) -> np.ndarray:
        n = len(self.pos)
        f = np.zeros((n, 2))
        if n < 2:
            return f
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(r=0.8, output_type="ndarray")
        if len(pairs) == 0:
            return f
        i, j = pairs[:, 0], pairs[:, 1]
        d_vec = self.pos[i] - self.pos[j]
        d = np.linalg.norm(d_vec, axis=1)
        d = np.maximum(d, 1e-3)
        nvec = d_vec / d[:, None]
        rij = self.r[i] + self.r[j]
        overlap = rij - d
        mag = self.A * np.exp(np.minimum(overlap, 0.0) / self.B) + self.K_BODY * np.maximum(overlap, 0)
        # damping on the normal relative velocity of touching people
        rel_n = ((self.vel[i] - self.vel[j]) * nvec).sum(1)
        mag -= np.where(overlap > 0, self.DAMP * rel_n, 0.0)
        fv = mag[:, None] * nvec
        np.add.at(f, i, fv)
        np.add.at(f, j, -fv)
        return f

    def _wall_forces(self) -> np.ndarray:
        p = self.pos[:, None, :]                      # (n, 1, 2)
        a = self.wall_a[None, :, :]                   # (1, w, 2)
        ab = (self.wall_b - self.wall_a)[None, :, :]
        L2 = np.maximum((ab ** 2).sum(-1), 1e-9)
        t = np.clip(((p - a) * ab).sum(-1) / L2, 0, 1)
        closest = a + t[..., None] * ab
        dv = p - closest
        d = np.maximum(np.linalg.norm(dv, axis=-1), 1e-3)
        nvec = dv / d[..., None]
        overlap = self.r[:, None] - d
        mag = self.A_WALL * np.exp(np.minimum(overlap, 0.0) / self.B) + 2 * self.K_BODY * np.maximum(overlap, 0)
        mag = np.where(d < 1.0, mag, 0.0)
        return (mag[..., None] * nvec).sum(axis=1)

    # ---- main loop ----------------------------------------------------
    def step(self):
        self._spawn()
        if len(self.pos):
            desired = self._desired()
            acc = (desired - self.vel) / self.TAU + self._agent_forces() + self._wall_forces()
            self.vel = self.vel + acc * self.DT
            speed = np.linalg.norm(self.vel, axis=1)
            too_fast = speed > self.MAX_SPEED
            self.vel[too_fast] *= (self.MAX_SPEED / speed[too_fast])[:, None]
            self.pos = self.pos + self.vel * self.DT
            self.vel_s = self.vel_s + (self.vel - self.vel_s) * (self.DT / self.FLOW_TAU)
            # people who walked out through a gate leave the simulation
            inside = (
                (self.pos[:, 0] > 0.05) & (self.pos[:, 0] < self.venue.width - 0.05)
                & (self.pos[:, 1] > 0.05) & (self.pos[:, 1] < self.venue.height - 0.05)
            )
            if not inside.all():
                self.pos, self.vel, self.vel_s = self.pos[inside], self.vel[inside], self.vel_s[inside]
                self.r, self.v0 = self.r[inside], self.v0[inside]
        self.t += self.DT

    def run(self, record_every: float = 2.0, progress: bool = False) -> Trajectory:
        tr = Trajectory(self.venue, self.sc)
        n_steps = int(round(self.sc.duration / self.DT))
        every = int(round(record_every / self.DT))
        for k in range(n_steps + 1):
            if k % every == 0:
                tr.times.append(round(self.t, 3))
                tr.pos.append(self.pos.astype(np.float32).copy())
                tr.vel.append(self.vel_s.astype(np.float32).copy())
                if progress and k % (every * 30) == 0:
                    print(f"  t={self.t:6.0f}s  people={len(self.pos):5d}  phase={self.sc.phase(self.t)}")
            if k < n_steps:
                self.step()
        return tr


def simulate(seed: int = 0, scenario: Scenario | None = None, progress: bool = False) -> Trajectory:
    return CrowdSim(scenario, seed).run(progress=progress)
