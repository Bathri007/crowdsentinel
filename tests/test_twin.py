"""Fast tests for the twin. Run from the project root:  python -m pytest -q"""
import numpy as np
import pytest

from simulation.crowd_sim import CrowdSim, Scenario
from simulation.sensors.base import Measurement
from simulation.venue import rally_ground
from twin.evacuation import Evacuation
from twin.fusion import CCTVOnly, KalmanFusion, WeightedFusion
from twin.ground_truth import zone_truth
from twin.predict import RateTracker
from twin.risk import level_of, score


@pytest.fixture(scope="module")
def venue():
    return rally_ground()


@pytest.fixture(scope="module")
def short_run():
    sc = Scenario(n_people=300, arrival_end=40, surge_start=60, surge_end=70, dispersal=80, duration=90)
    return CrowdSim(sc, seed=1).run()


def test_venue_grid(venue):
    assert (venue.rows, venue.cols) == (14, 20)
    assert venue.zone_walkable.sum() < venue.n_zones          # stage zones are blocked
    assert venue.zone_name(0) == "A1"
    z = venue.zone_index(np.array([52.0]), np.array([57.0]))[0]
    assert venue.zone_name(z) == "L11"


def test_simulation_people_stay_inside_and_respect_fence(short_run):
    v = short_run.venue
    for pos in short_run.pos:
        if len(pos):
            assert pos[:, 0].min() >= 0 and pos[:, 0].max() <= v.width
            assert pos[:, 1].min() >= 0 and pos[:, 1].max() <= v.height
    # nobody should be inside the stage
    x0, y0, x1, y1 = v.stage
    last = short_run.pos[-1]
    inside_stage = (last[:, 0] > x0 + 0.5) & (last[:, 0] < x1 - 0.5) & (last[:, 1] > y0 + 0.5)
    assert not inside_stage.any()


def test_ground_truth_counts_add_up(short_run):
    k = len(short_run) // 2
    g = zone_truth(short_run.venue, short_run.pos[k], short_run.vel[k])
    assert g["count"].sum() == len(short_run.pos[k])
    assert np.allclose(g["density"], g["count"] / 25.0)


def _m(kind, zones, value, var):
    return Measurement("T", kind, np.array(zones), value, var, 0.0)


def test_kalman_tracks_a_direct_measurement(venue):
    kf = KalmanFusion(venue)
    z = int(np.flatnonzero(venue.zone_walkable.ravel())[50])
    for _ in range(10):
        kf.step([_m("cctv", [z], 50.0, 4.0)], 2.0)
    assert abs(kf.counts()[z] - 50.0) < 2.0
    assert kf.density_std()[z] < 0.2


def test_kalman_uses_regional_totals(venue):
    """A tower total over many zones must move the sum of those zones towards it."""
    kf = KalmanFusion(venue)
    zones = list(np.flatnonzero(venue.zone_walkable.ravel())[:40])
    for _ in range(20):
        kf.step([_m("tower", zones, 400.0, 100.0)], 2.0)
    assert abs(kf.counts()[zones].sum() - 400.0) < 40.0


def test_uncertainty_grows_where_nobody_looks(venue):
    kf = KalmanFusion(venue)
    walk = np.flatnonzero(venue.zone_walkable.ravel())
    seen, unseen = int(walk[10]), int(walk[200])
    for _ in range(30):
        kf.step([_m("cctv", [seen], 10.0, 1.0)], 2.0)
    std = kf.density_std()
    assert std[unseen] > 3 * std[seen]


def test_kalman_stays_stable_with_all_source_types(venue):
    kf = KalmanFusion(venue)
    walk = np.flatnonzero(venue.zone_walkable.ravel())
    rng = np.random.default_rng(0)
    for _ in range(100):
        ms = [_m("cctv", [z], rng.uniform(0, 30), 9.0) for z in walk[:30]]
        ms.append(_m("tower", walk[:90], 900.0, 2500.0))
        ms.append(_m("wifi", walk[100:110], 120.0, 400.0))
        kf.step(ms, 2.0)
    assert np.isfinite(kf.x).all()
    assert kf.counts().sum() < 5000
    assert np.all(np.linalg.eigvalsh(kf.P) > -1e-6)        # covariance stays positive semi-definite


def test_baselines_run(venue):
    z = int(np.flatnonzero(venue.zone_walkable.ravel())[5])
    for E in (CCTVOnly, WeightedFusion):
        e = E(venue)
        e.step([_m("cctv", [z], 20.0, 4.0)], 2.0)
        assert e.counts()[z] == pytest.approx(20.0)


def test_risk_levels_and_unverified():
    d = np.array([0.5, 3.0, 5.0, 7.0, 3.5])
    assert level_of(d).tolist() == [0, 1, 2, 3, 1]
    std = np.array([0, 0, 0, 0, 3.0])
    r = score(d, std)
    assert r["unverified"].tolist() == [False, False, False, False, True]


def test_rate_tracker_warns_before_critical():
    rt = RateTracker(1)
    d, warned_at = 4.0, None
    for k in range(60):
        d += 0.02 * 2.0                       # rising 0.02 persons/m2 per second
        rt.update(np.array([d]), 2.0)
        if warned_at is None and np.isfinite(rt.time_to_critical()[0]):
            warned_at = d
    assert warned_at is not None and warned_at < 6.0


def test_evacuation_routes_reach_an_exit(venue):
    ev = Evacuation(venue)
    counts = np.zeros(venue.n_zones)
    counts[venue.zone_walkable.ravel()] = 10
    plan = ev.plan(counts, counts / 25, {g.name: g.is_open for g in venue.gates})
    assert set(plan["load"]) == {"G1", "G2", "G3"}             # G4 is closed by default
    assert sum(plan["load"].values()) == pytest.approx(counts.sum())
    # every zone above the fence must route through the gap (no path crosses the fence)
    z = venue.zone_index(np.array([10.0]), np.array([40.0]))[0]
    path, cur = [], z
    while plan["next"].get(cur) is not None and len(path) < 100:
        cur = plan["next"][cur]
        path.append(cur)
    xs = venue.zone_centers()[path][:, 0]
    ys = venue.zone_centers()[path][:, 1]
    crossing = np.flatnonzero(np.diff((ys > 25).astype(int)) != 0)
    for i in crossing:
        assert 40 <= xs[i] <= 60 or 40 <= xs[i + 1] <= 60


def test_opening_closed_gate_is_suggested_when_crowded(venue):
    ev = Evacuation(venue)
    counts = np.zeros(venue.n_zones)
    counts[venue.zone_walkable.ravel()] = 60
    out = ev.recommend(counts, counts / 25, critical_zones=[150], predicted_zones=[])
    assert any(a["type"] == "open_gate" and a["gate"] == "G4" for a in out["actions"])
