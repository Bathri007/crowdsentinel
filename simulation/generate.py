"""Generate and cache ground-truth scenarios.

Usage (from the project root):
    python -m simulation.generate --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

from .crowd_sim import CrowdSim, Scenario, Trajectory
from .venue import rally_ground

SCENARIO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "scenarios")


def scenario_path(seed: int) -> str:
    return os.path.join(SCENARIO_DIR, f"rally_seed{seed}.npz")


def _run(seed: int) -> str:
    path = scenario_path(seed)
    t0 = time.time()
    tr = CrowdSim(Scenario(), seed=seed).run()
    os.makedirs(SCENARIO_DIR, exist_ok=True)
    tr.save(path)
    return f"seed {seed}: {len(tr)} snapshots in {time.time() - t0:.0f}s -> {path}"


def load_or_generate(seed: int = 0) -> Trajectory:
    path = scenario_path(seed)
    if not os.path.exists(path):
        print(f"Simulating rally scenario (seed {seed}), takes a few minutes the first time...")
        print(_run(seed))
    return Trajectory.load(path, rally_ground(), Scenario())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    with ProcessPoolExecutor(max_workers=min(a.workers, len(a.seeds))) as ex:
        for msg in ex.map(_run, a.seeds):
            print(msg)


if __name__ == "__main__":
    main()
