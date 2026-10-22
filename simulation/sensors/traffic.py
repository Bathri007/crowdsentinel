"""Road traffic around the venue exits (are the escape roads clear?).

Real source: TomTom Traffic Flow API (free tier, needs a key from developer.tomtom.com).
Set the key and the road points outside each gate, then call `read()`:

    set TOMTOM_KEY=your_key_here            (Windows)
    roads = TomTomTraffic({"G1": (13.0827, 80.2707), "G2": (13.0830, 80.2712)})
    roads.read()   # -> {"G1": {"speed_kmh": 18, "free_flow_kmh": 40, "congestion": 0.55}, ...}

The synthetic venue has no real location, so the demo uses SimulatedTraffic,
which slows the road outside a gate as more people leave through it.
Not yet drawn on the dashboard - planned next step.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"


class TomTomTraffic:
    def __init__(self, gate_points: dict[str, tuple[float, float]], key: str | None = None):
        self.points = gate_points
        self.key = key or os.environ.get("TOMTOM_KEY")
        if not self.key:
            raise ValueError("set the TOMTOM_KEY environment variable or pass key=")

    def read(self) -> dict:
        out = {}
        for gate, (lat, lon) in self.points.items():
            q = urllib.parse.urlencode({"point": f"{lat},{lon}", "key": self.key})
            with urllib.request.urlopen(f"{URL}?{q}", timeout=10) as r:
                d = json.load(r)["flowSegmentData"]
            cur, free = d["currentSpeed"], max(d["freeFlowSpeed"], 1)
            out[gate] = {"speed_kmh": cur, "free_flow_kmh": free, "congestion": round(1 - cur / free, 2)}
        return out


class SimulatedTraffic:
    """Congestion outside a gate rises with the number of people leaving through it."""

    def __init__(self, gates, free_flow_kmh: float = 40.0):
        self.gates, self.free = gates, free_flow_kmh

    def read(self, outflow_per_min: dict[str, float]) -> dict:
        out = {}
        for g in self.gates:
            f = outflow_per_min.get(g.name, 0.0)
            congestion = min(0.9, f / 400.0)      # 400 people/min saturates a two-lane road
            out[g.name] = {"speed_kmh": round(self.free * (1 - congestion)), "free_flow_kmh": self.free,
                           "congestion": round(congestion, 2)}
        return out
