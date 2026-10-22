"""Evacuation routing on the zone graph.

Zones are graph nodes, neighbours are connected unless a wall/fence is in the
way. Walking cost grows with density, so routes avoid packed zones. Each open
gate is an exit node with a flow capacity of 1.3 persons per metre per second
(a common design value for egress through openings).
"""
from __future__ import annotations

import networkx as nx
import numpy as np

FLOW_PER_M = 1.3   # persons / m / s


def _intersects(p1, p2, q1, q2) -> bool:
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    o1, o2 = orient(p1, p2, q1), orient(p1, p2, q2)
    o3, o4 = orient(q1, q2, p1), orient(q1, q2, p2)
    return o1 != o2 and o3 != o4


class Evacuation:
    def __init__(self, venue):
        self.venue = venue
        self.centers = venue.zone_centers()
        self.walk = venue.zone_walkable.ravel()
        self.base = nx.Graph()
        segs = venue.all_wall_segments()
        open_or_closed_gate_segs = segs + [g.seg for g in venue.gates if g.is_open]
        R, C = venue.rows, venue.cols
        for r in range(R):
            for c in range(C):
                a = r * C + c
                if not self.walk[a]:
                    continue
                self.base.add_node(a)
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < R and 0 <= cc < C):
                        continue
                    b = rr * C + cc
                    if not self.walk[b]:
                        continue
                    pa, pb = self.centers[a], self.centers[b]
                    if any(_intersects(pa, pb, (s[0], s[1]), (s[2], s[3])) for s in open_or_closed_gate_segs):
                        continue
                    self.base.add_edge(a, b, length=float(np.hypot(*(pa - pb))))
        self.gate_zone = {}
        for g in venue.gates:
            gx, gy = g.center
            gx = min(max(gx, 0.1), venue.width - 0.1)
            gy = min(max(gy, 0.1), venue.height - 0.1)
            self.gate_zone[g.name] = int(venue.zone_index(np.array([gx]), np.array([gy]))[0])

    def plan(self, counts: np.ndarray, density: np.ndarray, gates_open: dict[str, bool]) -> dict:
        g = self.base.copy()
        for a, b, d in g.edges(data=True):
            dens = max(density[a], density[b])
            d["cost"] = d["length"] * (1.0 + (dens / 2.0) ** 2)
        exits = [gt for gt in self.venue.gates if gates_open.get(gt.name, gt.is_open)]
        for gt in exits:
            g.add_edge(("EXIT", gt.name), self.gate_zone[gt.name], cost=0.0, length=0.0)
        dist, paths = {}, {}
        for gt in exits:
            dd, pp = nx.single_source_dijkstra(g, ("EXIT", gt.name), weight="cost")
            dist[gt.name], paths[gt.name] = dd, pp
        assign, next_zone = {}, {}
        for z in self.base.nodes:
            best = min(exits, key=lambda gt: dist[gt.name].get(z, np.inf))
            if not np.isfinite(dist[best.name].get(z, np.inf)):
                continue
            assign[z] = best.name
            path = paths[best.name][z]           # exit -> ... -> z
            next_zone[z] = path[-2] if len(path) >= 3 else None
        load = {gt.name: 0.0 for gt in exits}
        for z, gname in assign.items():
            load[gname] += counts[z]
        clearance = {gt.name: load[gt.name] / (FLOW_PER_M * gt.width) for gt in exits}
        return {"assign": assign, "next": next_zone, "load": load, "clearance_s": clearance}

    def recommend(self, counts, density, critical_zones, predicted_zones) -> dict:
        now = {g.name: g.is_open for g in self.venue.gates}
        cur = self.plan(counts, density, now)
        actions = []
        worst = max(cur["clearance_s"].values()) if cur["clearance_s"] else 0.0
        for g in self.venue.gates:
            if not g.is_open:
                alt = self.plan(counts, density, {**now, g.name: True})
                alt_worst = max(alt["clearance_s"].values())
                if alt_worst < worst * 0.85 and (critical_zones or predicted_zones or worst > 240):
                    actions.append({
                        "type": "open_gate",
                        "gate": g.name,
                        "text": f"Open {g.name} to cut the longest exit time from {worst/60:.1f} to {alt_worst/60:.1f} min",
                    })
        for z in sorted(set(critical_zones) | set(predicted_zones)):
            gname = cur["assign"].get(z)
            if gname is None:
                continue
            tag = "is critical" if z in critical_zones else "will turn critical"
            actions.append({
                "type": "hold_and_redirect",
                "zone": int(z),
                "text": f"{self.venue.zone_name(z)} {tag}: stop people moving in, send them towards {gname}",
            })
        cur["actions"] = actions
        return cur
