"""CrowdSentinel control-room backend.

Runs the digital twin live (one tick every 2 s of event time, speed adjustable),
pushes every tick to the dashboard over a WebSocket, and stores history for replay.

Start it from the project root:
    python -m uvicorn backend.main:app --port 8000
then open http://localhost:8000
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from simulation.generate import load_or_generate
from twin.engine import TwinEngine

from .storage import History

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASH = os.path.join(ROOT, "dashboard")
SEED = int(os.environ.get("CS_SEED", "0"))


class Live:
    def __init__(self):
        self.tr = load_or_generate(SEED)
        self.policy = "active"
        self.speed = 1.0
        self.paused = False
        self.clients: set[WebSocket] = set()
        self.history = History(os.path.join(ROOT, "data", "history.sqlite"))
        self.latest: dict | None = None
        self.gate_override: dict[str, bool] = {}
        self._new_engine()

    def _new_engine(self):
        for g in self.tr.venue.gates:
            if g.name in self.gate_override:
                g.is_open = self.gate_override[g.name]
        self.engine = TwinEngine(self.tr, fusion="kalman", drone_policy=self.policy, seed=SEED)

    def restart(self):
        self.history.new_run()
        self._new_engine()

    async def broadcast(self, msg: dict):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def loop(self):
        while True:
            if self.paused:
                await asyncio.sleep(0.2)
                continue
            if self.engine.done():
                await asyncio.sleep(3.0)          # pause on the last frame, then replay the event
                self.restart()
                continue
            k = self.engine.k
            self.engine.step()
            snap = self.engine.snapshot()
            snap["k"] = k
            snap["k_max"] = len(self.tr) - 1
            snap["speed"] = self.speed
            snap["policy"] = self.policy
            self.latest = snap
            self.history.save(k, snap["t"], snap)
            await self.broadcast({"type": "tick", "data": snap})
            await asyncio.sleep(2.0 / self.speed)


live: Live | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global live
    live = Live()
    task = asyncio.create_task(live.loop())
    yield
    task.cancel()


app = FastAPI(title="CrowdSentinel", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=DASH), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(DASH, "index.html"))


@app.get("/api/venue")
def venue():
    v = live.tr.venue
    s = live.engine.sensors
    return {
        **v.to_dict(),
        "zone_names": [v.zone_name(i) for i in range(v.n_zones)],
        "cameras": [{"name": c.name, "zones": c.zones.tolist()} for c in s.cams],
        "towers": [{"name": t.name, "zones": t.zones.tolist()} for t in s.towers],
        "wifi": [{"name": w.name, "zones": w.zones.tolist()} for w in s.wifi],
        "duration": live.tr.times[-1],
        "phases": {
            "arrival": [0, live.tr.scenario.arrival_end],
            "surge": [live.tr.scenario.surge_start, live.tr.scenario.surge_end],
            "dispersal": [live.tr.scenario.dispersal, live.tr.scenario.duration],
        },
    }


@app.get("/api/state")
def state():
    return live.latest or {}


@app.get("/api/history/{k}")
def history(k: int):
    snap = live.history.get(k)
    if snap is None:
        return JSONResponse({"error": f"tick {k} not recorded yet"}, status_code=404)
    return snap


@app.get("/api/alert_marks")
def alert_marks():
    return live.history.alert_marks()


class Control(BaseModel):
    speed: float | None = None
    paused: bool | None = None
    policy: str | None = None
    restart: bool | None = None
    open_gate: str | None = None


@app.post("/api/control")
def control(c: Control):
    if c.speed is not None:
        live.speed = max(0.25, min(c.speed, 20.0))
    if c.paused is not None:
        live.paused = c.paused
    if c.policy is not None and c.policy in ("active", "patrol", "random", "hover"):
        live.policy = c.policy
        live.restart()
    if c.open_gate:
        # NOTE: the crowd simulation is pre-recorded, so opening a gate changes the
        # twin's evacuation plan, not the simulated people (who already moved).
        live.gate_override[c.open_gate] = True
        for g in live.tr.venue.gates:
            if g.name == c.open_gate:
                g.is_open = True
        from twin.evacuation import Evacuation

        live.engine.evac = Evacuation(live.tr.venue)
    if c.restart:
        live.gate_override.clear()
        for g in live.tr.venue.gates:
            g.is_open = g.name != "G4"
        live.restart()
    return {"speed": live.speed, "paused": live.paused, "policy": live.policy}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    live.clients.add(websocket)
    if live.latest:
        await websocket.send_json({"type": "tick", "data": live.latest})
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        live.clients.discard(websocket)
