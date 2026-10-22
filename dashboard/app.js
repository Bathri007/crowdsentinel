// CrowdSentinel control-room dashboard: draws the live twin on a canvas.
(() => {
  const $ = (id) => document.getElementById(id);
  const canvas = $("map");
  const ctx = canvas.getContext("2d");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let venue = null;
  let live = null;        // latest live snapshot
  let shown = null;       // snapshot on screen (live or replay)
  let replay = false;
  let paused = false;
  let hoverZone = null;
  let pinnedZone = null;
  let geom = null;
  let hatch = null;

  const layers = () => ({
    base: document.querySelector('input[name="base"]:checked').value,
    unc: $("lyr-unc").checked,
    flow: $("lyr-flow").checked,
    evac: $("lyr-evac").checked,
    cams: $("lyr-cams").checked,
  });

  // ---------- colour scale: persons per m2 ----------
  const STOPS = [
    [0, [34, 50, 63]], [1, [38, 82, 90]], [2, [47, 127, 134]], [3, [120, 150, 110]],
    [4, [201, 180, 88]], [5, [224, 138, 60]], [6, [229, 72, 77]], [8, [170, 30, 45]],
  ];
  function densColor(d) {
    if (d <= 0) return `rgb(${STOPS[0][1]})`;
    for (let i = 1; i < STOPS.length; i++) {
      if (d <= STOPS[i][0]) {
        const [a, ca] = STOPS[i - 1], [b, cb] = STOPS[i];
        const t = (d - a) / (b - a);
        return `rgb(${ca.map((v, j) => Math.round(v + (cb[j] - v) * t)).join(",")})`;
      }
    }
    return `rgb(${STOPS[STOPS.length - 1][1]})`;
  }

  function fmtTime(t) {
    const m = Math.floor(t / 60), s = Math.floor(t % 60);
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }

  // ---------- geometry ----------
  function resize() {
    const wrap = $("map-wrap");
    const dpr = window.devicePixelRatio || 1;
    const W = wrap.clientWidth, H = wrap.clientHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (venue) {
      const pad = 28;
      const s = Math.min((W - 2 * pad) / venue.width, (H - 2 * pad) / venue.height);
      const ox = (W - venue.width * s) / 2, oy = (H - venue.height * s) / 2;
      geom = { s, ox, oy, W, H };
    }
    draw();
  }
  const X = (x) => geom.ox + x * geom.s;
  const Y = (y) => geom.oy + (venue.height - y) * geom.s;   // y grows towards the stage (up)

  function makeHatch() {
    const c = document.createElement("canvas");
    c.width = c.height = 8;
    const g = c.getContext("2d");
    g.strokeStyle = "rgba(230,237,242,0.85)";
    g.lineWidth = 1.2;
    g.beginPath(); g.moveTo(0, 8); g.lineTo(8, 0); g.moveTo(-2, 2); g.lineTo(2, -2); g.moveTo(6, 10); g.lineTo(10, 6); g.stroke();
    return ctx.createPattern(c, "repeat");
  }

  function zoneRect(i) {
    const r = Math.floor(i / venue.cols), c = i % venue.cols, cs = venue.cell;
    return [X(c * cs), Y((r + 1) * cs), cs * geom.s, cs * geom.s];
  }
  function zoneCentre(i) {
    const r = Math.floor(i / venue.cols), c = i % venue.cols, cs = venue.cell;
    return [X((c + 0.5) * cs), Y((r + 0.5) * cs)];
  }

  // ---------- drawing ----------
  function draw(now = performance.now()) {
    if (!venue || !geom) return;
    const L = layers();
    ctx.clearRect(0, 0, geom.W, geom.H);
    ctx.fillStyle = "#101a22";
    ctx.fillRect(X(0), Y(venue.height), venue.width * geom.s, venue.height * geom.s);

    const snap = shown;
    const n = venue.rows * venue.cols;
    if (snap) {
      const dens = L.base === "truth" && snap.truth_density ? snap.truth_density : snap.density;
      for (let i = 0; i < n; i++) {
        if (!venue.walkable[i]) continue;
        const [x, y, w, h] = zoneRect(i);
        ctx.fillStyle = densColor(dens[i]);
        ctx.fillRect(x + 0.5, y + 0.5, w - 1, h - 1);
        if (L.unc && L.base === "density" && snap.std[i] > 0.3) {
          ctx.globalAlpha = Math.min(1, (snap.std[i] - 0.3) / 0.5) * 0.85 + 0.15;
          ctx.fillStyle = hatch;
          ctx.fillRect(x + 0.5, y + 0.5, w - 1, h - 1);
          ctx.globalAlpha = 1;
        }
      }
    }

    // grid labels (row letters, column numbers)
    ctx.fillStyle = "#8fa3b3";
    ctx.font = "600 11px 'Barlow Condensed', sans-serif";
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let r = 0; r < venue.rows; r++) ctx.fillText(String.fromCharCode(65 + r), X(0) - 6, Y((r + 0.5) * venue.cell));
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let c = 0; c < venue.cols; c += 1) ctx.fillText(c + 1, X((c + 0.5) * venue.cell), Y(0) + 6);

    // camera coverage
    if (L.cams) {
      ctx.strokeStyle = "rgba(124,196,232,0.55)"; ctx.lineWidth = 1;
      for (const cam of venue.cameras) {
        const xs = cam.zones.map((z) => z % venue.cols), ys = cam.zones.map((z) => Math.floor(z / venue.cols));
        const c0 = Math.min(...xs), c1 = Math.max(...xs) + 1, r0 = Math.min(...ys), r1 = Math.max(...ys) + 1;
        ctx.setLineDash([2, 3]);
        ctx.strokeRect(X(c0 * venue.cell) + 2, Y(r1 * venue.cell) + 2, (c1 - c0) * venue.cell * geom.s - 4, (r1 - r0) * venue.cell * geom.s - 4);
        ctx.setLineDash([]);
        ctx.fillStyle = "#7cc4e8"; ctx.font = "600 12px Barlow, sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "top";
        ctx.fillText(cam.name, X(c0 * venue.cell) + 6, Y(r1 * venue.cell) + 6);
      }
    }

    // walls, stage, gates
    ctx.strokeStyle = "#c7d3dc"; ctx.lineWidth = 3; ctx.lineCap = "round";
    for (const [x1, y1, x2, y2] of venue.walls) { ctx.beginPath(); ctx.moveTo(X(x1), Y(y1)); ctx.lineTo(X(x2), Y(y2)); ctx.stroke(); }
    const [sx0, sy0, sx1, sy1] = venue.stage;
    ctx.fillStyle = "#2c3e4d"; ctx.fillRect(X(sx0), Y(sy1), (sx1 - sx0) * geom.s, (sy1 - sy0) * geom.s);
    ctx.fillStyle = "#c7d3dc"; ctx.font = "600 13px Barlow, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("Stage", X((sx0 + sx1) / 2), Y((sy0 + sy1) / 2));
    const gatesOpen = snap && snap.evacuation ? snap.evacuation.gates_open : {};
    for (const g of venue.gates) {
      const open = g.name in gatesOpen ? gatesOpen[g.name] : g.open;
      const [x1, y1, x2, y2] = g.seg;
      ctx.strokeStyle = open ? "#5fd0a0" : "#e5484d"; ctx.lineWidth = 6; ctx.lineCap = "butt";
      ctx.beginPath(); ctx.moveTo(X(x1), Y(y1)); ctx.lineTo(X(x2), Y(y2)); ctx.stroke();
      ctx.fillStyle = open ? "#5fd0a0" : "#e5484d"; ctx.font = "700 13px 'Barlow Condensed', sans-serif";
      const gx = (x1 + x2) / 2, gy = (y1 + y2) / 2;
      if (gx >= venue.width) {           // gate on the east wall: label inside, left of the wall
        ctx.textAlign = "right"; ctx.fillText(`${g.name}${open ? "" : " closed"}`, X(gx) - 8, Y(gy));
      } else {
        ctx.textAlign = "center"; ctx.fillText(`${g.name}${open ? "" : " closed"}`, X(gx), gy <= 0 ? Y(gy) + 24 : Y(gy));
      }
    }

    if (snap) {
      // movement arrows
      if (L.flow) {
        ctx.strokeStyle = "rgba(240,246,250,0.8)"; ctx.lineWidth = 1.3;
        for (let i = 0; i < n; i++) {
          if (!venue.walkable[i] || snap.density[i] < 0.3) continue;
          const [vx, vy] = snap.flow[i];
          const sp = Math.hypot(vx, vy);
          if (sp < 0.08) continue;
          const [cx, cy] = zoneCentre(i);
          const len = Math.min(1, sp / 1.3) * geom.s * venue.cell * 0.42;
          const ux = vx / sp, uy = -vy / sp;
          arrow(cx - ux * len / 2, cy - uy * len / 2, cx + ux * len / 2, cy + uy * len / 2, 4);
        }
      }
      // evacuation routes
      if (L.evac && snap.evacuation) {
        ctx.strokeStyle = "#5fd0a0"; ctx.lineWidth = 1.6;
        for (const [z, nx] of Object.entries(snap.evacuation.next)) {
          if (nx === null || snap.density[z] < 0.2) continue;
          const [ax, ay] = zoneCentre(+z), [bx, by] = zoneCentre(nx);
          arrow(ax + (bx - ax) * 0.15, ay + (by - ay) * 0.15, ax + (bx - ax) * 0.75, ay + (by - ay) * 0.75, 5);
        }
      }
      // critical zones: the one animated element on the page
      const pulse = reduceMotion ? 1 : 0.55 + 0.45 * Math.abs(Math.sin(now / 380));
      ctx.lineWidth = 2.5;
      for (let i = 0; i < n; i++) {
        if (snap.level[i] >= 3) {
          const [x, y, w, h] = zoneRect(i);
          ctx.strokeStyle = `rgba(255,90,95,${pulse})`;
          ctx.strokeRect(x + 2, y + 2, w - 4, h - 4);
        } else if (snap.ttc[i] !== null && snap.ttc[i] > 0) {
          const [x, y, w, h] = zoneRect(i);
          ctx.setLineDash([4, 3]); ctx.strokeStyle = "#e08a3c"; ctx.strokeRect(x + 2, y + 2, w - 4, h - 4); ctx.setLineDash([]);
        }
      }
      // drone footprint and heading
      const d = snap.drone;
      if (d) {
        const half = 12.5;
        ctx.strokeStyle = "#7cc4e8"; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
        ctx.strokeRect(X(d.pos[0] - half), Y(d.pos[1] + half), 2 * half * geom.s, 2 * half * geom.s);
        ctx.setLineDash([]);
        if (Math.hypot(d.target[0] - d.pos[0], d.target[1] - d.pos[1]) > 1) {
          ctx.strokeStyle = "rgba(124,196,232,0.7)"; ctx.lineWidth = 1.5;
          arrow(X(d.pos[0]), Y(d.pos[1]), X(d.target[0]), Y(d.target[1]), 7);
        }
        ctx.fillStyle = "#7cc4e8";
        ctx.beginPath(); ctx.arc(X(d.pos[0]), Y(d.pos[1]), 5, 0, Math.PI * 2); ctx.fill();
      }
      // selected zone outline
      const sel = pinnedZone ?? hoverZone;
      if (sel !== null && venue.walkable[sel]) {
        const [x, y, w, h] = zoneRect(sel);
        ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2; ctx.strokeRect(x + 1, y + 1, w - 2, h - 2);
      }
    }
  }

  function arrow(x1, y1, x2, y2, head) {
    const a = Math.atan2(y2 - y1, x2 - x1);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    ctx.lineTo(x2 - head * Math.cos(a - 0.5), y2 - head * Math.sin(a - 0.5));
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - head * Math.cos(a + 0.5), y2 - head * Math.sin(a + 0.5));
    ctx.stroke();
  }

  // ---------- side panels ----------
  function render(snap) {
    shown = snap;
    $("clock").textContent = fmtTime(snap.t);
    $("phase").textContent = snap.phase;
    $("people").textContent = snap.estimated_total.toLocaleString("en-IN");
    if (snap.true_total !== undefined) $("truth-people").textContent = snap.true_total.toLocaleString("en-IN");
    $("tick").textContent = `${snap.tick_ms} ms`;

    const al = snap.alerts;
    $("alert-count").textContent = al.length;
    $("alert-count").classList.toggle("hot", al.some((a) => a.level === "critical"));
    const kindText = { critical: "Critical now", warning: "Predicted", unverified: "Needs eyes on it" };
    $("alerts").innerHTML = al.length
      ? al.map((a) => `<li class="${a.level}" data-zone="${a.zone_id}"><span class="kind">${kindText[a.level]}</span>${a.text}</li>`).join("")
      : `<li class="empty">No alerts. All zones are below 4 persons per m².</li>`;

    const ev = snap.evacuation;
    if (ev) {
      $("actions").innerHTML = ev.actions.length
        ? ev.actions.slice(0, 6).map((a) => `<li><span>${a.text}</span>${a.type === "open_gate" && !replay
            ? `<button type="button" data-gate="${a.gate}">Open ${a.gate}</button>` : ""}</li>`).join("")
        : `<li class="empty">Nothing to do right now.</li>`;
      $("gates").innerHTML = `<tr><th>Exit</th><th>People heading there</th><th>Time to clear</th></tr>` +
        venue.gates.map((g) => {
          const open = ev.gates_open[g.name];
          return open
            ? `<tr><td>${g.name}</td><td>${(ev.load[g.name] ?? 0).toLocaleString("en-IN")}</td><td>${ev.clearance_min[g.name]} min</td></tr>`
            : `<tr class="closed"><td>${g.name}</td><td colspan="2">closed</td></tr>`;
        }).join("");
    }
    $("drone-reason").textContent = snap.drone.reason ? `Heading: ${snap.drone.reason}` : "–";
    if (snap.policy && document.activeElement !== $("policy")) $("policy").value = snap.policy;

    $("sources").innerHTML = snap.sources.map((s) => `<li><span class="dot ${s.online ? "" : "off"}"></span>
      <span>${s.name}</span><span class="tag">${s.online ? "" : "offline · "}${s.real ? "real feed" : "simulated"}</span></li>`).join("");

    if (!replay) {
      $("scrub").max = snap.k_max;
      $("scrub").value = snap.k;
    }
    updateZoneCard();
    draw();
  }

  function updateZoneCard() {
    const card = $("zone-card");
    const z = pinnedZone ?? hoverZone;
    if (z === null || !shown || !venue.walkable[z]) { card.hidden = true; return; }
    const s = shown;
    const conf = Math.round(s.confidence[z] * 100);
    const ttc = s.ttc[z];
    card.innerHTML = `<h3>Zone ${venue.zone_names[z]}</h3><dl>
      <dt>Density</dt><dd>${s.density[z].toFixed(1)} ± ${s.std[z].toFixed(1)} /m²</dd>
      ${s.truth_density ? `<dt>Simulated truth</dt><dd>${s.truth_density[z].toFixed(1)} /m²</dd>` : ""}
      <dt>Confidence</dt><dd>${conf}%</dd>
      <dt>Level</dt><dd>${s.levels[s.level[z]]}</dd>
      <dt>Critical in</dt><dd>${ttc === null ? "not expected" : ttc === 0 ? "now" : `~${Math.round(ttc)} s`}</dd>
      <dt>Movement</dt><dd>${Math.hypot(...s.flow[z]).toFixed(2)} m/s</dd></dl>`;
    const [cx, cy] = zoneCentre(z);
    const W = $("map-wrap").clientWidth;
    card.style.left = `${Math.min(cx + 16, W - 240)}px`;
    card.style.top = `${Math.max(cy - 40, 4)}px`;
    card.hidden = false;
  }

  function zoneAt(ev) {
    if (!geom) return null;
    const r = canvas.getBoundingClientRect();
    const x = (ev.clientX - r.left - geom.ox) / geom.s;
    const y = venue.height - (ev.clientY - r.top - geom.oy) / geom.s;
    if (x < 0 || y < 0 || x >= venue.width || y >= venue.height) return null;
    return Math.floor(y / venue.cell) * venue.cols + Math.floor(x / venue.cell);
  }

  // ---------- server ----------
  async function post(body) {
    const r = await fetch("/api/control", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    return r.json();
  }

  function setMode() {
    const el = $("live-state");
    el.dataset.mode = replay ? "replay" : paused ? "paused" : "live";
    el.textContent = replay ? "Replay" : paused ? "Paused" : "Live";
    $("btn-live").hidden = !replay;
  }

  async function loadMarks() {
    try {
      const marks = await (await fetch("/api/alert_marks")).json();
      const max = +$("scrub").max || 1;
      $("marks").innerHTML = marks.map((m) => `<i class="${m.level}" style="left:${(m.k / max) * 100}%"></i>`).join("");
    } catch (e) { /* timeline marks are optional */ }
  }

  function connect() {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    ws.onmessage = (m) => {
      const msg = JSON.parse(m.data);
      if (msg.type !== "tick") return;
      live = msg.data;
      if (!replay) render(live);
      if (live.k % 10 === 0) loadMarks();
    };
    ws.onclose = () => {
      $("live-state").textContent = "Reconnecting…";
      setTimeout(connect, 1500);
    };
  }

  async function init() {
    venue = await (await fetch("/api/venue")).json();
    $("venue-name").textContent = venue.name;
    hatch = makeHatch();
    const ph = venue.phases, dur = venue.duration;
    const label = { arrival: "Arrival", surge: "Surge", dispersal: "Dispersal" };
    $("phases").innerHTML = Object.entries(ph).map(([k, [a]]) => `<span style="left:${(a / dur) * 100}%">${label[k]}</span>`).join("");
    resize();
    connect();
    post({ speed: 4 });
    const loop = (t) => { if (shown && shown.level.some((l) => l >= 3)) draw(t); requestAnimationFrame(loop); };
    if (!reduceMotion) requestAnimationFrame(loop);
  }

  // ---------- events ----------
  window.addEventListener("resize", resize);
  document.querySelectorAll(".layers input").forEach((el) => el.addEventListener("change", () => {
    document.body.classList.toggle("show-truth", layers().base === "truth");
    draw();
  }));
  canvas.addEventListener("mousemove", (e) => { hoverZone = zoneAt(e); updateZoneCard(); draw(); });
  canvas.addEventListener("mouseleave", () => { hoverZone = null; updateZoneCard(); draw(); });
  canvas.addEventListener("click", (e) => { const z = zoneAt(e); pinnedZone = pinnedZone === z ? null : z; updateZoneCard(); draw(); });
  $("alerts").addEventListener("click", (e) => {
    const li = e.target.closest("li[data-zone]");
    if (li) { pinnedZone = +li.dataset.zone; updateZoneCard(); draw(); }
  });
  $("actions").addEventListener("click", async (e) => {
    const b = e.target.closest("button[data-gate]");
    if (b) { b.disabled = true; b.textContent = "Opening…"; await post({ open_gate: b.dataset.gate }); }
  });
  $("btn-pause").addEventListener("click", async () => {
    paused = !paused;
    await post({ paused });
    $("btn-pause").textContent = paused ? "Resume" : "Pause";
    setMode();
  });
  document.querySelectorAll(".speed button").forEach((b) => b.addEventListener("click", async () => {
    document.querySelectorAll(".speed button").forEach((x) => x.classList.toggle("on", x === b));
    await post({ speed: +b.dataset.speed });
  }));
  $("btn-restart").addEventListener("click", async () => { replay = false; setMode(); await post({ restart: true }); });
  $("policy").addEventListener("change", async (e) => { replay = false; setMode(); await post({ policy: e.target.value }); });
  $("btn-live").addEventListener("click", () => { replay = false; setMode(); if (live) render(live); });
  $("scrub").addEventListener("input", async (e) => {
    const k = +e.target.value;
    if (live && k >= live.k) { replay = false; setMode(); render(live); return; }
    replay = true; setMode();
    const r = await fetch(`/api/history/${k}`);
    if (r.ok) render(await r.json());
  });

  init();
})();
