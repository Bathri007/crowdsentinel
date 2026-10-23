"""Record the 2-minute demo video of the control-room dashboard.

Start the dashboard first (run_dashboard.bat, or
`python -m uvicorn backend.main:app --port 8050`), then:

    pip install playwright && python -m playwright install chromium
    python -m tools.record_demo --url http://localhost:8050 --out docs/demo.mp4

Needs ffmpeg on PATH to convert the recording to MP4.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import os
import shutil
import subprocess
import tempfile

W, H = 1600, 900

CAPTION_JS = """
(text) => {
  let el = document.getElementById('demo-caption');
  if (!el) {
    el = document.createElement('div');
    el.id = 'demo-caption';
    el.style.cssText = 'position:fixed;left:50%;bottom:96px;transform:translateX(-50%);max-width:880px;' +
      'padding:14px 22px;background:rgba(10,18,25,.92);border:1px solid #4b6577;border-radius:8px;' +
      'color:#e6edf2;font:500 22px/1.35 Barlow,system-ui,sans-serif;text-align:center;z-index:9999;' +
      'transition:opacity .3s';
    document.body.appendChild(el);
  }
  el.style.opacity = text ? 1 : 0;
  if (text) el.textContent = text;
}
"""

CARD_JS = """
([title, lines]) => {
  let el = document.getElementById('demo-card');
  if (!title) { if (el) el.remove(); return; }
  if (!el) {
    el = document.createElement('div');
    el.id = 'demo-card';
    el.style.cssText = 'position:fixed;inset:0;background:#14202b;z-index:10000;display:flex;flex-direction:column;' +
      'justify-content:center;padding:0 160px;color:#e6edf2;font-family:Barlow,system-ui,sans-serif';
    document.body.appendChild(el);
  }
  el.innerHTML = `<div style="font-size:64px;font-weight:700;margin-bottom:18px">${title}</div>` +
    lines.map(l => `<div style="font-size:28px;color:#b8c7d3;margin:8px 0">${l}</div>`).join('');
}
"""

ZONE_XY_JS = """
async (level) => {
  const v = await (await fetch('/api/venue')).json();
  const s = await (await fetch('/api/state')).json();
  const c = document.getElementById('map').getBoundingClientRect();
  const sc = Math.min((c.width - 56) / v.width, (c.height - 56) / v.height);
  const ox = (c.width - v.width * sc) / 2, oy = (c.height - v.height * sc) / 2;
  let best = -1, bd = -1;
  s.level.forEach((l, i) => { if (l >= level && s.density[i] > bd) { bd = s.density[i]; best = i; } });
  if (best < 0) return null;
  const r = Math.floor(best / v.cols), col = best % v.cols;
  return [c.left + ox + (col + 0.5) * v.cell * sc, c.top + oy + (v.height - (r + 0.5) * v.cell) * sc];
}
"""


async def control(page, **body):
    await page.evaluate(
        "(b) => fetch('/api/control', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(b)})",
        body)


async def event_time(page):
    s = await page.evaluate("async () => (await (await fetch('/api/state')).json()).t || 0")
    return float(s)


async def wait_event(page, t_target, timeout=180):
    for _ in range(int(timeout * 4)):
        if await event_time(page) >= t_target:
            return
        await page.wait_for_timeout(250)


async def record(url, out):
    from playwright.async_api import async_playwright

    tmp = tempfile.mkdtemp()
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(viewport={"width": W, "height": H}, record_video_dir=tmp,
                                        record_video_size={"width": W, "height": H})
        page = await ctx.new_page()
        cap = lambda t: page.evaluate(CAPTION_JS, t)  # noqa: E731
        card = lambda t, lines=(): page.evaluate(CARD_JS, [t, list(lines)])  # noqa: E731

        await page.goto(url, wait_until="domcontentloaded")
        await card("CrowdSentinel", ["A crowd safety digital twin for police control rooms",
                                     "It knows where it is blind, and sends the drone there"])
        await control(page, restart=True, speed=1)
        await page.wait_for_timeout(300)
        await control(page, paused=True)
        await page.wait_for_timeout(4500)
        await card(None)
        await control(page, paused=False, speed=12)

        await cap("Live twin of a 4,000-person rally. Each 5 m zone shows the estimated crowd density.")
        await page.wait_for_timeout(6000)
        await cap("Data comes from CCTV, a drone, cell towers and WiFi, fused by a Kalman filter.")
        await page.wait_for_timeout(5500)
        await page.check("#lyr-cams")
        await cap("Cameras cover only part of the ground. The front-left of the stage has none.")
        await page.wait_for_timeout(5500)
        await page.uncheck("#lyr-cams")
        await cap("Hatching marks zones where the twin is unsure. Nobody has looked there recently.")
        await page.wait_for_timeout(6000)
        await cap("The drone (dashed square) is sent to the biggest blind spot every 10 seconds.")
        await page.wait_for_timeout(6500)
        await control(page, speed=20)
        await cap("Fast-forward: the crowd starts pushing towards the stage.")
        await wait_event(page, 500)
        await control(page, speed=4)
        await cap("The surge builds in the front-left, exactly where there is no camera.")
        await wait_event(page, 532)
        await control(page, paused=True)
        xy = await page.evaluate(ZONE_XY_JS, 2)
        if xy:
            await page.mouse.move(*xy)
        await cap("The twin still sees it: density, uncertainty and time until the zone turns critical.")
        await page.wait_for_timeout(6500)
        await page.mouse.move(W - 40, 40)
        await cap("Alerts on the right: critical now, predicted, and zones that need eyes on them.")
        await page.wait_for_timeout(5500)
        await control(page, paused=False, speed=4)
        await wait_event(page, 548)
        await control(page, paused=True)
        await page.check("#lyr-evac")
        await cap("Evacuation routes avoid packed zones. The twin suggests opening the closed gate G4.")
        await page.wait_for_timeout(6000)
        btn = page.locator("button[data-gate]")
        if await btn.count():
            await btn.first.scroll_into_view_if_needed()
            await btn.first.click(timeout=5000)
            await page.wait_for_timeout(1500)
        await cap("One click updates the plan: G4 open, exit times recomputed.")
        await page.wait_for_timeout(5000)
        await page.uncheck("#lyr-evac")
        await page.check('input[name="base"][value="truth"]')
        await cap("For evaluation, compare with the simulated truth...")
        await page.wait_for_timeout(4000)
        await page.check('input[name="base"][value="density"]')
        await cap("...and the twin's estimate. Close, even in zones no camera can see.")
        await page.wait_for_timeout(4500)
        k = await page.evaluate("+document.getElementById('scrub').value")
        await cap("Every tick is stored, so the control room can replay the event.")
        for frac in (0.5, 0.62, 0.74, 0.86):
            await page.evaluate("(v) => { const s = document.getElementById('scrub'); s.value = v; "
                                "s.dispatchEvent(new Event('input')); }", int(k * frac))
            await page.wait_for_timeout(1400)
        if await page.is_visible("#btn-live"):
            await page.click("#btn-live")
        await cap(None)
        await card("Results on 5 simulated rallies", [
            "Density error 73% lower than a CCTV-only dashboard",
            "76% lower in zones with no camera",
            "Active drone guidance: 49% lower blind-spot error than a fixed patrol",
            "25 of 25 critical zones detected, about 69 s early (CCTV only: 3 of 25)",
        ])
        await page.wait_for_timeout(7000)
        await control(page, paused=False, speed=4)
        await ctx.close()
        await browser.close()

    webm = glob.glob(os.path.join(tmp, "*.webm"))[0]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", webm, "-ss", "0.6", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "slow", "-movflags", "+faststart", out],
                       check=True)
    else:
        shutil.copy(webm, os.path.splitext(out)[0] + ".webm")
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8050")
    ap.add_argument("--out", default="docs/demo.mp4")
    a = ap.parse_args()
    asyncio.run(record(a.url, a.out))


if __name__ == "__main__":
    main()
