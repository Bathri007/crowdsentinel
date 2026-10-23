"""Run density + optical flow on a crowd video and save an annotated copy.

    python -m vision.run_video --video data/raw/crowd.mp4 --weights vision/weights/model_sh_A.pth

Output: data/raw/<name>_annotated.mp4 with a density heat map, motion arrows
and the live count; plus a per-second CSV of count and mean crowd pressure.
"""
from __future__ import annotations

import argparse
import csv
import os

import cv2
import numpy as np

from .flow import FlowEstimator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default="vision/weights/model_sh_A.pth")
    ap.add_argument("--rate", type=float, default=2.0, help="analysed frames per second")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = whole video)")
    a = ap.parse_args()

    from .density import DensityEstimator

    est = DensityEstimator(a.weights)
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    base = os.path.splitext(a.video)[0]
    out = cv2.VideoWriter(base + "_annotated.mp4", cv2.VideoWriter_fourcc(*"mp4v"), a.rate, (w, h))
    flow = FlowEstimator()
    every = max(1, int(round(fps / a.rate)))
    rows, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        if a.seconds and t > a.seconds:
            break
        if i % every == 0:
            dmap = est.predict(frame)
            count = float(dmap.sum())
            fl = flow.update(frame, t)
            heat = cv2.resize(dmap / max(dmap.max(), 1e-6), (w, h))
            heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
            vis = cv2.addWeighted(frame, 0.6, heat, 0.4, 0)
            pressure = 0.0
            if fl is not None:
                mx, my, var = fl
                R, C = mx.shape
                dens_cells = cv2.resize(dmap, (C, R), interpolation=cv2.INTER_AREA) * (dmap.size / (R * C))
                pressure = float((dens_cells * var).mean())
                for r in range(R):
                    for c in range(C):
                        x0, y0 = int((c + 0.5) * w / C), int((r + 0.5) * h / R)
                        x1, y1 = int(x0 + mx[r, c] * 0.5), int(y0 + my[r, c] * 0.5)
                        cv2.arrowedLine(vis, (x0, y0), (x1, y1), (255, 255, 255), 2, tipLength=0.3)
            cv2.putText(vis, f"t={t:5.1f}s  people~{count:6.0f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2)
            out.write(vis)
            rows.append({"t": round(t, 2), "count": round(count, 1), "pressure": round(pressure, 3)})
            print(f"t={t:6.1f}s count={count:7.1f}")
        i += 1
    cap.release()
    out.release()
    with open(base + "_counts.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["t", "count", "pressure"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"saved {base}_annotated.mp4 and {base}_counts.csv")


if __name__ == "__main__":
    main()
