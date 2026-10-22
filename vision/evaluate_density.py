"""Measure density-model accuracy on ShanghaiTech (MAE / RMSE of people counts).

1. Download ShanghaiTech (Part A and B) - e.g. from Kaggle "shanghaitech" - and
   unzip so you have:  data/raw/ShanghaiTech/part_A/test_data/images/IMG_1.jpg
                       data/raw/ShanghaiTech/part_A/test_data/ground-truth/GT_IMG_1.mat
2. Put DM-Count weights in vision/weights/model_sha.pth
3. Run:  python -m vision.evaluate_density --root data/raw/ShanghaiTech/part_A

Published DM-Count numbers for comparison (Wang et al. 2020): Part A MAE 59.7, RMSE 95.7;
Part B MAE 7.4, RMSE 11.8. Expect similar, not identical (we resize very large images).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import cv2
import numpy as np


def load_gt_count(mat_path: str) -> int:
    from scipy.io import loadmat

    m = loadmat(mat_path)
    pts = m["image_info"][0][0][0][0][0]
    return int(len(pts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g. data/raw/ShanghaiTech/part_A")
    ap.add_argument("--weights", default="vision/weights/model_sha.pth")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (quick check)")
    ap.add_argument("--max-side", type=int, default=2048)
    a = ap.parse_args()

    from .density import DensityEstimator

    est = DensityEstimator(a.weights, max_side=a.max_side)
    imgs = sorted(glob.glob(os.path.join(a.root, "test_data", "images", "*.jpg")))
    if a.limit:
        imgs = imgs[: a.limit]
    if not imgs:
        raise SystemExit(f"no images found under {a.root}/test_data/images")
    errs, times = [], []
    for p in imgs:
        name = os.path.splitext(os.path.basename(p))[0]
        gt = load_gt_count(os.path.join(a.root, "test_data", "ground-truth", f"GT_{name}.mat"))
        img = cv2.imread(p)
        t0 = time.perf_counter()
        pred = float(est.predict(img).sum())
        times.append(time.perf_counter() - t0)
        errs.append(pred - gt)
        print(f"{name:12s} gt={gt:5d} pred={pred:8.1f}")
    e = np.array(errs)
    res = {
        "dataset": a.root,
        "images": len(e),
        "MAE": float(np.abs(e).mean()),
        "RMSE": float(np.sqrt((e ** 2).mean())),
        "seconds_per_image": float(np.mean(times)),
        "device": est.device,
    }
    print(json.dumps(res, indent=1))
    os.makedirs("experiments/results", exist_ok=True)
    tag = os.path.basename(os.path.normpath(a.root))
    with open(f"experiments/results/density_{tag}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
