"""Download what the vision benchmark needs (run on a normal internet connection).

    python -m tools.download_vision_data

1. DM-Count pretrained weights (official Google Drive folder from the DM-Count README)
   -> vision/weights/
2. ShanghaiTech Part A + B test images (two-part zip mirrored by ESIEE Paris for a
   crowd-counting course) -> data/raw/ShanghaiTech_parts/

The ShanghaiTech archive is split in two (.zip + .z01). Join it with 7-Zip
(open ShanghaiTech.zip) or `zip -s 0 ShanghaiTech.zip --out full.zip`, then
extract into data/raw/ so you get data/raw/ShanghaiTech/part_A_final/...
"""
from __future__ import annotations

import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(ROOT, "vision", "weights")
PARTS_DIR = os.path.join(ROOT, "data", "raw", "ShanghaiTech_parts")

DMCOUNT_FOLDER = "https://drive.google.com/drive/folders/10U7F4iW_aPICM5-qJq21SXLLkzlum9tX"
SHANGHAI_PARTS = {
    "ShanghaiTech.zip": "https://perso.esiee.fr/~chierchg/deep-learning/_downloads/"
                        "8fa2b5234fa2aee0fa9aae9f3d8e62ab/ShanghaiTech.zip",
    "ShanghaiTech.z01": "https://perso.esiee.fr/~chierchg/deep-learning/_downloads/"
                        "86a7f671c171459c741eb6c826748ef6/ShanghaiTech.z01",
}


def _download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        print(f"  already have {os.path.basename(path)}")
        return
    print(f"  downloading {os.path.basename(path)} ...", flush=True)
    tmp = path + ".part"

    def hook(blocks, bs, total):
        if total > 0 and blocks % 200 == 0:
            print(f"    {min(100, blocks * bs * 100 // total)}%", flush=True)

    urllib.request.urlretrieve(url, tmp, hook)
    os.replace(tmp, path)
    print(f"  done: {os.path.getsize(path) / 1e6:.0f} MB")


def weights():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    have = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".pth")]
    names = [f.lower().replace("_", "") for f in have]
    if any("sha" in n for n in names) and any("shb" in n for n in names):
        print(f"  weights already present: {have}")
        return True
    try:
        import gdown
    except ImportError:
        print("  gdown missing: pip install gdown")
        return False
    try:
        gdown.download_folder(DMCOUNT_FOLDER, output=WEIGHTS_DIR, quiet=False, use_cookies=False)
    except Exception as e:  # Drive sometimes rate-limits; say what to do instead
        print(f"  Google Drive download failed: {e}")
        print(f"  Open {DMCOUNT_FOLDER} in a browser and save the .pth files into {WEIGHTS_DIR}")
        return False
    # gdown may create a sub-folder: move .pth files up
    for dirpath, _, files in os.walk(WEIGHTS_DIR):
        for f in files:
            if f.endswith(".pth") and dirpath != WEIGHTS_DIR:
                os.replace(os.path.join(dirpath, f), os.path.join(WEIGHTS_DIR, f))
    print(f"  weights: {[f for f in os.listdir(WEIGHTS_DIR) if f.endswith('.pth')]}")
    return True


def dataset():
    os.makedirs(PARTS_DIR, exist_ok=True)
    for name, url in SHANGHAI_PARTS.items():
        _download(url, os.path.join(PARTS_DIR, name))
    return True


def main():
    print("[1/2] DM-Count weights")
    ok_w = weights()
    print("[2/2] ShanghaiTech images")
    try:
        ok_d = dataset()
    except Exception as e:
        print(f"  dataset download failed: {e}")
        ok_d = False
    print("RESULT weights=%s dataset=%s" % ("ok" if ok_w else "FAILED", "ok" if ok_d else "FAILED"))
    sys.exit(0 if (ok_w and ok_d) else 1)


if __name__ == "__main__":
    main()
