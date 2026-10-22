"""Crowd density estimation with DM-Count (Wang et al., NeurIPS 2020).

Why a density model and not a person detector: in dense crowds people overlap
so much that detectors like YOLO miss most of them. A density model predicts a
heat map whose sum is the number of people, and works at 5+ persons/m2.

Pretrained weights (download once, put them in vision/weights/):
    https://github.com/cvlab-stonybrook/DM-Count  -> "Pretrained models"
    model_sha.pth  (ShanghaiTech Part A, dense crowds)  <- recommended
    model_qnrf.pth (UCF-QNRF, very dense crowds)

Usage:
    est = DensityEstimator("vision/weights/model_sha.pth")
    dmap = est.predict(frame_bgr)        # 2-D array, one value per 8x8 pixel block
    people = dmap.sum()
"""
from __future__ import annotations

import os

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # vision is optional; the twin runs without it
    torch = None
    nn = None

VGG19_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

if nn is not None:
    class DMCountNet(nn.Module):
        """Same layer names as the official DM-Count code so its weights load directly."""

        def __init__(self):
            super().__init__()
            layers, c_in = [], 3
            for v in VGG19_CFG:
                if v == "M":
                    layers.append(nn.MaxPool2d(2, 2))
                else:
                    layers += [nn.Conv2d(c_in, v, 3, padding=1), nn.ReLU(inplace=True)]
                    c_in = v
            self.features = nn.Sequential(*layers)
            self.reg_layer = nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True),
            )
            self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

        def forward(self, x):
            x = self.features(x)
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.reg_layer(x)
            return self.density_layer(x)


class DensityEstimator:
    def __init__(self, weights: str | None = None, device: str | None = None, max_side: int = 1024):
        if torch is None:
            raise ImportError("PyTorch is not installed. Run: pip install -r requirements-vision.txt")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_side = max_side
        self.net = DMCountNet().to(self.device).eval()
        self.loaded = False
        if weights:
            if not os.path.exists(weights):
                raise FileNotFoundError(
                    f"{weights} not found. Download DM-Count weights from "
                    "https://github.com/cvlab-stonybrook/DM-Count and place them in vision/weights/")
            state = torch.load(weights, map_location=self.device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            missing, unexpected = self.net.load_state_dict(state, strict=False)
            if missing:
                raise RuntimeError(f"weights do not match DM-Count layers, missing: {missing[:5]}")
            self.loaded = True

    def _prep(self, frame_bgr: np.ndarray):
        import cv2

        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        if scale < 1.0:
            frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = frame_bgr.shape[:2]
        # the network downsamples by 16 then upsamples by 2: pad to a multiple of 16
        ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
        frame_bgr = cv2.copyMakeBorder(frame_bgr, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        rgb = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
        rgb = (rgb - MEAN) / STD
        x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)
        return x, scale, (h, w)

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Density map at 1/8 of the (possibly resized) input resolution. Sum = people count."""
        x, scale, (h, w) = self._prep(frame_bgr)
        with torch.no_grad():
            mu = self.net(x)
        dmap = mu[0, 0].cpu().numpy()
        return dmap[: int(np.ceil(h / 8)), : int(np.ceil(w / 8))]

    @staticmethod
    def cell_size(frame_shape, dmap_shape) -> tuple[float, float]:
        """Size in ORIGINAL frame pixels of one density-map cell (for projecting to the ground)."""
        return frame_shape[0] / dmap_shape[0], frame_shape[1] / dmap_shape[1]
