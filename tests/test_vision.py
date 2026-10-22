"""Vision tests that need no model weights (a fake density model stands in for DM-Count)."""
import numpy as np

from simulation.venue import rally_ground
from vision.camera_source import VideoCameraSource
from vision.flow import FlowEstimator
from vision.homography import GroundMapper


def _blob_frame(x, w=320, h=240):
    f = np.zeros((h, w, 3), np.uint8)
    rng = np.random.default_rng(0)
    tex = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
    f[90:150, x:x + 60] = tex
    return f


def test_optical_flow_sees_rightward_motion():
    fe = FlowEstimator(grid=(4, 4), work_width=320)
    assert fe.update(_blob_frame(100), 0.0) is None
    mx, my, var = fe.update(_blob_frame(106), 0.5)       # moved 6 px in 0.5 s -> 12 px/s
    r, c = 1, 1                                          # cell containing the blob
    assert mx[1:3, 1:3].max() > 6
    assert abs(my[r, c]) < 3


def test_homography_round_trip():
    img = [[0, 480], [640, 480], [640, 0], [0, 0]]
    ground = [[20, 30], [60, 30], [60, 60], [20, 60]]
    gm = GroundMapper(img, ground)
    assert gm.reprojection_error(img, ground) < 1e-6
    centre = gm.to_ground(np.array([[320, 240]]))[0]
    assert np.allclose(centre, [40, 45])


class FakeDensity:
    """Pretends every 8x8 pixel block holds 0.01 people."""

    def predict(self, frame):
        return np.full((frame.shape[0] // 8, frame.shape[1] // 8), 0.01)


def test_camera_source_emits_measurements_that_add_up():
    venue = rally_ground()
    gm = GroundMapper([[0, 480], [640, 480], [640, 0], [0, 0]], [[20, 30], [60, 30], [60, 60], [20, 60]])
    src = VideoCameraSource("CAMX", "unused.mp4", gm, venue, FakeDensity())
    frame = np.zeros((480, 640, 3), np.uint8)
    ms = src.process_frame(frame, 0.0)
    total = sum(m.value for m in ms)
    expected = 0.01 * (480 // 8) * (640 // 8)
    assert len(ms) > 20
    assert abs(total - expected) / expected < 0.15      # small loss: cells at the edge of visible zones
    assert all(m.kind == "cctv" and len(m.zones) == 1 for m in ms)
