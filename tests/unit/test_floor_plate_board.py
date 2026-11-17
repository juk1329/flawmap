"""``pipeline.measure_pass_plate`` leaves passes that show the printed board to the board path.

On a mat the floor mosaic measures the paper (417 x 291 mm on the synthetic A3 capture), and
``--device auto`` then ended with a paper-sized plate model instead of the iPhone 12 (2026-11-17).
The mosaic itself is replaced by a stand-in here; only the board check is under test.
"""
import cv2
import numpy as np
import pytest

from flawmap.geometry.camera import Camera
from flawmap.geometry.transforms import look_at_opencv
from flawmap.pipeline import PassInput, measure_pass_plate
from flawmap.poses.board import BoardSpec, render_board_page
from flawmap.synth.render_capture import CV_TO_ARKIT_CAM, T_ARK_WORLD_BASE, write_stray


def _pass(root, frame):
    n = 12
    h, w = frame.shape[:2]
    cam = Camera(np.array([[900.0, 0, (w - 1) / 2], [0, 900.0, (h - 1) / 2], [0, 0, 1]]), w, h)
    depths = [np.full((h, w), 400.0, np.float32) for _ in range(n)]
    ark = [T_ARK_WORLD_BASE @ look_at_opencv(np.array([300 * np.cos(a), 300 * np.sin(a), 250.0]), np.zeros(3)) @ CV_TO_ARKIT_CAM
           for a in np.linspace(0, 5.5, n)]
    write_stray(root, cam, [frame] * n, depths, ark, fps=10.0)
    return PassInput(root, "back")


@pytest.fixture()
def stand_in(monkeypatch):
    import flawmap.anchor.floor_outline as fo
    import flawmap.poses.boardless as bl

    monkeypatch.setattr(bl, "floor_height", lambda *a, **k: 0.0)
    monkeypatch.setattr(fo, "measure_plate", lambda *a, **k: "measured")


def test_board_in_view_skips_the_floor_plate(tmp_path, stand_in):
    spec = BoardSpec.load()
    page = np.asarray(render_board_page(spec))
    frame = cv2.resize(page, (1280, 906), interpolation=cv2.INTER_AREA)[:904]
    p = _pass(tmp_path / "board", frame)
    assert measure_pass_plate(p, n_views=12, board=spec) is None
    assert measure_pass_plate(p, n_views=12) == "measured"          # no board given: measured as before


def test_no_board_in_view_measures(tmp_path, stand_in):
    rng = np.random.default_rng(0)
    wood = cv2.GaussianBlur((rng.random((904, 1280, 3)) * 255).astype(np.uint8), (0, 0), 3)
    p = _pass(tmp_path / "floor", wood)
    assert measure_pass_plate(p, n_views=12, board=BoardSpec.load()) == "measured"
