import cv2
import numpy as np
import pytest

from flawmap.geometry.camera import Camera
from flawmap.geometry.transforms import inv_T, look_at_opencv, pose_difference
from flawmap.poses.board import BoardDetector, BoardSpec, render_board_page


def _warp_page(page, spec, cam, T_wc):
    img = np.asarray(page)
    k = spec.px_per_mm
    ph, pw = img.shape[:2]
    A = np.array([[1 / k, 0, -pw / 2 / k], [0, -1 / k, ph / 2 / k], [0, 0, 1.0]])   # page px -> world (x, y, 1)
    T_cw = inv_T(T_wc)
    H = cam.K @ np.column_stack([T_cw[:3, 0], T_cw[:3, 1], T_cw[:3, 3]]) @ A
    return cv2.warpPerspective(img, H, (cam.width, cam.height), flags=cv2.INTER_AREA, borderValue=(90, 90, 90))


@pytest.fixture(scope="module")
def board():
    spec = BoardSpec.load()
    return spec, render_board_page(spec)


def test_grey_area_is_centred_and_drops_corners(board):
    spec, _ = board
    x0, x1, y0, y1 = spec.grey_rect_world()
    assert abs(x0 + x1) < 1e-9 and abs(y0 + y1) < 1e-9
    n_all = len(spec.cv_board().getChessboardCorners())
    assert 0 < len(spec.usable_corner_ids()) < n_all


@pytest.mark.parametrize("elev,az", [(65, 40), (25, 210)])
def test_pose_from_rendered_board(board, elev, az):
    spec, page = board
    cam = Camera(np.array([[1450.0, 0, 959.5], [0, 1450.0, 719.5], [0, 0, 1]]), 1920, 1440)
    e, a = np.radians(elev), np.radians(az)
    eye = 330.0 * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    T_wc = look_at_opencv(eye, np.zeros(3))
    gray = cv2.cvtColor(_warp_page(page, spec, cam, T_wc), cv2.COLOR_RGB2GRAY)
    bp = BoardDetector(spec).pose(gray, cam)
    assert bp is not None and bp.n_corners >= 30
    dr, dt = pose_difference(T_wc, bp.T_world_cam)
    assert dr < 0.1 and dt < 0.5
