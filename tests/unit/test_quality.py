"""The ``--quality max`` parts: sharp-view selection on chart samples, the COLMAP similarity, and
COLMAP failing quietly."""
import cv2
import numpy as np

from flawmap.geometry.camera import Camera
from flawmap.geometry.transforms import make_T, rot_z
from flawmap.poses.colmap import refine_with_colmap, similarity
from flawmap.surface.maps import ChartGeometry, ChartMaps, ViewSamples, prefer_sharp_views


def _chart(rows=120, cols=100):
    hit = np.ones((rows, cols), bool)
    n = rows * cols
    return ChartGeometry("back", 0.1, (rows, cols), hit, np.zeros((n, 3)), np.tile([0, 0, -1.0], (n, 1)),
                         np.zeros(n, int), np.zeros(n, int), np.arange(n))


def test_prefer_sharp_views_keeps_the_sharp_ones():
    g = _chart()
    rng = np.random.default_rng(0)
    base = (rng.random((120, 100)) > 0.5).astype(np.float32) * 200 + 20       # fine texture
    V = 20
    sharp = {2, 5, 11, 17, 19, 8}
    imgs = [base if v in sharp else cv2.GaussianBlur(base, (0, 0), 2.5) for v in range(V)]
    rgb = np.stack([np.repeat(im.reshape(-1, 1), 3, 1) for im in imgs]).astype(np.uint8)
    cos = np.full((V, g.shape[0] * g.shape[1]), 0.9, np.float16)
    cm = ChartMaps(g, ViewSamples(list(range(V)), rgb, cos))
    info = prefer_sharp_views(cm, keep_share=0.3, min_views=3)
    assert info["applied"] and info["kept"] == 6
    used = set(np.flatnonzero(cm.samples.cos[:, 0] > 0))
    assert used == sharp


def test_prefer_sharp_views_keeps_all_where_too_few_sharp_views_saw():
    g = _chart()
    V = 12
    rng = np.random.default_rng(1)
    base = (rng.random((120, 100)) > 0.5).astype(np.float32) * 200 + 20
    imgs = [base if v < 4 else cv2.GaussianBlur(base, (0, 0), 2.5) for v in range(V)]
    rgb = np.stack([np.repeat(im.reshape(-1, 1), 3, 1) for im in imgs]).astype(np.uint8)
    cos = np.full((V, 120 * 100), 0.9, np.float16)
    cos[:4, : 120 * 50] = 0             # the sharp views only saw the lower half
    cm = ChartMaps(g, ViewSamples(list(range(V)), rgb, cos))
    prefer_sharp_views(cm, keep_share=0.34, min_views=3)
    assert (cm.samples.cos[:, 0] > 0).sum() == 8                 # upper half: no sharp view, all kept
    assert set(np.flatnonzero(cm.samples.cos[:, -1] > 0)) == {0, 1, 2, 3}


def test_similarity_recovers_scale_rotation_translation():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(30, 3))
    R = rot_z(37.0) @ make_T(np.array([[1, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]))[:3, :3]
    B = (76.3 * (R @ A.T)).T + np.array([10.0, -4.0, 250.0])
    s, R2, t = similarity(A, B)
    assert abs(s - 76.3) < 1e-6 and np.allclose(R2, R, atol=1e-8) and np.allclose(t, [10.0, -4.0, 250.0], atol=1e-6)


def test_colmap_failure_keeps_the_input_poses():
    cam = Camera(np.array([[60.0, 0, 31.5], [0, 60.0, 23.5], [0, 0, 1]]), 64, 48)
    frames = {i: np.full((48, 64, 3), 100, np.uint8) for i in range(3)}      # nothing to match
    T = {i: make_T(np.eye(3), [i * 10.0, 0, 0]) for i in range(3)}
    out, info = refine_with_colmap(frames, {i: cam for i in frames}, T, log=lambda m: None)
    assert out is None and info["used"] is False and info.get("reason")
