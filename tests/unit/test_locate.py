"""``anchor.locate`` without SAM 2: the RANSAC floor plane, the masks' footprint and the plate fitted to it."""
import numpy as np

from flawmap.anchor.locate import box_masks, debias_footprint, fit_footprint, floor_extent, footprint, levelling, plausible
from flawmap.geometry.camera import Camera
from flawmap.geometry.transforms import look_at_opencv


def _plane_points(n=20000, tilt_deg=0.0, z0=-490.0, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-250, 250, (n, 2))
    t = np.radians(tilt_deg)
    z = z0 + np.tan(t) * xy[:, 0] + rng.normal(0, 0.8, n)
    return np.column_stack([xy, z])


def test_levelling_makes_the_plane_horizontal():
    pts = _plane_points(tilt_deg=2.0)
    c = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
    n = Vt[2] if Vt[2][2] > 0 else -Vt[2]
    T = levelling({"normal": n, "point": c})
    q = pts @ T[:3, :3].T + T[:3, 3]
    assert np.std(q[:, 2]) < 1.0                      # flat after levelling (noise 0.8 mm)


def _masks_of_box(L=146.7, W=71.5, H=7.4, n_views=16):
    """Exact silhouettes of a box lying on z = 0, seen from an orbit; returns masks, cams, poses."""
    cam = Camera(np.array([[1460.0, 0, 959.5], [0, 1460.0, 719.5], [0, 0, 1]]), 1920, 1440)   # masks ~1.1 mm/px as on the iPhone
    corners = np.array([[sx * W / 2, sy * L / 2, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (0.0, H)])
    masks, cams, T = {}, {}, {}
    import cv2
    for k, a in enumerate(np.linspace(0, 2 * np.pi, n_views, endpoint=False)):
        Twc = look_at_opencv(np.array([300 * np.cos(a), 300 * np.sin(a), 280.0]), np.zeros(3))
        Tcw = np.linalg.inv(Twc)
        pc = corners @ Tcw[:3, :3].T + Tcw[:3, 3]
        uv = np.stack([cam.fx * pc[:, 0] / pc[:, 2] + cam.cx, cam.fy * pc[:, 1] / pc[:, 2] + cam.cy], 1)
        m = np.zeros((360, 480), np.uint8)
        hull = cv2.convexHull(((uv + 0.5) * 0.25 - 0.5).astype(np.float32))
        cv2.fillConvexPoly(m, np.round(hull * 16).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
        masks[k], cams[k], T[k] = m > 0, cam, Twc
    return masks, cams, T


def test_footprint_of_box_masks_gives_its_size():
    masks, cams, T = _masks_of_box()
    share, seen = footprint(masks, cams, T, 3.7, -120.0, 120.0, 0.5, 480)
    # exact masks: the visual hull (nearly every view) gives the size
    plate, fp = fit_footprint(share, seen, -120.0, 120.0, 0.5, min_share=0.95, min_views=8)
    assert plate is not None and plausible(plate)
    assert abs(plate.length - 146.7) < 3.0 and abs(plate.width - 71.5) < 3.0
    assert np.hypot(*plate.centre_xy) < 1.5
    # the default share (0.6, for drifting ARKit poses) is biased outwards by the floor hidden behind
    # the device from the far side: here about half the thickness x cot(43 deg) per side
    loose, _ = fit_footprint(share, seen, -120.0, 120.0, 0.5, min_views=8)
    assert 0.0 < loose.length - 146.7 < 8.0 and 0.0 < loose.width - 71.5 < 8.0
    assert min(abs(plate.long_axis_deg - 90.0), abs(plate.long_axis_deg - 270.0)) < 2.0   # long side along world y


def test_floor_extent_of_a_mask():
    masks, cams, T = _masks_of_box(n_views=4)
    ext = floor_extent(masks[0], cams[0], T[0], 0.0)
    assert ext is not None and 140.0 < ext["length"] < 175.0 and 60.0 < ext["width"] < 95.0


def test_debias_takes_the_hulls_own_excess_off():
    # a 146.7 x 71.5 x 7.4 mm box seen from two orbit rings (22 and 50 deg, as the synthetic captures)
    from flawmap.anchor.floor_outline import FloorPlate

    cam = Camera(np.array([[1460.0, 0, 959.5], [0, 1460.0, 719.5], [0, 0, 1]]), 1920, 1440)
    true = FloorPlate(146.7, 71.5, 10.0, 7.4, np.zeros(2), 90.0, 1.0, {})
    cams, T = {}, {}
    for k, (el, a) in enumerate([(el, a) for el in (22.0, 50.0) for a in np.linspace(0, 2 * np.pi, 14, endpoint=False)]):
        e = np.radians(el)
        cams[k], T[k] = cam, look_at_opencv(np.array([300 * np.cos(e) * np.cos(a), 300 * np.cos(e) * np.sin(a), 300 * np.sin(e)]), np.zeros(3))
    masks = box_masks(true, cams, T, list(cams), 0.0, 7.4)
    share, seen = footprint(masks, cams, T, 3.7, -120.0, 120.0, 0.5, 480)
    measured, _ = fit_footprint(share, seen, -120.0, 120.0, 0.5)
    assert measured.length - 146.7 > 1.5 and measured.width - 71.5 > 1.5          # the hull's excess
    fixed, rec = debias_footprint(measured, masks, cams, T, 0.0, 7.4, 3.7, -120.0, 120.0, 0.5, 480)
    assert rec["used"]
    assert abs(fixed.length - 146.7) < 0.8 and abs(fixed.width - 71.5) < 0.8
