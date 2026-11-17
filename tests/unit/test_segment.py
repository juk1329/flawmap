"""Device-mask helpers (no SAM 2 needed): IoU, box, inside, mask filtering of colour samples, gate."""
import numpy as np

from flawmap.anchor.parametric import DeviceSpec, build_device_model, rest_pose
from flawmap.geometry.camera import Camera
from flawmap.geometry.transforms import inv_T, look_at_opencv
from flawmap.segment.device_masks import box_of, inside, mask_iou, silhouette
from flawmap.surface.maps import chart_geometry, collect_samples
from flawmap.surface.raycast import RaycastScene


def test_iou_box_inside():
    a = np.zeros((100, 100), bool)
    a[20:60, 30:70] = True
    b = np.zeros_like(a)
    b[30:70, 30:70] = True
    assert abs(mask_iou(a, a) - 1.0) < 1e-9
    assert abs(mask_iou(a, b) - (30 * 40) / (50 * 40)) < 1e-9
    x0, y0, x1, y1 = box_of(a, pad=0.0)
    assert (x0, y0, x1, y1) == (30, 20, 69, 59)
    assert inside(a, np.array([40.0, 5.0]), np.array([40.0, 5.0])).tolist() == [True, False]


def test_masks_keep_floor_out_of_samples():
    model = build_device_model(DeviceSpec.by_slug("iphone-12"))
    T_wc = rest_pose(model, "back")
    cam = Camera(np.array([[700.0, 0, 319.5], [0, 700.0, 239.5], [0, 0, 1]]), 640, 480)
    T = look_at_opencv(np.array([0.0, -60.0, 260.0]), np.zeros(3))      # world = model rest frame
    Tcad = inv_T(T_wc) @ T
    sil = silhouette(RaycastScene(model.mesh, None), cam, Tcad, scale=1.0)
    frame = np.full((480, 640, 3), 200, np.uint8)
    geom = {"back": chart_geometry(model, "back", res=0.5)}
    full = collect_samples(model, geom, {1: frame}, {1: cam}, {1: Tcad}, np.eye(4))["back"]
    seen = int((full.samples.cos[0] > 0).sum())
    assert seen > 1000
    # a mask that only covers the left half of the device: samples elsewhere are dropped
    half = sil.copy()
    xs = np.nonzero(sil.any(0))[0]
    half[:, (xs.min() + xs.max()) // 2:] = False
    part = collect_samples(model, geom, {1: frame}, {1: cam}, {1: Tcad}, np.eye(4), masks={1: half}, mask_grow_px=0)["back"]
    kept = int((part.samples.cos[0] > 0).sum())
    assert 0.3 * seen < kept < 0.7 * seen
    # the full silhouette as the mask keeps (nearly) everything
    same = collect_samples(model, geom, {1: frame}, {1: cam}, {1: Tcad}, np.eye(4), masks={1: sil})["back"]
    assert (same.samples.cos[0] > 0).sum() >= 0.98 * seen


def test_batch_gate_on_mask_iou():
    from flawmap.batch import Manifest, result_gates

    m = Manifest(root=None, name="x", device="auto", board="none", square_mm=None, passes=[])
    gates = {"result": {"max_outline_px": 1.0, "min_coverage_both_faces": 0.9, "min_coverage_one_face": 0.5,
                        "expect_board_poses": True, "min_mask_iou": 0.95}}
    rep = {"coverage": {"overall": 0.99}, "passes": [
        {"name": "back", "face_up": "back", "pose_mode": "boardless", "registration": {"median_px": 0.7}, "mask_iou": {"iou_p50": 0.979}},
        {"name": "screen", "face_up": "screen", "pose_mode": "boardless", "registration": {"median_px": 0.8}, "mask_iou": {"iou_p50": 0.88}}]}
    probs = result_gates(rep, m, gates)
    assert len(probs) == 1 and "screen" in probs[0] and "mask IoU" in probs[0]
