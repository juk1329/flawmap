"""Plate model fitted to the capture (no reference drawing), and the public-data camera reader."""
from pathlib import Path

import numpy as np
import pytest

from flawmap.anchor.parametric import build_device_model
from flawmap.anchor.template import fit_dimensions, identify, plate_box, template_spec
from flawmap.geometry.camera import Camera
from flawmap.geometry.transforms import inv_T, look_at_opencv, make_T, rot_z
from flawmap.poses.boardless import axes_meeting_point
from flawmap.surface.raycast import RaycastScene


def _silhouette_views(spec, T_world_cad, n=10, size=(800, 600), f=900.0):
    model = build_device_model(spec, corner_step=0.5, edge_step=2.0, profile_step=0.3)
    scene = RaycastScene(model.mesh, T_world_cad)
    cam = Camera(np.array([[f, 0, (size[0] - 1) / 2], [0, f, (size[1] - 1) / 2], [0, 0, 1.0]]), *size)
    frames, cams, Twc = {}, {}, {}
    for k in range(n):
        a = np.radians(360.0 * k / n)
        e = np.radians(25.0 + 35.0 * (k % 3) / 2)
        eye = 330.0 * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        T = look_at_opencv(eye, np.zeros(3))
        h = scene.cast(cam, T, step=1)
        img = np.where(h.hit, 200, 60).astype(np.uint8)
        frames[k] = np.repeat(img[..., None], 3, axis=2)
        cams[k], Twc[k] = cam, T
    return frames, cams, Twc


def test_fit_dimensions_recovers_plate_size():
    truth = template_spec(150.9, 75.7, 8.3, 10.0)          # iPhone-11-sized plate
    T_true = make_T(rot_z(20.0), [3.0, -2.0, 4.15])
    frames, cams, Twc = _silhouette_views(truth, T_true)
    start = template_spec(150.9 * 1.05, 75.7 * 0.95, 8.3 * 1.05, 8.0)
    T0 = make_T(rot_z(21.0), [4.0, -1.0, 4.15])
    spec, T, info = fit_dimensions(start, frames, cams, Twc, T0, sorted(frames), log=lambda m: None)
    L, W, Th = info["size_mm"]
    # one pixel is ~0.37 mm here (f 900 px at 330 mm) and the silhouettes are hard-edged binary images,
    # so ~1.5 px is the bar (the rendered synthetic capture came back within 0.2 mm)
    assert abs(L - 150.9) < 0.55 and abs(W - 75.7) < 0.55 and abs(Th - 8.3) < 0.55
    assert np.linalg.norm(T[:3, 3] - T_true[:3, 3]) < 0.5


def test_identify_by_size():
    assert identify(146.8, 71.6).slug == "iphone-12"
    assert identify(150.8, 75.6).slug == "iphone-11"
    assert identify(160.0, 77.0) is None


def test_plate_box_axes_and_size():
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-35, 35, 20000), rng.uniform(-73, 73, 20000), rng.uniform(-3.7, 3.7, 20000)])
    R = rot_z(30.0)
    world = pts @ R.T + [10.0, 5.0, 0.0]
    cams = np.array([[0.0, 0.0, 300.0]])
    T, size = plate_box(world, cams, "screen")
    assert np.allclose(size[:2], [146, 70], atol=3.0)         # smallest rectangle around the face points
    assert T[:3, 2] @ [0, 0, 1] > 0.99                      # screen normal towards the cameras
    assert abs(abs(T[:3, 1] @ (R @ [0, 1.0, 0])) - 1) < 1e-3     # length along the long axis


def test_axes_meeting_point():
    c = np.array([5.0, -3.0, 2.0])
    Ts = [look_at_opencv(c + 300 * np.array([np.cos(a), np.sin(a), 0.6]), c) for a in np.linspace(0, 6, 12)]
    assert np.allclose(axes_meeting_point(Ts), c, atol=1e-6)


UCO3D = Path(__file__).resolve().parents[2] / "data" / "datasets" / "uco3d"


@pytest.mark.skipif(not (UCO3D / "metadata.sqlite").exists(), reason="uCO3D not downloaded")
def test_uco3d_cameras_project_points_into_masks():
    from flawmap.io.uco3d import load_sequence, mask_agreement

    seq = load_sequence(UCO3D, "59015-27660-21826")
    assert (seq.width, seq.height) == (1080, 1920)
    assert mask_agreement(seq, seq.frame_numbers[::50]) > 0.9


def test_border_lines_dropped_cracks_kept():
    from flawmap.detect.appearance import Region
    from flawmap.pipeline import drop_border_lines
    from flawmap.surface.maps import ChartMaps, ViewSamples, chart_geometry

    model = build_device_model(template_spec(146.7, 71.5, 7.6, 10.0), corner_step=0.5, edge_step=2.0, profile_step=0.3)
    g = chart_geometry(model, "screen", res=0.5)
    cm = ChartMaps(g, ViewSamples([], np.zeros((0, len(g.point), 3), np.uint8), np.zeros((0, len(g.point)), np.float16)))
    uv = g.uv_of(np.arange(len(g.point)))

    def texels(mask):
        return np.flatnonzero(mask)

    border = texels((np.abs(uv[:, 0] - 4.0) < 0.6) & (uv[:, 1] > 20) & (uv[:, 1] < 110))       # display edge, 4 mm in
    diag = texels((np.abs((uv[:, 0] - 20) - 0.6 * (uv[:, 1] - 40)) < 0.6) & (uv[:, 1] > 40) & (uv[:, 1] < 90))
    short = texels((np.abs(uv[:, 0] - 4.0) < 0.6) & (uv[:, 1] > 60) & (uv[:, 1] < 66))       # short line at the edge
    regs = [Region("screen", "crack", t, 10.0) for t in (border, diag, short)]
    kept, dropped = drop_border_lines(regs, {"screen": cm}, model)
    assert len(dropped) == 1 and dropped[0]["length_mm"] > 80
    assert [len(r.texels) for r in kept] == [len(diag), len(short)]
