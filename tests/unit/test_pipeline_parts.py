import json
import struct

import numpy as np
import pytest

from flawmap.anchor.parametric import DeviceSpec, build_device_model
from flawmap.detect.outline import outline_dents, outline_profile
from flawmap.export.bundle import atlas_uv, write_glb
from flawmap.geometry.camera import Camera
from flawmap.geometry.transforms import look_at_opencv, make_T, pose_difference, rot_x, rot_z
from flawmap.io.stray import read_stray
from flawmap.poses.align import align_arkit, rigid_fit
from flawmap.synth.render_capture import CV_TO_ARKIT_CAM, T_ARK_WORLD_BASE, write_stray


@pytest.fixture(scope="module")
def iphone12():
    return build_device_model(DeviceSpec.by_slug("iphone-12"))


def test_rigid_fit_recovers_transform():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(30, 3)) * 100
    T = make_T(rot_z(30) @ rot_x(10), [5.0, -3.0, 12.0])
    B = A @ T[:3, :3].T + T[:3, 3]
    assert np.allclose(rigid_fit(A, B), T, atol=1e-6)


def test_align_arkit_recovers_world_offset():
    rng = np.random.default_rng(1)
    T_ark_world = make_T(None, [100.0, 50.0, -30.0]) @ T_ARK_WORLD_BASE @ make_T(rot_z(40))
    board = {}
    ark = []
    for i in range(12):
        eye = 300 * np.array([np.cos(i), np.sin(i), 0.6])
        Twc = look_at_opencv(eye, np.zeros(3))
        board[i] = Twc
        ark.append(T_ark_world @ Twc)
    T, res = align_arkit(board, np.array(ark))
    dr, dt = pose_difference(T @ ark[3], board[3])
    assert dr < 1e-6 and dt < 1e-6 and res["pos_mm_max"] < 1e-6


def test_stray_roundtrip(tmp_path):
    cam = Camera(np.array([[500.0, 0, 159.5], [0, 500.0, 119.5], [0, 0, 1]]), 320, 240)
    frames = [np.full((240, 320, 3), 40 * k, np.uint8) for k in range(3)]
    depths = [np.full((240, 320), 300.0, np.float32) for _ in range(3)]
    Twc = [look_at_opencv(np.array([100.0 * k, -200.0, 250.0]), np.zeros(3)) for k in range(3)]
    ark = [T_ARK_WORLD_BASE @ T @ CV_TO_ARKIT_CAM for T in Twc]
    write_stray(tmp_path, cam, frames, depths, ark, fps=10.0)
    cap = read_stray(tmp_path)
    assert len(cap) == 3 and (cap.width, cap.height) == (320, 240)
    for k in range(3):
        dr, dt = pose_difference(cap.T_ark_cam[k], T_ARK_WORLD_BASE @ Twc[k])
        assert dr < 1e-3 and dt < 1e-2               # metres in the csv: 6 decimals = 1 micron
    assert np.allclose(cap.K[0], cam.K)
    assert cap.depth(0).shape == (192, 256)


def test_outline_profile_finds_inward_bump_only(iphone12):
    rng = np.random.default_rng(2)
    P = iphone12.perimeter
    s = rng.uniform(0, P, 60000)
    r = rng.normal(0, 0.05, len(s)) + 0.2 * np.sin(2 * np.pi * s / P)      # slow model mismatch
    flat = outline_dents(outline_profile(iphone12, s, r))
    assert flat == []
    bump = r + 0.7 * np.clip(1 - ((s - 150.0) / 3.0) ** 2, 0, None) ** 2   # 0.7 mm dent at s = 150
    dents = outline_dents(outline_profile(iphone12, s, bump))
    assert len(dents) == 1 and abs(dents[0]["s_peak"] - 150.0) < 1.5 and dents[0]["peak_mm"] > 0.4


def test_glb_uvs_are_top_left(tmp_path, iphone12):
    write_glb(iphone12, tmp_path / "m.glb")
    b = (tmp_path / "m.glb").read_bytes()
    n = struct.unpack("<I", b[12:16])[0]
    j = json.loads(b[20:20 + n])
    acc = j["accessors"][j["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"]]
    bv = j["bufferViews"][acc["bufferView"]]
    off = 20 + n + 8 + bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    uv_file = np.frombuffer(b[off:off + acc["count"] * 8], np.float32).reshape(-1, 2)
    uv = atlas_uv(iphone12)
    assert uv.min() >= 0 and uv.max() <= 1
    assert np.allclose(uv_file, uv, atol=1e-5)


def test_coverage_counts_unseen_charts_and_weights_by_area(iphone12):
    from flawmap.surface.maps import ChartMaps, ViewSamples, chart_geometry, coverage

    res = {"screen": 1.0, "back": 1.0, "frame": 0.5}
    geoms = {c: chart_geometry(iphone12, c, res=r) for c, r in res.items()}
    maps = {}
    for c in ("back", "frame"):        # a back-up-only capture: the screen chart was never sampled
        g = geoms[c]
        cm = ChartMaps(g, ViewSamples([], np.zeros((0, len(g.point), 3), np.uint8), np.zeros((0, len(g.point)), np.float16)))
        n_good = np.full(len(g.point), 5, np.int32)
        if c == "frame":
            n_good[: len(n_good) // 2] = 0
        cm.stats = {"n_good": n_good}
        maps[c] = cm
    cov = coverage(maps, iphone12, geoms)
    area = {c: len(g.point) * g.res ** 2 for c, g in geoms.items()}
    expect = (area["back"] + area["frame"] * (len(geoms["frame"].point) - len(geoms["frame"].point) // 2)
              / len(geoms["frame"].point)) / sum(area.values())
    assert cov["overall"] == pytest.approx(expect, rel=1e-6)
    assert cov["overall"] < 0.6                  # the old texel count without the screen said ~0.9
    assert cov["by_chart"]["screen"] == {"seen": 0.0, "area_mm2": pytest.approx(area["screen"], abs=0.1), "captured": False}
    assert cov["by_chart"]["back"]["seen"] == 1.0
    assert cov["by_part"]["screen"] < 0.05
