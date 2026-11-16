"""Small synthetic capture through the real stages: board poses, registration, maps, detection."""
import json

import numpy as np
import pytest

from flawmap.anchor.parametric import DeviceSpec, build_device_model, rest_pose
from flawmap.anchor.register import register_device
from flawmap.detect.appearance import detect_all
from flawmap.geometry.transforms import pose_difference
from flawmap.io.stray import iter_frames, read_stray
from flawmap.poses.align import estimate_poses
from flawmap.poses.board import BoardSpec
from flawmap.surface.maps import chart_geometry, collect_samples, compute_stats
from flawmap.synth.render_capture import SynthConfig, render_capture


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    cfg = SynthConfig(device="iphone-12", face_up="back", width=960, height=720, f_px=730.0, rings=[(25.0, 12), (55.0, 8)],
                      top_frames=3, seed=3, decals=[{"kind": "chip", "chart": "frame", "u": 100.0, "v": 3.5, "size": 2.5, "seed": 1}])
    render_capture(cfg, out, progress=False)
    return out


@pytest.mark.slow
def test_poses_registration_and_chip(capture):
    cap = read_stray(capture)
    gt = json.loads((capture / "gt.json").read_text())
    frames = dict(iter_frames(capture / "rgb.mp4"))
    cams = {i: cap.camera(i) for i in frames}
    P = estimate_poses(frames, cams, BoardSpec.load(), cap.T_ark_cam)
    assert len(P.frames) == len(frames)
    errs = [pose_difference(np.array(gt["frames"][i]["T_world_cam"]), P.T_world_cam[i]) for i in P.frames if P.source[i] == "board"]
    assert len(errs) >= 0.8 * len(frames)
    assert np.median([e[1] for e in errs]) < 0.4 and np.median([e[0] for e in errs]) < 0.05
    model = build_device_model(DeviceSpec.by_slug("iphone-12"))
    reg = register_device(model, frames, cams, P.T_world_cam, rest_pose(model, "back"), searches=(12.0, 5.0, 3.0), log=lambda *a: None)
    dr, dt = pose_difference(np.array(gt["T_world_cad"]), reg.T_world_cad)
    assert dr < 0.15 and dt < 0.4
    geoms = {c: chart_geometry(model, c) for c in ("back", "frame")}
    maps = collect_samples(model, geoms, frames, cams, P.T_world_cam, reg.T_world_cad)
    for cm in maps.values():
        compute_stats(cm)
    regions, _ = detect_all(maps, model)
    chips = [r for r in regions if r.kind == "chip"]
    assert len(chips) == 1
    u, v = maps["frame"].geom.uv_of(chips[0].texels).mean(0)
    assert abs(u - 100.0) < 1.0 and abs(v - 3.5) < 1.0
