"""``flawmap batch``: manifests, intake gates, result gates and the summary (pipeline replaced)."""
import json
import shutil

import numpy as np
import pytest
import yaml

from flawmap.batch import find_devices, load_manifest, run_batch
from flawmap.geometry.camera import Camera
from flawmap.geometry.transforms import look_at_opencv
from flawmap.synth.render_capture import CV_TO_ARKIT_CAM, T_ARK_WORLD_BASE, write_stray


def _stray(root, n, fps=10.0, sweep=6.0):
    """A tiny Stray Scanner folder: ``n`` frames at ``fps``, camera going ``sweep`` radians round."""
    cam = Camera(np.array([[60.0, 0, 31.5], [0, 60.0, 23.5], [0, 0, 1]]), 64, 48)
    frames = [np.full((48, 64, 3), 3 * (k % 80), np.uint8) for k in range(n)]
    depths = [np.full((48, 64), 300.0, np.float32) for _ in range(n)]
    ark = [T_ARK_WORLD_BASE @ look_at_opencv(np.array([200 * np.cos(a), 200 * np.sin(a), 150.0]), np.zeros(3)) @ CV_TO_ARKIT_CAM
           for a in np.linspace(0, sweep, n)]
    write_stray(root, cam, frames, depths, ark, fps=fps)


@pytest.fixture(scope="module")
def recordings(tmp_path_factory):
    base = tmp_path_factory.mktemp("rec")
    _stray(base / "long", 50 * 10, fps=10.0)          # 50 s, full orbit
    _stray(base / "short", 20 * 10, fps=10.0)         # 20 s
    return base


def _device(root, name, rec, which, manifest_extra=None):
    d = root / name
    d.mkdir(parents=True)
    for face in ("screen", "back"):
        shutil.copytree(rec / which, d / f"{face}_orbit")
    man = {"name": name, "device": "auto", "board": "none",
           "passes": [{"face": "screen", "kind": "orbit", "path": "screen_orbit"},
                      {"face": "back", "kind": "orbit", "path": "back_orbit"}], **(manifest_extra or {})}
    (d / "manifest.yaml").write_text(yaml.safe_dump(man), encoding="utf-8")
    return d


def fake_runner(passes, m, out, gates, outline_px=0.6, coverage=0.97):
    out.mkdir(parents=True, exist_ok=True)
    rep = {"device": "template", "device_name": "추정 판형 모델", "coverage": {"overall": coverage},
           "passes": [{"name": p.name, "face_up": p.face_up, "pose_mode": "boardless", "registration": {"median_px": outline_px}}
                      for p in passes],
           "processing": {"device_estimate": {"size_mm": [281.0, 213.5, 5.8]}},
           "defects": [{"type": "scratch"}, {"type": "chip"}, {"type": "scratch"}]}
    (out / "run.json").write_text(json.dumps(rep), encoding="utf-8")
    return rep


def test_manifest_and_discovery(tmp_path, recordings):
    _device(tmp_path, "a", recordings, "long")
    (tmp_path / "not_a_device").mkdir()
    assert [p.name for p in find_devices(tmp_path)] == ["a"]
    m = load_manifest(tmp_path / "a")
    assert m.board == "none" and [r.face for r in m.passes] == ["screen", "back"]
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "manifest.yaml").write_text("passes: [{face: top, path: x}]\nboard: a5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="board must be"):
        load_manifest(bad)


def test_batch_statuses_and_summary(tmp_path, recordings):
    caps = tmp_path / "captures"
    _device(caps, "good", recordings, "long")
    _device(caps, "short", recordings, "short")                  # intake: too short -> recapture
    _device(caps, "matlost", recordings, "long", {"board": "a2"})  # result: no board poses -> warn
    (caps / "broken").mkdir()
    (caps / "broken" / "manifest.yaml").write_text("board: none\n", encoding="utf-8")   # no passes -> failed
    calls = []

    def runner(passes, m, out, gates):
        calls.append(m.name)
        return fake_runner(passes, m, out, gates)

    s = run_batch(caps, tmp_path / "out", runner=runner)
    by = {r["name"]: r for r in s["devices"]}
    assert by["good"]["status"] == "ok" and by["good"]["n_defects"] == 3 and by["good"]["defects_by_type"] == {"scratch": 2, "chip": 1}
    assert by["short"]["status"] == "recapture" and any("at least 45 s" in x for x in by["short"]["reasons"])
    assert by["matlost"]["status"] == "warn" and any("mat was not found" in x for x in by["matlost"]["reasons"])
    assert by["broken"]["status"] == "failed"
    assert sorted(calls) == ["good", "matlost"]                  # the short capture never reached the pipeline
    md = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert "| good | ok |" in md and "281.0 × 213.5 × 5.8" in md
    assert json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))["gates_sha256"]

    # a second run reuses existing results unless forced
    calls.clear()
    run_batch(caps, tmp_path / "out", runner=runner)
    assert calls == []
    run_batch(caps, tmp_path / "out", runner=runner, force=True)
    assert sorted(calls) == ["good", "matlost"]
