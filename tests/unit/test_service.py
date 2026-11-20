"""Upload service: intake safety, the inspection lifecycle and who may see what."""
import io
import json
import time
import zipfile

import numpy as np
import pytest

from flawmap.errors import FlawmapError
from flawmap.geometry.camera import Camera
from flawmap.geometry.transforms import look_at_opencv
from flawmap.service.intake import find_stray_root, safe_extract
from flawmap.service.store import Store
from flawmap.synth.render_capture import CV_TO_ARKIT_CAM, T_ARK_WORLD_BASE, write_stray


@pytest.fixture(scope="module")
def stray_zip(tmp_path_factory):
    """A tiny Stray Scanner folder (40 frames at 10 fps), zipped with its folder name inside."""
    root = tmp_path_factory.mktemp("cap") / "2026-11-16_rec"
    cam = Camera(np.array([[60.0, 0, 31.5], [0, 60.0, 23.5], [0, 0, 1]]), 64, 48)
    n = 40
    frames = [np.full((48, 64, 3), 3 * k, np.uint8) for k in range(n)]
    depths = [np.full((48, 64), 300.0, np.float32) for _ in range(n)]
    ark = [T_ARK_WORLD_BASE @ look_at_opencv(np.array([200 * np.cos(a), 200 * np.sin(a), 150.0]), np.zeros(3)) @ CV_TO_ARKIT_CAM
           for a in np.linspace(0, 6.0, n)]
    write_stray(root, cam, frames, depths, ark, fps=10.0)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in root.rglob("*"):
            z.write(f, f"{root.name}/{f.relative_to(root)}")
        z.writestr("__MACOSX/._rgb.mp4", b"junk")
    return buf.getvalue()


def fake_runner(passes, device, out, progress=None, meta=None):
    """Stands in for flawmap.pipeline.run: writes the files the service reads."""
    for st in ("decode", "poses", "detect", "bundle"):
        progress(st, 0.5)
    (out / "bundle" / "spin").mkdir(parents=True, exist_ok=True)
    (out / "bundle" / "spin" / "back_000001.jpg").write_bytes(b"\xff\xd8\xff")
    rep = {"device": device, "device_name": "iPhone 12", "passes": [{"name": p.face_up} for p in passes],
           "coverage": {"overall": 0.55, "by_part": {"back": 1.0, "screen": 0.0}},
           "defects": [{"id": 1, "type": "dent", "part": "frame", "confidence": "high", "location_ko": "오른쪽 옆면"}],
           "advice": ["화면이 위로 오게 놓고 한 번 더 찍어 주세요."], "processing": {"pose_mode": "board", "anchor": "cad", **(meta or {})},
           "timing_s": {"total": 1.0}}
    (out / "run.json").write_text(json.dumps(rep))
    (out / "bundle" / "defects.json").write_text(json.dumps(
        {"spin": {"passes": [{"name": "back", "bands": [{"elev": [35, 60], "frames": [{"img": "spin/back_000001.jpg"}]}]}]}}))
    return rep


def fake_public_runner(name, out, face_up="screen", progress=None, meta=None, data_root=None):
    """Stands in for flawmap.pipeline.run_uco3d."""
    from flawmap.pipeline import PassInput

    rep = fake_runner([PassInput(out, face_up)], "template", out, progress, meta)
    rep["processing"].update({"pose_mode": "sfm", "anchor": "template", "source": f"uco3d:{name}"})
    (out / "run.json").write_text(json.dumps(rep))
    return rep


@pytest.fixture()
def client(tmp_path):
    from fastapi.testclient import TestClient

    from flawmap.server.app import create_app

    (tmp_path / "runs").mkdir()
    app = create_app(tmp_path / "runs", tmp_path / "service", inline_runner=fake_runner, inline_public_runner=fake_public_runner)
    with TestClient(app) as c:
        c.store = app.state.store
        yield c


def _wait(c, owner, status, timeout=15.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = c.get(f"/api/inspections/{owner}").json()
        if d["status"] == status:
            return d
        time.sleep(0.2)
    raise AssertionError(f"status stayed {d['status']}")


def test_upload_process_publish_and_access(client, stray_zip):
    c = client
    r = c.post("/api/inspections", data={"device": "iphone-12", "title": "아이폰 12 블랙", "requester": "수리 접수 창구"},
               files={"back": ("cap.zip", stray_zip, "application/zip")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ok"] and d["status"] in ("queued", "processing", "done")
    assert d["passes"][0]["check"]["n_frames"] == 40 and len(d["passes"][0]["sha256"]) == 12
    owner, share = d["owner_url"].split("/")[-1], d["share_url"].split("/")[-1]
    d = _wait(c, owner, "done")
    assert d["summary"]["n_defects"] == 1 and d["summary"]["processing"]["pose_mode"] == "board"
    assert d["requester"] == "수리 접수 창구"
    # the requester sees the result; nobody else until it is shared
    assert c.get(f"/s/{owner}/view").status_code == 200
    assert c.get(f"/s/{owner}/bundle/run.json").status_code == 404           # only bundle files
    assert c.get(f"/s/{owner}/bundle/spin/back_000001.jpg").status_code == 200
    assert c.get(f"/api/reports/{share}").status_code == 404
    assert c.get(f"/r/{share}").status_code == 404
    assert c.get(f"/r/{share}/bundle/defects.json").status_code == 404
    assert c.get("/api/gallery").json() == []
    d = c.post(f"/api/inspections/{owner}/share", json={"published": True}).json()
    assert d["published"] and not d["listed"]
    page = c.get(f"/r/{share}")
    assert page.status_code == 200 and "window.FLAWMAP" in page.text and '"role": "viewer"' in page.text
    assert owner not in page.text                                            # the shared page never carries the owner token
    assert c.get(f"/r/{share}/bundle/defects.json").status_code == 200
    assert c.get("/api/gallery").json() == []                               # a link share is not on the public list
    d = c.post(f"/api/inspections/{owner}/share", json={"listed": True}).json()
    assert d["listed"]
    ls = c.get("/api/gallery").json()
    assert len(ls) == 1 and ls[0]["thumbnail"].endswith("spin/back_000001.jpg") and "owner_url" not in ls[0]
    d = c.post(f"/api/inspections/{owner}/share", json={"published": False, "listed": True}).json()
    assert not d["published"] and not d["listed"]                            # no list entry without a working link
    assert c.get(f"/r/{share}").status_code == 404
    assert c.get("/sell", follow_redirects=False).headers["location"] == "/new"
    assert c.get("/market", follow_redirects=False).headers["location"] == "/gallery"
    assert c.get("/api/reports/not-a-token").status_code == 404
    assert c.get("/s/not-a-token").status_code == 404
    # delete removes the files and the record
    assert c.delete(f"/api/inspections/{owner}").json()["ok"]
    assert c.get(f"/api/inspections/{owner}").status_code == 404


def test_reshoot_supersedes_and_keeps_listing(client, stray_zip):
    c = client
    d1 = c.post("/api/inspections", data={"device": "iphone-12", "title": "첫 촬영"}, files={"back": ("a.zip", stray_zip)}).json()
    o1 = d1["owner_url"].split("/")[-1]
    _wait(c, o1, "done")
    c.post(f"/api/inspections/{o1}/share", json={"published": True, "listed": True})
    d2 = c.post("/api/inspections", data={"device": "iphone-12", "replaces": o1}, files={"back": ("b.zip", stray_zip)}).json()
    o2 = d2["owner_url"].split("/")[-1]
    d2 = _wait(c, o2, "done")
    assert d2["published"] and d2["title"] == "첫 촬영"
    old = c.get(f"/api/inspections/{o1}").json()
    assert old["superseded_by"] == d2["id"] and not old["published"]
    assert [x["id"] for x in c.get("/api/gallery").json()] == [d2["id"]]
    rep = c.get(f"/api/reports/{d1['share_url'].split('/')[-1]}").json()     # the old link still opens, marked as replaced
    assert rep["superseded_url"] == d2["share_url"]


def test_public_video_request(client, tmp_path, monkeypatch):
    import flawmap.server.app as appmod

    video = tmp_path / "uco3d" / "x" / "y" / "1-2-3" / "rgb_video.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"not really a video")
    monkeypatch.setattr(appmod, "_public_samples", lambda: [{"name": "1-2-3", "face_up": "screen", "title": "공개 영상 시험",
                                                              "license": "CC BY 4.0", "video": str(video), "data_root": str(tmp_path / "uco3d")}])
    c = client
    s = c.get("/api/samples").json()
    assert [x["name"] for x in s["public"]] == ["1-2-3"]
    d = c.post("/api/inspections", data={"public": "1-2-3"}).json()
    assert d["ok"] and d["title"] == "공개 영상 시험" and d["passes"][0]["public"] == "1-2-3"
    d = _wait(c, d["owner_url"].split("/")[-1], "done")
    assert d["summary"]["processing"]["anchor"] == "template" and d["source"] == "public"
    assert c.post("/api/inspections", data={"public": "not-on-this-server"}).status_code == 400
    assert c.post("/api/inspections", data={"category": "tablet", "public": "1-2-3"}).status_code == 400


def test_rejects_bad_uploads(client, tmp_path):
    c = client
    r = c.post("/api/inspections", data={"device": "iphone-12"}, files={"back": ("x.zip", b"not a zip")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "UPLOAD_INVALID"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("photos/IMG_0001.jpg", b"x")
    r = c.post("/api/inspections", data={"device": "iphone-12"}, files={"back": ("x.zip", buf.getvalue())})
    assert r.status_code == 422 and r.json()["error"]["code"] == "NOT_STRAY"
    assert c.post("/api/inspections", data={"device": "no-such-phone"}, files={"back": ("x.zip", b"x")}).status_code == 400
    assert c.post("/api/inspections", data={"device": "iphone-12"}).status_code == 400


def _zip_with(member: zipfile.ZipInfo, data=b"x") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(member, data)
    return buf.getvalue()


@pytest.mark.parametrize("name", ["../evil.txt", "/abs/evil.txt", "a/../../evil.txt", "C:/evil.txt"])
def test_safe_extract_refuses_escaping_paths(tmp_path, name):
    zp = tmp_path / "x.zip"
    zp.write_bytes(_zip_with(zipfile.ZipInfo(name)))
    with pytest.raises(FlawmapError) as e:
        safe_extract(zp, tmp_path / "out")
    assert e.value.code == "UPLOAD_INVALID"
    assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_refuses_symlinks_and_bombs(tmp_path):
    link = zipfile.ZipInfo("rec/rgb.mp4")
    link.external_attr = (0o120777 << 16)
    zp = tmp_path / "l.zip"
    zp.write_bytes(_zip_with(link, b"/etc/passwd"))
    with pytest.raises(FlawmapError):
        safe_extract(zp, tmp_path / "out")
    zp.write_bytes(_zip_with(zipfile.ZipInfo("rec/big.bin"), b"\0" * 5000))
    with pytest.raises(FlawmapError):
        safe_extract(zp, tmp_path / "out2", max_bytes=1000)


def test_find_stray_root_nested(tmp_path):
    rec = tmp_path / "a" / "b"
    rec.mkdir(parents=True)
    for f in ("rgb.mp4", "odometry.csv", "camera_matrix.csv"):
        (rec / f).write_text("x")
    assert find_stray_root(tmp_path) == rec
    with pytest.raises(FlawmapError):
        find_stray_root(tmp_path / "a" / "b" / "..", max_depth=0)


def test_recover_requeues_once_then_fails(tmp_path):
    s = Store(tmp_path)
    d = s.create(device="iphone-12", status="queued")
    assert s.claim_next()["id"] == d["id"]
    s.update(d["id"], pid=999999)
    assert s.recover(lambda pid: False) == [d["id"]] and s.get(d["id"])["status"] == "queued"
    s.claim_next()
    s.recover(lambda pid: False)
    assert s.get(d["id"])["status"] == "failed"


def test_compare_ratings_are_appended_for_known_recordings(client, tmp_path):
    rec = tmp_path / "runs" / "compare" / "abc123"
    rec.mkdir(parents=True)
    (rec / "index.json").write_text('{"rec": "abc123"}')
    body = {"rec": "abc123", "blind": True, "ratings": [{"panel": "A", "candidate": "c2", "realism": 4, "stable": 5, "defects": 3, "note": ""}]}
    r = client.post("/api/compare/ratings", json=body)
    assert r.status_code == 200 and r.json()["n"] == 1
    assert client.post("/api/compare/ratings", json=body).json()["n"] == 2
    import json as _json
    rows = _json.loads((rec / "ratings.json").read_text())
    assert rows[1]["ratings"][0]["candidate"] == "c2" and "received" in rows[1]
    assert client.post("/api/compare/ratings", json={**body, "rec": "../x"}).status_code == 400
    assert client.post("/api/compare/ratings", json={**body, "rec": "nope"}).status_code == 404


def test_realistic_view_is_shared_only_when_allowed(client, stray_zip):
    """The realistic view's files (bundle/real/, their reflections show the room) reach the shared link
    only with share_env; the requester always gets them."""

    def runner_with_real(passes, device, out, progress=None, meta=None):
        rep = fake_runner(passes, device, out, progress, meta)
        (out / "bundle" / "real" / "back").mkdir(parents=True)
        (out / "bundle" / "real" / "back" / "manifest.json").write_text("{}")
        return rep

    c = client
    c.app.state.worker.inline_runner = runner_with_real
    r = c.post("/api/inspections", data={"device": "iphone-12"}, files={"back": ("cap.zip", stray_zip, "application/zip")})
    d = r.json()
    owner, share = d["owner_url"].split("/")[-1], d["share_url"].split("/")[-1]
    d = _wait(c, owner, "done")
    assert d["share_env"] is False
    assert c.get(f"/s/{owner}/bundle/real/back/manifest.json").status_code == 200
    assert '"realistic": true' in c.get(f"/s/{owner}/view").text
    d = c.post(f"/api/inspections/{owner}/share", json={"published": True}).json()
    assert d["published"] and not d["share_env"]
    assert c.get(f"/r/{share}/bundle/defects.json").status_code == 200
    assert c.get(f"/r/{share}/bundle/real/back/manifest.json").status_code == 404
    assert '"realistic": false' in c.get(f"/r/{share}").text
    d = c.post(f"/api/inspections/{owner}/share", json={"share_env": True}).json()
    assert d["share_env"] and d["published"]
    assert c.get(f"/r/{share}/bundle/real/back/manifest.json").status_code == 200
    assert '"realistic": true' in c.get(f"/r/{share}").text


def test_realistic_file_check_resolves_the_path(tmp_path):
    from flawmap.server.app import _realistic_file

    (tmp_path / "real" / "back").mkdir(parents=True)
    assert _realistic_file(tmp_path, "real/back/manifest.json")
    assert _realistic_file(tmp_path, "evidence/../real/back/views/000.webp")
    assert not _realistic_file(tmp_path, "evidence/real/a.jpg") and not _realistic_file(tmp_path, "realx/a")


def test_service_split_holds_validation_blocks_only():
    from flawmap.recon.service import service_split

    s = service_split(np.arange(0, 20.0, 1 / 30))
    assert not s.test and len(s.val) > 0 and len(s.guard) > 0
    assert not (set(s.val) & set(s.train)) and len(s.train) + len(s.val) + len(s.guard) == 600


def test_seeded_example_replaces_the_older_one(tmp_path):
    """A gallery example built again (here with the realistic view) takes the older one's place:
    the old one leaves the list and its link says where the new one is."""
    from flawmap.server.app import seed_demo

    runs = tmp_path / "runs"
    for name in ("ex", "ex_r5"):
        (runs / name / "bundle").mkdir(parents=True)
        (runs / name / "run.json").write_text(json.dumps({"passes": [{"name": "back"}], "defects": [], "coverage": {}}))
        (runs / name / "bundle" / "defects.json").write_text(json.dumps({"defects": []}))
    store = Store(tmp_path / "service")
    (old_id,) = seed_demo(store, runs, {"ex": {"title": "예시"}})
    (new_id,) = seed_demo(store, runs, {"ex_r5": {"title": "예시", "share_env": True, "replaces": "ex"}})
    old, new = store.get(old_id), store.get(new_id)
    assert old["superseded_by"] == new_id and not old["listed"] and old["published"]
    assert new["listed"] and new["share_env"] and not new["superseded_by"]
    assert seed_demo(store, runs, {"ex_r5": {"title": "예시", "share_env": True, "replaces": "ex"}}) == []   # once
