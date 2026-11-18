"""Web server: the inspection platform (request -> queue -> requester page -> shared result) and the viewer.

Someone who wants an item checked (a seller, a renter returning a device, a repair desk, ...) films
it and uploads the recording; the server processes it; the requester looks at the result and
decides who sees it: nobody, anyone with the link, or also the public gallery.

Service (``data_dir`` given, the default for ``flawmap serve``):

    GET  /                          home
    GET  /new                       request form: upload, or start from a public video or a synthetic sample
    GET  /s/<owner>                 requester page: progress, result summary, sharing, re-shoot advice
    GET  /s/<owner>/view            the viewer on the requester's result
    GET  /r/<share>                 the viewer as the read-only shared result (once shared)
    GET  /gallery                   results the requesters put on the public list
    GET  /ops                       operator view of the queue (localhost only)
    POST /api/inspections           multipart: device, title, requester, screen/back zip | sample | public
    GET  /api/inspections/<owner>   status for the requester page
    POST /api/inspections/<owner>/share     {"published": bool, "listed": bool}
    DELETE /api/inspections/<owner>
    GET  /api/reports/<share>       summary of a shared result
    GET  /api/gallery               results on the public list
    GET  /api/samples               synthetic samples and public videos the form offers
    GET  /{s|r}/<token>/bundle/...  result files, after checking the token

``/sell`` and ``/market`` (the first, used-phone-market wording) redirect to ``/new`` and ``/gallery``.

Development (always on):

    GET /?run=<name>, /viewer       viewer on runs/<name>/bundle
    GET /api/bundles                bundles under runs/
    GET /runs/<name>/bundle/...     their files

Service results live under ``data_dir`` and are reachable only through the token routes.
"""
from __future__ import annotations

import json
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ..errors import FlawmapError
from ..utils.io import read_yaml, repo_root

LOCAL = {"127.0.0.1", "::1", "localhost", "testclient"}
SAMPLES = {   # synthetic captures on the server: name -> (device, {face: folder}, title)
    "iphone12_dent": ("iphone-12", {"back": "data/synthetic/iphone12_dent_small"}, "아이폰 12 블랙 (가상 촬영, 프레임 찍힘 2곳)"),
    "iphone12_small": ("iphone-12", {"back": "data/synthetic/iphone12_back_small"}, "아이폰 12 블랙 (가상 촬영, 짧은 시험본)"),
    "iphone12_demo": ("iphone-12", {"back": "data/synthetic/iphone12_demo_back", "screen": "data/synthetic/iphone12_demo_screen"},
                      "아이폰 12 블랙 (가상 촬영, 두 면)"),
    "iphone12_wood": ("auto", {"back": "data/synthetic/iphone12_wood_small"}, "아이폰 12 블랙 (가상 촬영, 나무 바닥, 보드 없음, 기종 모름)"),
}
CATEGORIES = [   # what the platform can inspect; only plate-shaped phones for now
    {"slug": "phone", "name": "휴대폰", "ready": True},
    {"slug": "tablet", "name": "태블릿", "ready": False, "note": "준비 중"},
    {"slug": "other", "name": "그 밖의 물건", "ready": False, "note": "준비 중"},
]

_DEVICES: list[dict] | None = None


def _devices() -> list[dict]:
    global _DEVICES
    if _DEVICES is not None:
        return _DEVICES
    from ..anchor.parametric import DeviceSpec

    out = []
    for p in sorted((repo_root() / "configs" / "devices").glob("*.yaml")):
        try:
            spec = DeviceSpec.by_slug(p.stem)
        except Exception:
            continue
        out.append({"slug": p.stem, "name": spec.name})
    out.append({"slug": "auto", "name": "모름 (촬영본에서 치수를 재서 추정)"})
    _DEVICES = out
    return out


def _device_name(slug: str) -> str:
    if slug == "template":
        return "도면 없는 휴대폰 (판 모양 모델)"
    return next((d["name"] for d in _devices() if d["slug"] == slug), slug)


def _public_samples() -> list[dict]:
    """Public videos (``configs/demo_public.yaml``) that are on this server."""
    p = repo_root() / "configs" / "demo_public.yaml"
    if not p.exists():
        return []
    cfg = read_yaml(p) or {}
    root = repo_root() / cfg.get("uco3d_root", "data/datasets/uco3d")
    out = []
    for e in cfg.get("uco3d", []) or []:
        hits = list(root.glob(f"*/*/{e['name']}/rgb_video.mp4"))
        if hits and (root / "metadata.sqlite").exists():
            out.append({**e, "source": "uco3d", "video": str(hits[0]), "data_root": str(root)})
    return out


def _page(name: str) -> FileResponse:
    return FileResponse(repo_root() / "web" / "service" / name, media_type="text/html")


def _viewer_html(config: dict | None = None) -> HTMLResponse:
    html = (repo_root() / "web" / "index.html").read_text()
    inject = '<base href="/">'
    if config is not None:
        inject += "<script>window.FLAWMAP = " + json.dumps(config, ensure_ascii=False).replace("</", "<\\/") + ";</script>"
    return HTMLResponse(html.replace("<head>", "<head>\n" + inject, 1))


def _file_in(base: Path, rel: str) -> FileResponse:
    base = base.resolve()
    target = (base / rel).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(404)
    return FileResponse(target)


def create_app(runs_dir: Path, data_dir: Path | None = None, start_worker: bool = True, inline_runner=None,
               inline_public_runner=None) -> FastAPI:
    runs_dir = Path(runs_dir).resolve()
    web_dir = repo_root() / "web"
    store = worker = None
    if data_dir is not None:
        from ..service.store import Store
        from ..service.worker import Worker

        store = Store(Path(data_dir))
        worker = Worker(store, inline_runner=inline_runner, inline_public_runner=inline_public_runner)

    @asynccontextmanager
    async def lifespan(app):
        if worker is not None and start_worker:
            worker.start()
        yield
        if worker is not None:
            worker.stop()

    app = FastAPI(title="flawmap", lifespan=lifespan)
    app.state.store, app.state.worker = store, worker

    # ------------------------------------------------------------------ development viewer
    @app.get("/api/bundles")
    def bundles():
        out = []
        for d in sorted(runs_dir.glob("*/bundle/defects.json")):
            try:
                j = json.loads(d.read_text())
            except Exception:
                continue
            name = d.parent.parent.name
            out.append({"name": name, "device": j.get("device", {}).get("name"), "n_defects": len(j.get("defects", [])),
                        "coverage": j.get("coverage", {}).get("overall"), "url": f"/runs/{name}/bundle/"})
        return JSONResponse(out)

    @app.get("/viewer")
    def viewer():
        return _viewer_html()

    @app.get("/")
    def index(request: Request):
        if store is None or "run" in request.query_params or "bundle" in request.query_params:
            return _viewer_html()
        return _page("home.html")

    if store is None:
        app.mount("/runs", StaticFiles(directory=str(runs_dir)), name="runs")
        app.mount("/", StaticFiles(directory=str(web_dir)), name="web")
        return app

    from ..service.intake import sha256_file, take_folder, take_zip
    from ..service.job import STAGE_KO

    def need(d: dict | None) -> dict:
        if d is None:
            raise HTTPException(404)
        return d

    def shared(d: dict) -> bool:
        return d["status"] == "done" and bool(d.get("published") or d.get("superseded_by"))

    def bundle_dir(d: dict) -> Path:
        return Path(d["run_dir"]) / "bundle"

    def passes_public(d: dict) -> list[dict]:
        return [{"face_up": p["face_up"], "sha256": (p.get("sha256") or "")[:12], "bytes": p.get("bytes"),
                 "sample": p.get("sample"), "public": p.get("name") if p.get("kind") == "uco3d" else None,
                 "license": p.get("license"), "check": p.get("check")} for p in (d.get("passes") or [])]

    def report_public(d: dict) -> dict:
        """What the requester and anyone with the shared link may see."""
        s = d.get("summary") or {}
        return {"id": d["id"], "device": d["device"], "device_name": s.get("device_name") or _device_name(d["device"]),
                "title": d.get("title"), "requester": d.get("seller"), "status": d["status"], "created": d["created"],
                "finished": d.get("finished"), "passes": passes_public(d), "summary": s, "superseded_by": d.get("superseded_by"),
                "source": d.get("source")}

    def owner_view(d: dict) -> dict:
        out = report_public(d)
        eta = None
        if d["status"] in ("queued", "processing"):
            typical = store.typical_seconds(d.get("n_passes") or 1)
            if d["status"] == "processing":
                done = max(d.get("progress") or 0.0, 0.02)
                spent = time.time() - (d.get("started") or time.time())
                eta = max(typical * (1 - done), spent / done * (1 - done) if done > 0.15 else typical * (1 - done))
            else:
                eta = typical * (1 + (store.jobs_ahead(d["id"]) or 0))
        out.update({"stage": d.get("stage"), "stage_ko": STAGE_KO.get(d.get("stage") or "", d.get("stage")),
                    "progress": d.get("progress"), "jobs_ahead": store.jobs_ahead(d["id"]), "eta_s": eta,
                    "started": d.get("started"), "error": d.get("error"), "published": bool(d.get("published")),
                    "listed": bool(d.get("listed")), "share_url": f"/r/{d['share']}", "view_url": f"/s/{d['owner']}/view",
                    "owner_url": f"/s/{d['owner']}"})
        return out

    # ------------------------------------------------------------------ pages
    @app.get("/new")
    def new_page():
        return _page("new.html")

    @app.get("/gallery")
    def gallery_page():
        return _page("gallery.html")

    @app.get("/sell")
    def old_sell(request: Request):
        q = f"?{request.url.query}" if request.url.query else ""
        return RedirectResponse(f"/new{q}", status_code=308)

    @app.get("/market")
    def old_market():
        return RedirectResponse("/gallery", status_code=308)

    @app.get("/ops")
    def ops(request: Request):
        if request.client is None or request.client.host not in LOCAL:
            raise HTTPException(403)
        return _page("ops.html")

    @app.get("/s/{owner}")
    def owner_page(owner: str):
        need(store.by_owner(owner))
        return _page("inspection.html")

    @app.get("/s/{owner}/view")
    def owner_viewer(owner: str):
        d = need(store.by_owner(owner))
        if d["status"] != "done":
            raise HTTPException(404)
        return _viewer_html({"bundle": f"/s/{owner}/bundle/", "role": "owner", "back": f"/s/{owner}", "inspection": report_public(d),
                             "share_url": f"/r/{d['share']}", "published": bool(d.get("published")), "listed": bool(d.get("listed"))})

    @app.get("/r/{share}")
    def shared_result(share: str):
        d = store.by_share(share)
        if d is None or not shared(d):
            return HTMLResponse((repo_root() / "web" / "service" / "unavailable.html").read_text(), status_code=404)
        return _viewer_html({"bundle": f"/r/{share}/bundle/", "role": "viewer", "back": "/gallery" if d.get("listed") else "/",
                             "inspection": report_public(d)})

    @app.get("/s/{owner}/bundle/{path:path}")
    def owner_file(owner: str, path: str):
        return _file_in(bundle_dir(need(store.by_owner(owner))), path)

    @app.get("/r/{share}/bundle/{path:path}")
    def shared_file(share: str, path: str):
        d = need(store.by_share(share))
        if not shared(d):
            raise HTTPException(404)
        return _file_in(bundle_dir(d), path)

    # ------------------------------------------------------------------ API
    @app.get("/api/devices")
    def devices():
        return _devices()

    @app.get("/api/categories")
    def categories():
        return CATEGORIES

    @app.get("/api/samples")
    def samples():
        root = repo_root()
        synth = [{"kind": "synthetic", "name": k, "device": v[0], "faces": list(v[1]), "title": v[2]} for k, v in SAMPLES.items()
                 if all((root / f / "rgb.mp4").exists() for f in v[1].values())]
        public = [{"kind": "public", "name": e["name"], "faces": [e.get("face_up", "screen")], "title": e["title"],
                   "note": e.get("note", ""), "license": e.get("license", "")} for e in _public_samples()]
        return {"public": public, "synthetic": synth}

    @app.post("/api/inspections")
    def create_inspection(device: str = Form("auto"), title: str = Form(""), requester: str = Form(""), category: str = Form("phone"),
                          sample: str = Form(""), public: str = Form(""), replaces: str = Form(""),
                          screen: UploadFile | None = File(None), back: UploadFile | None = File(None)):
        if category not in {c["slug"] for c in CATEGORIES if c["ready"]}:
            raise HTTPException(400, "this kind of item is not supported yet")
        uploads = {f: u for f, u in (("screen", screen), ("back", back)) if u is not None and u.filename}
        pub = next((e for e in _public_samples() if e["name"] == public), None) if public and not uploads else None
        if not uploads and sample not in SAMPLES and pub is None:
            raise HTTPException(400, "upload a screen-up or back-up capture (zip), or pick a sample or a public video")
        if pub is None and device not in {x["slug"] for x in _devices()}:
            raise HTTPException(400, "unknown device")
        old = store.by_owner(replaces) if replaces else None
        source = "upload" if uploads else ("public" if pub else "sample")
        dev = "template" if pub else (SAMPLES[sample][0] if source == "sample" else device)
        d = store.create(device=dev, title=title.strip()[:120], seller=requester.strip()[:40], source=source)
        if old:
            store.update(d["id"], replaces=old["id"])
        updir = store.dir(d["id"]) / "upload"
        passes = []
        try:
            if uploads:
                for face, u in uploads.items():
                    passes.append({"face_up": face, **take_zip(u.file, updir / face)})
            elif pub:
                video = Path(pub["video"])
                passes.append({"kind": "uco3d", "name": pub["name"], "face_up": pub.get("face_up", "screen"),
                               "data_root": pub["data_root"], "sha256": sha256_file(video), "bytes": video.stat().st_size,
                               "license": pub.get("license", ""), "check": {"source": "uCO3D", "caption": pub.get("note", "")}})
                if not title.strip():
                    store.update(d["id"], title=pub["title"])
            else:
                _, faces, stitle = SAMPLES[sample]
                for face, folder in faces.items():
                    passes.append({"face_up": face, **take_folder(repo_root() / folder, updir / face)})
                if not title.strip():
                    store.update(d["id"], title=stitle)
        except FlawmapError as e:
            store.update(d["id"], status="failed", error=e.to_json(), finished=time.time())
            return JSONResponse({"ok": False, "error": e.to_json(), **owner_view(store.get(d["id"]))}, status_code=422)
        store.update(d["id"], status="queued", passes=passes, n_passes=len(passes))
        return {"ok": True, **owner_view(store.get(d["id"]))}

    @app.get("/api/inspections/{owner}")
    def inspection_status(owner: str):
        return owner_view(need(store.by_owner(owner)))

    @app.post("/api/inspections/{owner}/share")
    async def share_result(owner: str, request: Request):
        d = need(store.by_owner(owner))
        body = await request.json()
        if d["status"] != "done":
            raise HTTPException(409, "not processed yet")
        published = bool(body.get("published", d.get("published")))
        listed = bool(body.get("listed", d.get("listed"))) and published       # the gallery needs a working link
        store.update(d["id"], published=int(published), listed=int(listed))
        return owner_view(store.get(d["id"]))

    @app.delete("/api/inspections/{owner}")
    def delete_inspection(owner: str):
        d = need(store.by_owner(owner))
        if d["status"] == "processing":
            raise HTTPException(409, "being processed")
        if d.get("source") != "demo":
            shutil.rmtree(store.dir(d["id"]), ignore_errors=True)
        store.delete(d["id"])
        return {"ok": True}

    @app.get("/api/reports/{share}")
    def report(share: str):
        d = need(store.by_share(share))
        if not shared(d):
            raise HTTPException(404)
        out = report_public(d)
        if d.get("superseded_by"):
            new = store.get(d["superseded_by"])
            out["superseded_url"] = f"/r/{new['share']}" if new and new.get("published") else None
        return out

    @app.get("/api/gallery")
    def gallery():
        out = []
        for d in store.list(status="done", published=True, listed=True):
            if d.get("superseded_by"):
                continue
            s = d.get("summary") or {}
            out.append({**report_public(d), "report_url": f"/r/{d['share']}",
                        "thumbnail": f"/r/{d['share']}/bundle/{s['thumbnail']}" if s.get("thumbnail") else None})
        return out

    @app.get("/api/ops/jobs")
    def ops_jobs(request: Request):
        if request.client is None or request.client.host not in LOCAL:
            raise HTTPException(403)
        free = shutil.disk_usage(store.root).free
        rows = []
        for d in store.list(limit=500):
            rows.append({**owner_view(d), "attempts": d.get("attempts"), "pid": d.get("pid")})
        return {"jobs": rows, "disk_free_gb": round(free / 1e9, 1), "current": worker.current if worker else None,
                "worker_alive": bool(worker and worker.is_alive())}

    app.mount("/runs", StaticFiles(directory=str(runs_dir)), name="runs")
    app.mount("/service", StaticFiles(directory=str(web_dir / "service")), name="service")
    app.mount("/", StaticFiles(directory=str(web_dir)), name="web")
    return app


def seed_demo(store, runs_dir: Path, names: dict[str, dict]) -> list[str]:
    """Register already processed runs (``runs/<name>``) as shared, listed example results, once;
    an example whose run was processed again since (newer ``run.json``) gets its summary, title and
    device refreshed."""
    from ..service.job import summarize

    have = {Path(d["run_dir"]).resolve(): d for d in store.list(limit=1000) if d.get("run_dir")}
    added = []
    for name, meta in names.items():
        rd = (Path(runs_dir) / name).resolve()
        if not (rd / "bundle" / "defects.json").exists() or not (rd / "run.json").exists():
            continue
        if rd in have:
            d = have[rd]
            if d.get("source") == "demo" and (rd / "run.json").stat().st_mtime > float(d.get("finished") or 0):
                s = summarize(rd)
                store.update(d["id"], summary=s, finished=time.time(), title=meta.get("title", name), device=meta.get("device", "iphone-12"),
                             n_passes=len(json.loads((rd / "run.json").read_text()).get("passes", [])))
            continue
        d = store.create(device=meta.get("device", "iphone-12"), title=meta.get("title", name), seller=meta.get("requester", ""),
                         source="demo", status="done", run_dir=str(rd), published=True, listed=True)
        s = summarize(rd)
        store.update(d["id"], summary=s, finished=time.time(), started=time.time() - (s.get("seconds") or 0),
                     n_passes=len(json.loads((rd / "run.json").read_text()).get("passes", [])))
        added.append(d["id"])
    return added


DEMO_RESULTS = {   # seeded in this order; the gallery shows the newest first
    "synth_dent": {"title": "아이폰 12 블랙, 모서리 찍힘 (가상 촬영 예시)", "requester": "예시"},
    "synth_demo": {"title": "아이폰 12 블랙 (가상 촬영 예시, 두 면)", "requester": "예시"},
    "uco3d_59015-27660-21826": {"device": "template", "title": "흰색 아이폰, 액정 금 (공개 영상 uCO3D 59015-27660-21826, CC BY 4.0)",
                                "requester": "공개 데이터"},
    "uco3d_2712-37354-8763": {"device": "template", "title": "청록 테두리 휴대폰, 액정 금 (공개 영상 uCO3D 2712-37354-8763, CC BY 4.0)",
                              "requester": "공개 데이터"},
    "real_ipad": {"device": "template", "title": "아이패드 프로 13\" 은색, 두 면 (팀 실제 촬영, 나무 바닥, 보드·도면 없음)",
                  "requester": "팀 촬영"},
}


def serve(runs_dir: Path, host: str = "127.0.0.1", port: int = 8000, data_dir: Path | None = None, seed: bool = True) -> None:
    import uvicorn

    app = create_app(runs_dir, data_dir)
    if data_dir is not None and seed:
        seed_demo(app.state.store, runs_dir, DEMO_RESULTS)
    uvicorn.run(app, host=host, port=port, log_level="info")
