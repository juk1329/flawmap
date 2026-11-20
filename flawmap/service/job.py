"""Process one inspection: ``python -m flawmap.service.job <data_dir> <inspection_id>``.

Started by the worker in its own process (under ``scripts/with_memory_guard.sh``), so a crash or an
out-of-memory stop cannot take the web server down. Progress goes into the inspection row, the
outcome too: ``done`` with a summary, or ``failed`` with an error code the seller page explains.
"""
from __future__ import annotations

import subprocess
import sys
import time
import traceback
from pathlib import Path

from ..errors import FlawmapError
from ..utils.io import repo_root
from .store import Store

STAGE_KO = {
    "start": "처리 준비", "decode": "영상 읽기", "poses": "카메라 위치 찾기", "register": "기기 모델 맞추기",
    "samples": "표면 색 모으기", "detect": "하자 찾기", "evidence": "근거 장면 고르기", "bundle": "결과 만들기",
    "realistic": "실물처럼 보기 만들기",
}


def code_version() -> str:
    try:
        r = subprocess.run(["git", "-C", str(repo_root()), "describe", "--always", "--dirty"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def summarize(run_dir: Path) -> dict:
    """What the seller page, the buyer report and the listing badge show without opening the
    bundle (read from ``run.json`` and the bundle's spin frames)."""
    from ..utils.io import read_json

    run_dir = Path(run_dir)
    rep = read_json(run_dir / "run.json")
    by_type: dict[str, int] = {}
    for d in rep.get("defects", []):
        by_type[d["type"]] = by_type.get(d["type"], 0) + 1
    cov = rep.get("coverage", {})
    thumb = None
    try:
        spin = read_json(run_dir / "bundle" / "defects.json").get("spin", {})
        bands = [b for p in spin.get("passes", []) for b in p.get("bands", []) if b.get("frames")]
        if bands:
            mid = sorted(bands, key=lambda b: abs(sum(b["elev"]) / 2 - 45))[0]["frames"]
            thumb = mid[len(mid) // 3]["img"]
    except Exception:
        pass
    real = rep.get("realistic") or {}
    return {"n_defects": len(rep.get("defects", [])), "by_type": by_type, "coverage": cov.get("overall"),
            "realistic": [f["face"] for f in real.get("faces", [])],
            "coverage_by_part": cov.get("by_part"), "coverage_by_chart": cov.get("by_chart"), "advice": rep.get("advice", []),
            "processing": rep.get("processing", {}), "device_name": rep.get("device_name"),
            "defects": [{k: d.get(k) for k in ("id", "type", "part", "confidence", "location_ko")} for d in rep.get("defects", [])],
            "seconds": rep.get("timing_s", {}).get("total"), "thumbnail": thumb}


def finish_superseding(store: Store, d: dict) -> None:
    """A finished re-shoot replaces the seller's earlier inspection of the same listing."""
    if d.get("replaces"):
        old = store.get(d["replaces"])
        if old and old["id"] != d["id"]:
            store.update(old["id"], superseded_by=d["id"], published=0, listed=0)
            store.update(d["id"], title=d.get("title") or old.get("title"), price=d.get("price") or old.get("price"),
                         published=old.get("published") or d.get("published") or 0, listed=old.get("listed") or d.get("listed") or 0)


def process(store: Store, iid: str, runner=None, public_runner=None, realistic=None) -> dict:
    """Run the pipeline for inspection ``iid``. ``runner(passes, device, out, progress, meta)`` for
    uploaded Stray Scanner captures defaults to ``flawmap.pipeline.run``; ``public_runner(name, out,
    face_up, progress, meta)`` for public uCO3D videos to ``flawmap.pipeline.run_uco3d`` (tests pass fakes).
    ``realistic(run_dir, progress)`` then adds the realistic view to a Stray capture's result
    (``flawmap.recon.service.build_realistic``; False: none). The result is done without it if it fails."""
    from ..pipeline import PassInput

    d = store.get(iid)
    if d is None:
        raise SystemExit(f"no inspection {iid}")
    public = bool(d["passes"]) and d["passes"][0].get("kind") == "uco3d"
    if runner is None:
        from ..pipeline import run as runner
    if public_runner is None:
        from ..pipeline import run_uco3d as public_runner
    passes = [] if public else [PassInput(Path(p["root"]), p["face_up"]) for p in d["passes"]]
    last = {"t": 0.0, "stage": None}

    def progress(stage: str, frac: float, note: str = "") -> None:
        now = time.time()
        if stage != last["stage"] or now - last["t"] > 1.0 or frac >= 1.0:
            store.update(iid, stage=stage, progress=round(float(frac), 3))
            last.update(t=now, stage=stage)

    try:
        meta = {"inspection": iid, "version": code_version(), "input_sha256": {p["face_up"]: p.get("sha256") for p in d["passes"]}}
        if public:
            p0 = d["passes"][0]
            kw = {"data_root": p0["data_root"]} if p0.get("data_root") else {}
            rep = public_runner(p0["name"], Path(d["run_dir"]), face_up=p0["face_up"], progress=progress, meta=meta, **kw)
        else:
            rep = runner(passes, d["device"], Path(d["run_dir"]), progress=progress, meta=meta)
            if realistic is None:
                from ..recon.service import build_realistic as realistic
            if realistic:
                try:
                    realistic(Path(d["run_dir"]), progress=progress)
                except Exception as e:      # the inspection stands without the realistic view
                    print(f"realistic view failed: {type(e).__name__}: {e}", file=sys.stderr)
        summary = summarize(Path(d["run_dir"]))
        # carry the earlier inspection's title and sharing over before the result shows as done,
        # so nobody sees a finished re-shoot without them
        finish_superseding(store, d)
        store.update(iid, status="done", stage="bundle", progress=1.0, finished=time.time(), summary=summary, pid=None)
        return rep
    except FlawmapError as e:
        store.update(iid, status="failed", finished=time.time(), error=e.to_json(), pid=None)
        raise
    except Exception as e:
        store.update(iid, status="failed", finished=time.time(), pid=None,
                     error={**FlawmapError("INTERNAL").to_json(), "detail": f"{type(e).__name__}: {e}"[:500]})
        raise


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: python -m flawmap.service.job <data_dir> <inspection_id>", file=sys.stderr)
        return 2
    store = Store(Path(argv[0]))
    try:
        process(store, argv[1])
    except FlawmapError as e:
        print(f"failed: {e}", file=sys.stderr)
        return 1
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
