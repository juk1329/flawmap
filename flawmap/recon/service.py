"""The "실물처럼 보기" (realistic view) of an inspection result: C7 of the candidate comparison,
round 5, which the user picked for the service on 2026-11-19 after the round-4 blind ratings and
the round-5 results.

For each pass of a finished run (one recording each, its own room: recordings are never fused),
C7 is fitted on that recording alone: C1 (the inspection texture with the reflection map measured
from the same recording) plus the real frames' residuals against C1, blended near the directions
they were taken from (``candidates.c7_hybrid``; view choice and blend of round 5 in
``candidates.c2_ulr``). Its browser files go to ``bundle/real/<face>/`` with ``bundle/real/index.json``
listing the faces, and ``bundle/defects.json`` gets ``bundle_version: 2`` and ``realistic``, which the
viewer reads (``web/viewer.js``: the C7 shader of the comparison page, ``web/recon/mesh_hybrid.js``).

The inspection itself is not touched: defects, coverage and the inspection texture come from the
run as it was; the realistic view is a way of looking at the device. Defects are judged on the
inspection texture, so the viewer switches back to it when a defect is picked.

The run is not made with ``--holdout blocks`` (holding a third of the frames out would weaken the
inspection), so the validation frames C7 chooses its settings on are taken here: one 1-second
block in six with guard bands, as in ``eval.heldout``, and no test frames. They are kept out of
C7's own fitting; the inspection texture C1 starts from did see them, the same for every setting
compared. Nothing is scored on them. A recording too short for a validation block (under about
6 s) gets the middle settings (``c2_ulr.FADE_NO_VAL``, ``BETA_NO_VAL``).

Recordings: Stray Scanner captures (ARKit poses as the run corrected them; LiDAR keeps a stand or a
hand out) and public uCO3D phone videos (SfM cameras scaled as the run scaled them; the dataset's
object masks keep the hand out; ``context.Uco3dContext``).

The files show the room the recording was made in (the reflection map and the frames' residuals),
so the server gives them to people with the shared link only when the requester allows it
(``share_env``); the requester always sees them.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from ..eval.heldout import Split, block_split
from ..utils.logging import get_logger

log = get_logger("flawmap.realistic")

METHOD = "c7-r5"
REAL_DIR = "real"


def service_split(timestamps) -> Split:
    """Validation blocks only (one 1-second block in six, guard bands), no test frames."""
    return block_split(timestamps, test_at=-1, val_at=5)


def _usable_pass(run_dir: Path, p: dict) -> bool:
    """A pass the realistic view can be built for: a Stray Scanner recording (ARKit poses corrected by
    the run), or a public uCO3D video (SfM cameras, scaled by the run)."""
    root = Path(p.get("root") or "")
    pdir = run_dir / "passes" / str(p.get("name"))
    if not (pdir / "registration.json").exists():
        return False
    if str(p.get("source", "")).startswith("uco3d:"):
        return (root / "rgb_video.mp4").exists() and "scale_mm_per_sfm_unit" in (p.get("template") or {})
    return (root / "rgb.mp4").exists() and (pdir / "poses.json").exists()


def build_realistic(run_dir: str | Path, progress=None, keep_work: bool = False) -> dict:
    """Fit C7 for every pass of ``run_dir`` and add its files to the bundle; returns the index
    written to ``bundle/real/index.json`` (``faces``: one entry per face built). A pass that fails
    is logged and left out; the inspection result stays as it was. The fitting's own folder
    (``run_dir/realistic``: decoded frames, C1 and C7 artifacts, about 130 MB a recording) is removed
    afterwards unless ``keep_work``; the bundle's files are all the viewer needs."""
    run_dir = Path(run_dir)
    rep = json.loads((run_dir / "run.json").read_text())
    bundle = run_dir / "bundle"
    real = bundle / REAL_DIR
    work = run_dir / "realistic"
    passes = [(k, p) for k, p in enumerate(rep.get("passes", [])) if _usable_pass(run_dir, p)]
    faces, failed = [], []
    t0 = time.time()

    def step(k: int, frac: float) -> None:
        if progress is not None and passes:
            progress("realistic", min(1.0, (k + frac) / len(passes)))

    if passes:
        from .candidates.c7_hybrid import C7Hybrid
        from .context import context_for_pass
    for n, (k, p) in enumerate(passes):
        face = p["face_up"]
        t1 = time.time()
        try:
            step(n, 0.0)
            ctx = context_for_pass(run_dir, k, work)          # validation blocks: service_split
            ctx.ensure_frames()
            step(n, 0.15)
            cand = C7Hybrid(ctx)
            fit = cand.run_fit()
            step(n, 0.9)
            web = real / face
            if web.exists():
                shutil.rmtree(web)
            man = cand.export_web(web)
            faces.append({"face": face, "pass": p["name"], "recording": ctx.rec, "manifest": f"{REAL_DIR}/{face}/manifest.json",
                          "renderer": man["renderer"], "web_mb": man.get("web_mb"), "seconds": round(time.time() - t1, 1),
                          "choice": {k2: fit.get(k2) for k2 in ("fade_deg", "beta", "min_view_sep_deg", "turn_jump_every_view", "views")},
                          "frames": {"train": len(ctx.train_frames), "val": len(ctx.val_frames)}})
            log.info(f"realistic view of the {face} ({ctx.rec}): {time.time() - t1:.0f} s, {man.get('web_mb')} MB")
        except Exception as e:      # the inspection result stands without it
            log.warning(f"realistic view of the {face} failed: {type(e).__name__}: {e}")
            failed.append({"face": face, "pass": p.get("name"), "error": f"{type(e).__name__}: {e}"[:300]})
        finally:
            for c in work.glob("*/ctx"):          # the decoded frames go in any case
                shutil.rmtree(c, ignore_errors=True)
        step(n, 1.0)
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    index = {"version": 1, "method": METHOD, "faces": faces, "failed": failed, "seconds": round(time.time() - t0, 1),
             "made": time.strftime("%Y-%m-%d %H:%M")}
    if faces:
        real.mkdir(parents=True, exist_ok=True)
        (real / "index.json").write_text(json.dumps(index, indent=1))
        dpath = bundle / "defects.json"
        data = json.loads(dpath.read_text())
        data["bundle_version"] = 2
        data["realistic"] = {"index": f"{REAL_DIR}/index.json", "method": METHOD, "faces": [f["face"] for f in faces]}
        dpath.write_text(json.dumps(data))
    rep["realistic"] = {k2: index[k2] for k2 in ("method", "seconds", "made")} | {
        "faces": [{k2: f[k2] for k2 in ("face", "web_mb", "seconds", "choice")} for f in faces], "failed": failed}
    (run_dir / "run.json").write_text(json.dumps(rep, indent=1))
    return index
