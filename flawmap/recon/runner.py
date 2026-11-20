"""``flawmap compare``: build the capture context once, fit and judge each candidate, and write what
the comparison page (``web/compare.html``) reads:

    runs/compare/<rec>/index.json          context, test-frame cameras, candidates and their scores
    runs/compare/<rec>/<cand>/web/          the candidate's files and manifest.json
    runs/compare/<rec>/<cand>/renders/      held-out pairs (real | render) and stress views
    runs/compare/<rec>/<cand>/turntable.mp4
    runs/compare/<rec>/<cand>/metrics.json
    runs/compare/leaderboard.json           one row per (recording, candidate, run time)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..utils.logging import get_logger
from . import registry
from .context import CaptureContext
from .metrics import heldout_scores, stress_scores, train_colour_range
from .stress import stress_views

log = get_logger("flawmap.recon")


def _summary(m: dict) -> dict:
    h, s = m.get("heldout", {}), m.get("stress", {})
    out = {k: h.get(k) for k in ("psnr", "psnr_g", "ssim_g", "lpips_g", "hf_ncc", "coverage")}
    for g in ("turn30", "graze", "under", "zoom"):
        for k in ("float", "hole", "blowup"):
            if k in s.get(g, {}):
                out[f"{g}_{k}"] = s[g][k]
    if "flicker" in s:
        out["flicker"] = s["flicker"]["mean"]
        out["flicker_p95"] = s["flicker"]["p95"]
        out["turn_jump"] = s["flicker"].get("jump")
        out["turn_jump_p99"] = s["flicker"].get("jump_p99")
    out.update({k: m.get("fit", {}).get(k) for k in ("seconds", "artifact_mb")})
    out["web_mb"] = m.get("manifest", {}).get("web_mb")
    return out


def run_compare(base_run: str | Path, candidates: list[str], out_root: str | Path = "runs/compare", refit: bool = False,
                stress: bool = True, round_name: str = "", turn_only: bool = False) -> dict:
    """``turn_only``: render the turntable again and score only it (flicker and jumps), keeping the
    candidate's other scores and files; for scoring earlier candidates by a new turntable measure."""
    ctx = CaptureContext.from_run(base_run, out_root)
    ctx.ensure_frames()
    idx_path = ctx.out / "index.json"
    index = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    index.update(ctx.to_json())
    index["test_views"] = [{**ctx.view(i).to_json(), "image": f"ctx/test/{i:05d}.png"} for i in ctx.test_frames]
    cands = {c["id"]: c for c in index.get("candidates", [])}
    crange = train_colour_range(ctx, ctx.train_frames[::10])
    views = stress_views(ctx.model, ctx.face, ctx.defects.get("defects", [])) if (stress or turn_only) else []
    if turn_only:
        views = [v for v in views if v.name.split()[0] == "turn30"]
    board_p = Path(out_root) / "leaderboard.json"
    for cid in candidates:
        cls = registry.get(cid)
        cand = cls(ctx)
        t0 = time.time()
        old_m = cand.out / "metrics.json"
        if turn_only:
            if not (cand.fitted() and old_m.exists()):
                log.info(f"[{ctx.rec}] {cid}: no scored fit to add the turntable to; skipped")
                continue
            prev = json.loads(old_m.read_text())
            cand.load()
            st = dict(prev.get("stress", {}))
            st.update(stress_scores(ctx, cand.render, views, crange, video_path=cand.out / "turntable.mp4"))
            metrics = {**prev, "stress": st, "turn_scored": time.strftime("%Y-%m-%d %H:%M")}
            (cand.out / "metrics.json").write_text(json.dumps(metrics, default=float))
            index = _record(ctx, cand, cid, metrics, index, idx_path, board_p, round_name or prev.get("round", ""))
            continue
        fit = cand.run_fit() if (refit or not cand.fitted()) else cand.fit_stats()
        cand.load()
        log.info(f"[{ctx.rec}] {cid}: fitted ({fit.get('seconds')} s); scoring")
        held = heldout_scores(ctx, cand.render, save_dir=cand.out / "renders" / "heldout")
        if stress:
            st = stress_scores(ctx, cand.render, views, crange, save_dir=cand.out / "renders" / "stress",
                               video_path=cand.out / "turntable.mp4")
        else:
            # held-out frames only: keep the stress scores of the last full scoring of this fit
            st = json.loads(old_m.read_text()).get("stress", {}) if (old_m.exists() and not refit) else {}
        man = cand.export_web(cand.out / "web")
        metrics = {"candidate": cid, "rec": ctx.rec, "round": round_name, "fit": fit, "heldout": held, "stress": st, "manifest": man,
                   "scored_s": round(time.time() - t0, 1)}
        (cand.out / "metrics.json").write_text(json.dumps(metrics, default=float))
        index = _record(ctx, cand, cid, metrics, index, idx_path, board_p, round_name)
    _update_recs(Path(out_root))
    return index


def _record(ctx, cand, cid: str, metrics: dict, index: dict, idx_path: Path, board_p: Path, round_name: str) -> dict:
    """The candidate's entry in the recording's index.json and a leaderboard row."""
    summ = _summary(metrics)
    entry = {"id": cid, "label": cand.label_ko, "family": cand.family, "renderer": cand.renderer, "research": cand.research,
             "manifest": f"{cid}/web/manifest.json", "turntable": f"{cid}/turntable.mp4" if (cand.out / "turntable.mp4").exists() else None,
             "summary": summ}
    log.info(f"[{ctx.rec}] {cid}: " + ", ".join(f"{k} {v}" for k, v in summ.items() if v is not None))
    # other comparisons may run at the same time: merge into what is on disk now, write atomically
    cur = json.loads(idx_path.read_text()) if idx_path.exists() else dict(index)
    cur.update({k: v for k, v in index.items() if k != "candidates"})
    merged = {c["id"]: c for c in cur.get("candidates", [])}
    merged[cid] = entry
    cur["candidates"] = sorted(merged.values(), key=lambda c: c["id"])
    _atomic(idx_path, json.dumps(cur, default=float))
    board = json.loads(board_p.read_text()) if board_p.exists() else []
    board.append({"rec": ctx.rec, "candidate": cid, "round": round_name, "time": time.strftime("%Y-%m-%d %H:%M"), **summ})
    _atomic(board_p, json.dumps(board, default=float, indent=0))
    return cur


def _atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + f".{int(time.time() * 1e6)}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _update_recs(root: Path) -> None:
    """``runs/compare/recs.json``: the recordings with a comparison, for the page's selector."""
    rows = []
    for p in sorted(root.glob("*/index.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        rows.append({"rec": d["rec"], "face": d.get("face"), "n": len(d.get("candidates", [])), "updated": int(p.stat().st_mtime)})
    _atomic(root / "recs.json", json.dumps(sorted(rows, key=lambda r: -r["updated"])))
