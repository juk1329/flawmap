"""Batch processing of captures filmed to the capture spec (docs/project_guide.md 7.0).

    flawmap batch data/captures            every device folder below that has a manifest.yaml
    flawmap batch data/captures/<device>   one device

No decision is left to a person or a model: what goes in, how it is processed and when a result is
good enough all come from the device's ``manifest.yaml`` and ``configs/batch.yaml`` (gates).

Per device:
1. **intake**: every recording is read and checked against the intake gates (length, orbit, depth).
   A failed gate ends the device as ``recapture`` with the reasons; nothing is processed.
2. **process**: ``pipeline.run`` on the recordings of the kinds listed in the config (the orbit of each
   face today), with the mat from the manifest (A3, A2 or none -> boardless).
3. **result gates** on ``run.json``: outline residual per pass, coverage, board poses when a mat was
   used. A failed gate gives ``warn`` (the result is kept); a ``FlawmapError`` gives ``recapture``
   with its code; anything else ``failed``.

Results go to ``<out>/<name>/`` (the usual run folder, viewer bundle included), and every invocation
writes ``<out>/summary.json`` and ``summary.md`` with one row per device.
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from .errors import FlawmapError
from .utils.io import read_json, read_yaml, repo_root, write_json
from .utils.logging import get_logger

log = get_logger("flawmap.batch")
BOARDS = {"a3": "configs/board_a3.yaml", "a2": "configs/board_a2.yaml"}
FACES, KINDS = ("screen", "back"), ("orbit", "close")


@dataclass
class Recording:
    face: str
    kind: str
    path: Path


@dataclass
class Manifest:
    root: Path
    name: str
    device: str
    board: str
    square_mm: float | None
    passes: list[Recording]
    gt: Path | None = None
    raw: dict = field(default_factory=dict)


def load_manifest(folder: Path) -> Manifest:
    folder = Path(folder)
    d = read_yaml(folder / "manifest.yaml") or {}
    problems = []
    board = str(d.get("board", "none")).lower()
    if board not in (*BOARDS, "none"):
        problems.append(f"board must be one of {[*BOARDS, 'none']}, not {board!r}")
    recs = []
    for k, p in enumerate(d.get("passes") or []):
        face, kind, path = p.get("face"), p.get("kind", "orbit"), p.get("path")
        if face not in FACES or kind not in KINDS or not path:
            problems.append(f"passes[{k}] needs face {FACES}, kind {KINDS} and path: {p}")
            continue
        recs.append(Recording(face, kind, folder / path))
    if not recs:
        problems.append("no passes")
    if problems:
        raise ValueError(f"{folder / 'manifest.yaml'}: " + "; ".join(problems))
    gt = folder / d["gt"] if d.get("gt") else (folder / "gt.yaml" if (folder / "gt.yaml").exists() else None)
    sq = d.get("square_mm")
    return Manifest(folder, str(d.get("name") or folder.name), str(d.get("device", "auto")), board,
                    float(sq) if sq else None, recs, gt, d)


def find_devices(root: Path) -> list[Path]:
    root = Path(root)
    if (root / "manifest.yaml").exists():
        return [root]
    return sorted(p.parent for p in root.glob("*/manifest.yaml"))


def load_gates(path: Path | None = None) -> tuple[dict, str]:
    p = Path(path) if path else repo_root() / "configs" / "batch.yaml"
    return read_yaml(p), hashlib.sha256(p.read_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------------------- gates
def intake(m: Manifest, gates: dict) -> tuple[list[dict], list[str]]:
    """Quick check of every recording against the intake gates: (per-recording facts, problems)."""
    from .service.intake import quick_check

    facts, problems = [], []
    for r in m.passes:
        g = gates["intake"].get(r.kind, {})
        tag = f"{r.face}_{r.kind}"
        if not (r.path / "odometry.csv").exists():
            problems.append(f"{tag}: no Stray Scanner recording at {r.path.name}")
            continue
        try:
            q = quick_check(r.path)
        except FlawmapError as e:
            problems.append(f"{tag}: {e.code} {e.detail}")
            continue
        facts.append({"recording": tag, **{k: q[k] for k in ("n_frames", "duration_s", "orbit", "has_depth")}})
        if q["duration_s"] < g.get("min_seconds", 0):
            problems.append(f"{tag}: {q['duration_s']:.0f} s long, the spec asks for at least {g['min_seconds']} s")
        if q["orbit"] < g.get("min_orbit_share", 0):
            problems.append(f"{tag}: went round {q['orbit'] * 100:.0f}% of the device, needs {g['min_orbit_share'] * 100:.0f}%")
        if g.get("require_depth") and not q["has_depth"]:
            problems.append(f"{tag}: no LiDAR depth (turn on depth recording in Stray Scanner)")
    used = {(r.face) for r in m.passes if r.kind in gates["process"]["passes"]}
    if not used:
        problems.append(f"no recording of the kinds {gates['process']['passes']} to process")
    return facts, problems


def result_gates(rep: dict, m: Manifest, gates: dict) -> list[str]:
    g = gates["result"]
    problems = []
    for p in rep.get("passes", []):
        px = (p.get("registration") or {}).get("median_px")
        if px is None or px > g["max_outline_px"]:
            problems.append(f"{p.get('name')}: outline residual {px} px > {g['max_outline_px']} px")
        if g.get("expect_board_poses") and m.board != "none" and p.get("pose_mode") != "board":
            problems.append(f"{p.get('name')}: the mat was not found (poses: {p.get('pose_mode')}); keep the mat in view")
        iou = (p.get("mask_iou") or {}).get("iou_p50")
        if g.get("min_mask_iou") and iou is not None and iou < g["min_mask_iou"]:
            problems.append(f"{p.get('name')}: device mask IoU {iou} < {g['min_mask_iou']} (registration does not cover the device)")
    faces = {p.get("face_up") for p in rep.get("passes", [])}
    cov = (rep.get("coverage") or {}).get("overall")
    need = g["min_coverage_both_faces"] if len(faces) >= 2 else g["min_coverage_one_face"]
    if cov is None or cov < need:
        problems.append(f"coverage {cov} < {need}")
    return problems


# ---------------------------------------------------------------------------- one device
def process_device(m: Manifest, out: Path, gates: dict, runner=None, force: bool = False) -> dict:
    """Intake, pipeline and result gates for one device; returns its summary row."""
    from .pipeline import PassInput

    runner = runner or _run_pipeline
    row = {"name": m.name, "folder": str(m.root), "device_requested": m.device, "board": m.board, "status": None,
           "reasons": [], "out": str(out)}
    t0 = time.time()
    facts, problems = intake(m, gates)
    row["intake"] = facts
    if problems:
        row.update(status="recapture", reasons=problems)
        return row
    if (out / "run.json").exists() and not force:
        log.info(f"[{m.name}] already processed ({out / 'run.json'}); --force to redo")
        rep = read_json(out / "run.json")
        row["skipped"] = True
    else:
        passes = [PassInput(r.path, r.face, r.face) for r in m.passes if r.kind in gates["process"]["passes"]]
        try:
            rep = runner(passes, m, out, gates)
        except FlawmapError as e:
            row.update(status="recapture", reasons=[f"{e.code}: {e.message} {e.hint}", e.detail])
            return row
        except Exception as e:
            row.update(status="failed", reasons=[f"{type(e).__name__}: {e}"], traceback=traceback.format_exc()[-2000:])
            return row
    problems = result_gates(rep, m, gates)
    cov = (rep.get("coverage") or {}).get("overall")
    est = (rep.get("processing") or {}).get("device_estimate") or {}
    row.update(status="warn" if problems else "ok", reasons=problems, device=rep.get("device"),
               device_name=rep.get("device_name"), size_mm=est.get("size_mm"), coverage=cov,
               n_defects=len(rep.get("defects", [])),
               defects_by_type=_count(d.get("type") for d in rep.get("defects", [])),
               outline_px={p.get("name"): (p.get("registration") or {}).get("median_px") for p in rep.get("passes", [])},
               pose_mode={p.get("name"): p.get("pose_mode") for p in rep.get("passes", [])},
               seconds=round(time.time() - t0, 1), gt=str(m.gt) if m.gt else None)
    return row


def _count(xs) -> dict:
    out: dict = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def _run_pipeline(passes, m: Manifest, out: Path, gates: dict) -> dict:
    from .pipeline import run

    poses = "boardless" if m.board == "none" else "auto"
    board_path = repo_root() / BOARDS[m.board] if m.board in BOARDS else None
    pr = gates["process"]
    return run(passes, m.device, out, keyframes=pr.get("keyframes", 140), board_square_mm=m.square_mm,
               poses=poses, board_path=board_path, meta={"batch": True, "manifest": m.raw},
               refine=pr.get("refine"), sharp_views=bool(pr.get("sharp_views", False)),
               segment=bool(pr.get("segment", False)))


# ---------------------------------------------------------------------------- many devices
def run_batch(root: Path, out_root: Path, gates_path: Path | None = None, runner=None, force: bool = False) -> dict:
    gates, gates_hash = load_gates(gates_path)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for folder in find_devices(root):
        try:
            m = load_manifest(folder)
        except Exception as e:
            rows.append({"name": folder.name, "folder": str(folder), "status": "failed", "reasons": [str(e)]})
            continue
        log.info(f"[{m.name}] {len(m.passes)} recordings, device {m.device}, mat {m.board}")
        rows.append(process_device(m, out_root / m.name, gates, runner=runner, force=force))
        log.info(f"[{m.name}] {rows[-1]['status']}" + (f": {rows[-1]['reasons']}" if rows[-1]["reasons"] else ""))
    summary = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "root": str(root), "gates": str(gates_path or "configs/batch.yaml"),
               "gates_sha256": gates_hash, "counts": _count(r["status"] for r in rows), "devices": rows}
    write_json(out_root / "summary.json", summary)
    (out_root / "summary.md").write_text(summary_md(summary), encoding="utf-8")
    return summary


def summary_md(s: dict) -> str:
    lines = [f"# flawmap batch — {s['created']}", "",
             f"입력 `{s['root']}`, 기준 `{s['gates']}` (sha256 {s['gates_sha256']}), 결과 {s['counts']}", "",
             "| 기기 | 상태 | 기종 | 크기 (mm) | 하자 | 커버리지 | 외곽 잔차 (px) | 위치 방식 | 이유 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in s["devices"]:
        size = " × ".join(f"{v:.1f}" for v in r["size_mm"]) if r.get("size_mm") else ""
        cov = f"{r['coverage'] * 100:.0f}%" if r.get("coverage") is not None else ""
        px = ", ".join(f"{k} {v:.2f}" for k, v in (r.get("outline_px") or {}).items() if v is not None)
        pm = ", ".join(f"{k} {v}" for k, v in (r.get("pose_mode") or {}).items())
        why = "<br>".join(str(x) for x in r.get("reasons", []) if x)
        lines.append(f"| {r['name']} | {r['status']} | {r.get('device_name') or r.get('device_requested', '')} | {size} | "
                     f"{r.get('n_defects', '')} | {cov} | {px} | {pm} | {why} |")
    return "\n".join(lines) + "\n"


def main_summary(summary: dict) -> int:
    """Exit code: 0 when every device is ok or warn, 1 otherwise."""
    bad = [r for r in summary["devices"] if r["status"] in ("recapture", "failed")]
    print(json.dumps(summary["counts"], ensure_ascii=False))
    for r in summary["devices"]:
        print(f"  {r['name']}: {r['status']}" + (f" - {'; '.join(str(x) for x in r['reasons'] if x)}" if r["reasons"] else ""))
    return 1 if bad else 0
