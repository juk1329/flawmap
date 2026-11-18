"""Compare a processed synthetic capture with its ground truth (``gt.json``).

    python -m flawmap.synth.evaluate runs/<name> --pass back=data/synthetic/<capture> [--pass screen=...]

* **Camera poses in the device frame** (what the maps and evidence use): for every keyframe the
  rotation and camera-position error of ``T_cad_cam``, and the mean image displacement (px) of the
  model's vertices between the estimated and the true pose.
* **Defects:** every injected decal or dent on a filmed chart is matched to the nearest found
  defect of the same chart within ``tol_mm`` (the frame chart wraps around the perimeter).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ..anchor.parametric import DeviceSpec, build_device_model
from ..geometry.camera import Camera
from ..geometry.transforms import inv_T, pose_difference
from ..utils.io import read_json, write_json


def pose_errors(run_dir: Path, capture: Path, pass_name: str, model=None) -> dict:
    gt = read_json(capture / "gt.json")
    model = model or build_device_model(DeviceSpec.by_slug(gt["device"]))
    poses = read_json(run_dir / "passes" / pass_name / "poses.json")
    reg = read_json(run_dir / "passes" / pass_name / "registration.json")
    T_wcad = np.asarray(reg["T_world_cad"])
    T_wcad_gt = np.asarray(gt["T_world_cad"])
    from ..io.stray import read_stray

    cap = read_stray(capture)
    V = model.mesh.vertices[:: max(1, len(model.mesh.vertices) // 2000)]
    rot, pos, px = [], [], []
    for k, p in poses["poses"].items():
        i = int(k)
        T_est = inv_T(T_wcad) @ np.asarray(p["T_world_cam"])
        T_gt = inv_T(T_wcad_gt) @ np.asarray(gt["frames"][i]["T_world_cam"])
        r, t = pose_difference(T_est, T_gt)
        rot.append(r)
        pos.append(t)
        cam = Camera(cap.K[i], cap.width, cap.height)
        uv = []
        for T in (T_est, T_gt):
            Tc = inv_T(T)
            pc = V @ Tc[:3, :3].T + Tc[:3, 3]
            uv.append(np.stack([cam.fx * pc[:, 0] / pc[:, 2] + cam.cx, cam.fy * pc[:, 1] / pc[:, 2] + cam.cy], 1))
        px.append(float(np.linalg.norm(uv[0] - uv[1], axis=1).mean()))
    rot, pos, px = map(np.asarray, (rot, pos, px))
    return {"n": len(rot), "rot_deg_median": float(np.median(rot)), "rot_deg_p90": float(np.percentile(rot, 90)),
            "pos_mm_median": float(np.median(pos)), "pos_mm_p90": float(np.percentile(pos, 90)),
            "reproj_px_median": float(np.median(px)), "reproj_px_p90": float(np.percentile(px, 90)), "reproj_px_max": float(px.max()),
            "pose_mode": poses.get("arkit_residual", {}).get("mode", "board")}


def drawn_centres(capture: Path) -> list[tuple[float, float] | None]:
    """Centre (chart mm) of every decal of ``gt.json`` as the renderer actually drew it: the same
    painter with the same seed. A crack or scratch is a random walk (with branches) from its nominal
    start, so its own centre can lie several millimetres from the nominal ``u, v`` (the back crack of
    the general-floor captures: 4.7 mm, which put a correct detection at 5.01 mm, 2026-11-17)."""
    from .render_capture import Decal, _chart_canvas, _paint_decal

    gt = read_json(capture / "gt.json")
    model = build_device_model(DeviceSpec.by_slug(gt["device"]))
    res = float(gt.get("config", {}).get("texture_mm_per_px", 0.05))
    out = []
    for k, d in enumerate(gt["decals"]):
        dec = Decal(**{f: d[f] for f in ("kind", "chart", "u", "v", "size", "angle", "seed") if f in d})
        alpha, ids = _chart_canvas(model, dec.chart, res)
        _paint_decal(alpha, ids, dec, k + 1, res, np.random.default_rng(dec.seed + 1000 * k))
        yy, xx = np.nonzero(ids == k + 1)
        out.append((float(xx.mean() * res), float(yy.mean() * res)) if len(xx) else None)
    return out


def defect_matches(run_dir: Path, captures: dict[str, Path], tol_mm: float = 5.0, nominal: bool = False) -> dict:
    from ..anchor.template import model_of_run

    run = read_json(run_dir / "run.json")
    found = run["defects"]
    model = model_of_run(run)     # a plate when no drawing matched: its charts are within a few mm of the drawing's
    P = model.perimeter
    filmed = {"frame"} | {("screen" if f == "screen" else "back") for f in captures}
    truth = []
    for f, cap in captures.items():
        gt = read_json(cap / "gt.json")
        try:
            centres = [None] * len(gt["decals"]) if nominal else drawn_centres(cap)
        except Exception:           # captures made before the painter took seeds: the nominal points
            centres = [None] * len(gt["decals"])
        for d, c in zip(gt["decals"], centres):
            truth.append({"kind": d["kind"], "chart": d["chart"], "u": d["u"], "v": d["v"],
                          **({"u_drawn": round(c[0], 2), "v_drawn": round(c[1], 2)} if c is not None else {})})
        for d in gt.get("dents", []):
            truth.append({"kind": "dent", "chart": "frame", "u": d["s"], "v": d["v"]})
    uniq = []
    for t in truth:     # both passes carry the same defects
        if t["chart"] in filmed and not any(t["chart"] == q["chart"] and abs(t["u"] - q["u"]) < 0.5 and abs(t["v"] - q["v"]) < 0.5 for q in uniq):
            uniq.append(t)
    rows, used = [], set()
    for t in uniq:
        best, bd = None, np.inf
        for k, d in enumerate(found):
            if d["unwrap"]["chart"] != t["chart"]:
                continue
            tu, tv = t.get("u_drawn", t["u"]), t.get("v_drawn", t["v"])
            du = abs(d["unwrap"]["u"] - tu)
            if t["chart"] == "frame":
                du = min(du, P - du)
            dist = float(np.hypot(du, d["unwrap"]["v"] - tv))
            if dist < bd:
                best, bd = k, dist
        hit = best is not None and bd <= tol_mm
        if hit:
            used.add(best)
        rows.append({**t, "found": hit, "error_mm": round(bd, 2) if best is not None else None,
                     "as": found[best]["type"] if hit else None})
    return {"truth": len(rows), "found": sum(r["found"] for r in rows), "false_alarms": len(found) - len(used), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m flawmap.synth.evaluate")
    ap.add_argument("run")
    ap.add_argument("--pass", dest="passes", action="append", required=True, help="NAME=CAPTURE (pass name in the run, e.g. back)")
    ap.add_argument("--nominal", action="store_true", help="match against the nominal decal points (the rule before 2026-11-17)")
    a = ap.parse_args(argv)
    run = Path(a.run)
    caps = {k: Path(v) for k, v in (s.split("=", 1) for s in a.passes)}
    out = {"poses": {k: pose_errors(run, v, k) for k, v in caps.items()}, "defects": defect_matches(run, caps, nominal=a.nominal)}
    for k, e in out["poses"].items():
        print(f"[{k}] {e['pose_mode']}: {e['n']} keyframes, camera in device frame: rot {e['rot_deg_median']:.3f} deg "
              f"(p90 {e['rot_deg_p90']:.3f}), position {e['pos_mm_median']:.3f} mm (p90 {e['pos_mm_p90']:.3f}), "
              f"model reprojection {e['reproj_px_median']:.2f} px (p90 {e['reproj_px_p90']:.2f}, max {e['reproj_px_max']:.2f})")
    d = out["defects"]
    print(f"defects: {d['found']}/{d['truth']} found, {d['false_alarms']} false alarms")
    for r in d["rows"]:
        print(f"  {r['kind']:<7} {r['chart']:<6} u {r['u']:6.1f} v {r['v']:5.1f}: {'found as ' + r['as'] if r['found'] else 'missed'} "
              f"({r['error_mm']} mm)")
    write_json(run / ("eval_nominal.json" if a.nominal else "eval.json"), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
