"""``flawmap check``: the first look at a real capture (M1 go/no-go tests).

Answers, before anything is tuned on it:

* Does the video decode in the orientation of the ARKit intrinsics, is it HDR, are there as many
  frames as odometry rows? (risk R5 of the plan review)
* How often is the board found at each viewing height, with what reprojection error? (R4)
* How far does ARKit drift from the board poses? (R4 fallback)
* With ``--device``: how well does the reference model's outline fit the image edges (px and mm
  per height band), and what does the outline profile around the frame look like (R1, R3)?

Writes ``report.json`` and overlay images (model silhouette in green, board corners in yellow) to
the output folder.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .frames.select import frame_quality
from .geometry.transforms import inv_T
from .io.stray import iter_frames, read_stray
from .poses.align import estimate_poses
from .poses.board import BoardDetector, BoardSpec
from .utils.image import imwrite
from .utils.io import write_json

BANDS = [(0, 30), (30, 50), (50, 70), (70, 91)]


def _elev(T_world_cam: np.ndarray) -> float:
    c = T_world_cam[:3, 3]
    return float(np.degrees(np.arctan2(c[2], np.hypot(c[0], c[1]))))


def check_capture(root: str | Path, out: str | Path, device: str | None = None, face_up: str = "back",
                  n_frames: int = 120, square_mm: float | None = None, log=print) -> dict:
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cap = read_stray(root)
    rep: dict = {"capture": str(root), "n_odometry_rows": len(cap), "video": {k: v for k, v in cap.info.items() if k != "ffmpeg_banner"}}
    ts = cap.timestamps
    rep["duration_s"] = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    rep["fps_from_timestamps"] = float((len(ts) - 1) / max(ts[-1] - ts[0], 1e-6)) if len(ts) > 1 else None
    fx = cap.K[:, 0, 0]
    rep["intrinsics"] = {"fx_median": float(np.median(fx)), "fx_range": [float(fx.min()), float(fx.max())],
                         "cx_median": float(np.median(cap.K[:, 0, 2])), "cy_median": float(np.median(cap.K[:, 1, 2])),
                         "frame_size": [cap.width, cap.height]}
    step = max(1, len(cap) // n_frames)
    idx = list(range(0, len(cap), step))
    frames = dict(iter_frames(root / "rgb.mp4", idx))
    rep["n_decoded"] = len(frames)
    if len(frames) < len(idx):
        rep.setdefault("warnings", []).append(f"decoded {len(frames)} of {len(idx)} requested frames: fewer video frames than odometry rows?")
    q = {i: frame_quality(img) for i, img in frames.items()}
    sh = np.array([v["sharpness"] for v in q.values()])
    rep["sharpness"] = {"p10": float(np.percentile(sh, 10)), "p50": float(np.median(sh)), "p90": float(np.percentile(sh, 90))}
    rep["blown_highlights_median"] = float(np.median([v["clipped"] for v in q.values()]))
    board = BoardSpec.load(square_mm=square_mm)
    from .io.stray import choose_undistortion
    rep["distortion_lut"] = (root / "distortion").exists()
    rep["undistortion_choice"] = choose_undistortion(cap, frames, board, log=log)
    cams = {i: cap.camera(i) for i in frames}
    poses = estimate_poses(frames, cams, board, cap.T_ark_cam)
    rep["arkit_vs_board"] = poses.arkit_residual
    rows = []
    for i in frames:
        T = poses.T_world_cam.get(i)
        st = poses.board_stats.get(i, {})
        rows.append({"frame": i, "elev": _elev(T) if T is not None else None, "board": poses.source.get(i) == "board",
                     "markers": st.get("n_markers", 0), "corners": st.get("n_corners", 0), "reproj_px": st.get("reproj_px")})
    bands = []
    for lo, hi in BANDS:
        sel = [r for r in rows if r["elev"] is not None and lo <= r["elev"] < hi]
        det = [r for r in sel if r["board"]]
        bands.append({"elev_deg": [lo, hi], "frames": len(sel), "board_found": len(det),
                      "median_markers": float(np.median([r["markers"] for r in det])) if det else None,
                      "median_corners": float(np.median([r["corners"] for r in det])) if det else None,
                      "median_reproj_px": float(np.median([r["reproj_px"] for r in det])) if det else None})
    rep["board_by_height"] = bands
    rep["frames_without_any_pose"] = int(sum(r["elev"] is None for r in rows))
    log(f"board found in {sum(r['board'] for r in rows)}/{len(rows)} sampled frames; ARKit vs board {poses.arkit_residual}")
    if device:
        from .anchor.parametric import DeviceSpec, build_device_model, rest_pose
        from .anchor.register import edge_residuals, register_device, silhouette_points
        from .detect.outline import outline_dents, outline_profile, outline_samples
        from .surface.raycast import RaycastScene

        model = build_device_model(DeviceSpec.by_slug(device))
        board_ids = [i for i in poses.frames if poses.source[i] == "board"]
        sel = [board_ids[int(k)] for k in np.linspace(0, len(board_ids) - 1, min(24, len(board_ids))).round()] if board_ids else []
        reg = register_device(model, {i: frames[i] for i in sel}, cams, poses.T_world_cam, rest_pose(model, face_up), log=log)
        per = reg.per_view
        rep["registration"] = {"T_world_cad": reg.T_world_cad, "rounds": reg.rounds,
                               "median_px": float(np.median([v["median_px"] for v in per.values()])) if per else None,
                               "median_mm": float(np.median([v["median_mm"] for v in per.values()])) if per else None,
                               "by_height": [{"elev_deg": [lo, hi], "median_px": float(np.median([per[i]["median_px"] for i in per if lo <= _elev(poses.T_world_cam[i]) < hi]))
                                              if any(lo <= _elev(poses.T_world_cam[i]) < hi for i in per) else None} for lo, hi in BANDS]}
        S, R = outline_samples(model, edge_residuals(model, frames, cams, poses.T_world_cam, reg.T_world_cad, views=board_ids))
        if len(S):
            prof = outline_profile(model, S, R)
            rep["outline"] = {"n_samples": int(len(S)), "sigma_mm": prof["sigma_mm"], "dents": outline_dents(prof)}
            write_json(out / "outline_profile.json", prof)
        # overlays: model silhouette (green) and board corners (yellow) on a few frames per band
        scene = RaycastScene(model.mesh, None)
        det = BoardDetector(board)
        for lo, hi in BANDS:
            cand = [i for i in sel if lo <= _elev(poses.T_world_cam[i]) < hi][:2]
            for i in cand:
                img = frames[i].copy()
                T_cad_cam = inv_T(reg.T_world_cad) @ poses.T_world_cam[i]
                h = scene.cast(cams[i].scaled(0.5), T_cad_cam, step=1)
                m = cv2.resize(h.hit.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, (40, 230, 60), 1, cv2.LINE_AA)
                bp = det.pose(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY), cams[i])
                if bp is not None:
                    for x, y in bp.image_pts:
                        cv2.circle(img, (int(x), int(y)), 4, (250, 220, 0), 1, cv2.LINE_AA)
                imwrite(str(out / f"overlay_elev{lo:02d}_{i:06d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    write_json(out / "report.json", rep)
    return rep
