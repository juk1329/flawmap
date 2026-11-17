"""End-to-end processing of one inspected device (one or two capture passes).

    passes (Stray Scanner folders, screen-up and/or back-up)
      -> keyframes (sharpest per time window)
      -> board poses (+ ARKit where the board is not seen)
      -> reference-model registration per pass (edge alignment)
      -> camera poses expressed in the model frame, so the passes share one coordinate system
      -> unwrapped appearance maps (samples of every view) and coverage
      -> appearance detectors -> regions -> merge, classify (crack vs scratch by persistence)
      -> evidence frames -> viewer bundle

Outputs go to ``out/``: ``run.json`` (everything measured on the way), ``passes/<name>/`` (poses,
registration), ``bundle/`` (see ``export.bundle``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from scipy import ndimage as ndi

from .anchor.parametric import CHARTS, PARTS, DeviceSpec, build_device_model, rest_pose
from .anchor.register import edge_residuals, register_device
from .detect.appearance import Region, detect_all
from .detect.outline import dents_to_regions, merge_dents_with_chips, outline_dents, outline_profile, outline_samples
from .evidence.select import persistence, pick_diverse, project_region, region_outline, render_evidence, view_scores
from .errors import FlawmapError
from .export import bundle as B
from .frames.select import frame_quality, select_keyframes
from .geometry.transforms import inv_T
from .io.stray import choose_undistortion, distortion_lut, iter_frames, read_stray, undistort_maps
from .poses.align import estimate_poses
from .poses.board import BoardSpec
from .surface.maps import ChartMaps, ViewSamples, chart_geometry, collect_samples, compute_stats, coverage, prefer_sharp_views
from .utils.io import write_json
from .utils.logging import add_file_handler, get_logger

log = get_logger("flawmap.pipeline")

TYPE_KO = {"chip": "프레임 까짐", "dent": "프레임 찍힘", "crack": "유리 깨짐", "scratch": "흠집"}
PART_KO = {"screen": "액정", "back": "후면", "frame": "프레임", "camera": "카메라", "lens": "카메라 렌즈", "button": "버튼"}


@dataclass
class PassInput:
    root: Path
    face_up: str          # "screen" or "back"
    name: str = ""

    def __post_init__(self):
        self.root = Path(self.root)
        self.name = self.name or self.face_up


def _gid(p: int, i: int) -> int:
    return p * 100000 + i


def _spread(ids: list[int], n: int) -> list[int]:
    if len(ids) <= n:
        return list(ids)
    return [ids[int(k)] for k in np.linspace(0, len(ids) - 1, n).round()]


def merge_maps(parts: list[dict[str, ChartMaps]]) -> dict[str, ChartMaps]:
    out = {}
    for c in CHARTS:
        items = [p[c] for p in parts if c in p]
        if not items:
            continue
        g = items[0].geom
        frames = sum((it.samples.frames for it in items), [])
        rgb = np.concatenate([it.samples.rgb for it in items], 0)
        cos = np.concatenate([it.samples.cos for it in items], 0)
        out[c] = ChartMaps(g, ViewSamples(frames, rgb, cos))
    return out


def merge_regions(regions: list[Region], maps: dict[str, ChartMaps], gap_mm: float = 1.5) -> list[Region]:
    """Merge regions of the same chart and kind that lie within ``gap_mm`` of each other."""
    from skimage.measure import label

    out = []
    keys = sorted({(r.chart, r.kind) for r in regions})
    for chart, kind in keys:
        rs = [r for r in regions if r.chart == chart and r.kind == kind]
        if len(rs) == 1:
            out += rs
            continue
        g = maps[chart].geom
        m = np.zeros(g.shape, bool)
        owner = np.full(g.shape, -1, int)
        for k, r in enumerate(rs):
            rr, cc = np.divmod(g.flat_index[r.texels], g.shape[1])
            m[rr, cc] = True
            owner[rr, cc] = k
        lab = label(ndi.binary_dilation(m, iterations=max(int(round(gap_mm / 2 / g.res)), 1)), connectivity=2)
        groups: dict[int, list[int]] = {}
        for k, r in enumerate(rs):
            rr, cc = np.divmod(g.flat_index[r.texels[:1]], g.shape[1])
            groups.setdefault(int(lab[rr[0], cc[0]]), []).append(k)
        for ks in groups.values():
            if len(ks) == 1:
                out.append(rs[ks[0]])
                continue
            tex = np.unique(np.concatenate([rs[k].texels for k in ks]))
            uv = g.uv_of(tex)
            ext = uv.max(0) - uv.min(0)
            out.append(Region(chart, kind, tex, max(rs[k].score for k in ks),
                              {"area_mm2": float(len(tex) * g.res ** 2), "length_mm": float(np.hypot(*ext)),
                               "merged_from": len(ks)}))
    return out


def describe_location(model, chart: str, u: float, v: float) -> str:
    W, L = model.spec.width, model.spec.length
    if chart == "screen":
        return f"화면을 볼 때 왼쪽에서 {u:.0f} mm, 위에서 {v:.0f} mm"
    if chart == "back":
        return f"뒷면을 볼 때 왼쪽에서 {u:.0f} mm, 위에서 {v:.0f} mm"
    # frame: which side, measured from the nearest corner of that side (s is clockwise from top centre)
    P = model.perimeter
    xy = np.array([np.interp(u, np.r_[model.outline_s, P], np.r_[model.outline_xy[:, 0], model.outline_xy[0, 0]]),
                   np.interp(u, np.r_[model.outline_s, P], np.r_[model.outline_xy[:, 1], model.outline_xy[0, 1]])])
    x, y = xy
    near_x, near_y = abs(abs(x) - W / 2) < 3.0, abs(abs(y) - L / 2) < 3.0
    side_x = "오른쪽" if x > 0 else "왼쪽"
    side_y = "위" if y > 0 else "아래"
    if near_x and near_y:
        where = f"{side_y}{side_x} 모서리"
    elif near_x:
        where = f"{side_x} 옆면, {'위' if y > 0 else '아래'} 끝에서 {L / 2 - abs(y):.0f} mm"
    elif near_y:
        where = f"{'윗' if y > 0 else '아랫'}면, {side_x} 끝에서 {W / 2 - abs(x):.0f} mm"
    else:
        where = f"{side_y}{side_x} 모서리 근처"
    return f"{where} (화면 기준)"


@dataclass
class PreparedPass:
    """One capture pass, ready for the shared steps: keyframes with cameras posed in the model frame.

    Frame ids are global (``_gid``), so the passes of one device can be merged."""
    name: str
    face_up: str
    frames: dict[int, np.ndarray]
    cams: dict
    T_cad_cam: dict[int, np.ndarray]
    sharpness: dict[int, float]
    info: dict[int, dict]                 # gid -> {pass, frame, time_s}
    video: Path | None
    report: dict
    outline_S: np.ndarray
    outline_R: np.ndarray
    pose_mode: str = "board"

    @property
    def charts(self) -> list[str]:
        return ["screen", "frame"] if self.face_up == "screen" else ["back", "frame"]


Progress = Callable[[str, float, str], None]


class _Stages:
    """Maps stage progress onto 0..1 for the whole run (monotonic) and keeps stage timings.

    The schedule is: decode, poses, register for every pass; colour samples for every pass; then
    detection, evidence and the bundle. Weights are seconds measured on the two-pass synthetic demo
    (2026-11-16): per pass decode 9, board poses 4, registration and edge residuals 51, samples 46;
    then detection 35, evidence 14, bundle 47."""
    PREP = {"decode": 9.0, "poses": 4.0, "register": 51.0}
    SAMPLES = 46.0
    FINAL = {"detect": 35.0, "evidence": 14.0, "bundle": 47.0}

    def __init__(self, n_passes: int, cb: Progress | None):
        n = max(n_passes, 1)
        sched = [((k, p), w) for p in range(n) for k, w in self.PREP.items()]
        sched += [(("samples", p), self.SAMPLES) for p in range(n)]
        sched += [((k, None), w) for k, w in self.FINAL.items()]
        total = sum(w for _, w in sched)
        self.start, self.weight, acc = {}, {}, 0.0
        for key, w in sched:
            self.start[key], self.weight[key] = acc / total, w / total
            acc += w
        self.cb = cb
        self.timing: dict[str, float] = {}
        self._t, self._key = time.time(), None

    def at(self, stage: str, p: int | None = None, frac: float = 0.0) -> None:
        key = (stage, p)
        if frac == 0.0 and key != self._key:
            now = time.time()
            if self._key is not None:
                self.timing[self._key[0]] = round(self.timing.get(self._key[0], 0.0) + now - self._t, 1)
            self._t, self._key = now, key
        if self.cb and key in self.start:
            self.cb(stage, min(1.0, self.start[key] + self.weight[key] * frac), "")

    def done(self) -> None:
        self.at("_end")
        if self.cb:
            self.cb("bundle", 1.0, "")


def prepare_stray_pass(p: PassInput, p_i: int, model, board: BoardSpec, keyframes: int = 140, reg_views: int = 36,
                       poses_mode: str = "auto", stages: _Stages | None = None, max_reg_px: float = 1.5,
                       plate=None, refine: str | None = None, segment: bool = False) -> PreparedPass:
    """A Stray Scanner pass: keyframes, camera poses (board, ARKit), model registration.
    ``segment``: SAM 2 device masks (``segment.device_masks``) to judge the registration, choose
    between pose alternatives, and keep floor pixels out of the colour samples."""
    stages = stages or _Stages(1, None)
    t0 = time.time()
    stages.at("decode", p_i)
    cap = read_stray(p.root)
    log.info(f"[{p.name}] {len(cap)} frames {cap.width}x{cap.height}, video {cap.info.get('codec')} rot {cap.info.get('rotation')}")
    quality = {i: frame_quality(img) for i, img in iter_frames(p.root / "rgb.mp4", scale=0.5)}
    keys = select_keyframes(quality, cap.timestamps[: len(quality)], target=keyframes)
    frames = dict(iter_frames(p.root / "rgb.mp4", keys))
    if len(frames) < 8:
        raise FlawmapError("VIDEO_SHORT", f"[{p.name}] only {len(frames)} keyframes decoded")
    undist = choose_undistortion(cap, frames, board, log=lambda m: log.info(f"[{p.name}] {m}"))
    if undist != "none":
        for i in list(frames):
            lut = distortion_lut(cap, i)
            c = cap.distortion_center[i]
            if lut is None or not np.all(np.isfinite(c)):
                continue
            if c[0] > cap.width or c[1] > cap.height:
                c = c * (cap.width / (2 * c[0]))
            mx, my = undistort_maps(lut, (float(c[0]), float(c[1])), cap.width, cap.height, inverse=(undist == "inverse"))
            frames[i] = cv2.remap(frames[i], mx, my, cv2.INTER_LINEAR)
    cams = {i: cap.camera(i) for i in frames}
    stages.at("poses", p_i)
    poses = estimate_poses(frames, cams, board, cap.T_ark_cam) if poses_mode != "boardless" else None
    board_ids = [i for i in poses.frames if poses.source[i] == "board"] if poses is not None else []
    use_board = poses_mode == "board" or (poses_mode == "auto" and len(board_ids) >= 10)
    if use_board:
        log.info(f"[{p.name}] keyframes {len(frames)}; board poses {len(board_ids)}, ARKit-filled {len(poses.frames) - len(board_ids)}; "
                 f"ARKit vs board {poses.arkit_residual}")
        if len(board_ids) < 3:
            raise FlawmapError("NO_POSE", f"[{p.name}] capture board found in {len(board_ids)} of {len(frames)} keyframes")
        T_init, pose_ids, searches, pose_mode = rest_pose(model, p.face_up), board_ids, (12.0, 5.0, 3.0, 2.5), "board"
    else:
        from .poses.boardless import boardless_poses

        log.info(f"[{p.name}] keyframes {len(frames)}; board in {len(board_ids)}: camera poses from ARKit and the model outline")
        if cap.T_ark_cam is None or len(cap.T_ark_cam) < len(cap):
            raise FlawmapError("NO_POSE", f"[{p.name}] no ARKit trajectory and no board")
        bl = boardless_poses(cap, frames, cams, model, p.face_up, plate=plate, log=lambda m: log.info(f"[{p.name}]{m}"))
        poses, T_init, pose_ids, searches, pose_mode = bl.poses, bl.T_world_cad, bl.poses.frames, (3.0, 2.5), "boardless"
    colmap_info = None
    stages.at("register", p_i)
    reg_ids = _spread(pose_ids, reg_views)

    def _register(T_wc, srch):
        r = register_device(model, {i: frames[i] for i in reg_ids}, cams, T_wc, T_init, searches=srch,
                            log=lambda m: log.info(f"[{p.name}]{m}"))
        pv_ = list(r.per_view.values())
        return r, (float(np.median([v["median_px"] for v in pv_])) if pv_ else float("inf"))

    reg, med_px = _register(poses.T_world_cam, searches)
    masks, mask_iou = None, None

    def _iou(T_wc, r):
        return iou_report(model, masks, cams, {i: inv_T(r.T_world_cad) @ T_wc[i] for i in masks if i in T_wc})

    if segment:
        from .segment.device_masks import available, box_of, device_masks, iou_report, silhouette

        if available():
            # prompts from the first registration (board, or boardless started from the floor-mosaic
            # plate), on six keyframes; SAM 2 then follows the device on its own through the rest
            from .surface.raycast import RaycastScene

            sc = RaycastScene(model.mesh, None)
            pids = [pose_ids[int(k)] for k in np.linspace(0, len(pose_ids) - 1, min(6, len(pose_ids))).round()]
            boxes = {}
            for i in pids:
                b = box_of(silhouette(sc, cams[i], inv_T(reg.T_world_cad) @ poses.T_world_cam[i], 0.25))
                boxes[i] = None if b is None else b / 0.25
            masks = device_masks({i: frames[i] for i in poses.frames}, boxes, log=lambda m: log.info(f"[{p.name}]{m}"))
            mask_iou = _iou(poses.T_world_cam, reg)
            log.info(f"[{p.name}]  mask IoU {mask_iou['iou_p50']} (p10 {mask_iou['iou_p10']}) with {pose_mode} poses")
        else:
            log.info(f"[{p.name}]  segment requested but SAM 2 / torch / the checkpoint is not available")
    if refine == "colmap" and pose_mode == "boardless":
        # every keyframe on its own by bundle adjustment on the floor's texture (poses/colmap.py). The
        # similarity onto this world leaves a few mm of common offset (camera centres p50 2.5-6.8 mm
        # on the iPad), so the device is registered again from wide search windows; the COLMAP poses
        # are kept only when the outline fits at least about as well as with the boardless poses
        # (on the iPad screen pass the old 8 px windows left the device 1 cm off, 2026-11-17).
        from .poses.colmap import refine_with_colmap

        new_T, colmap_info = refine_with_colmap(frames, cams, poses.T_world_cam, log=lambda m: log.info(f"[{p.name}]{m}"))
        if new_T is not None:
            reg_c, med_c = _register(new_T, (24.0, 12.0, 6.0, 3.0, 2.5))
            colmap_info.update(outline_px_boardless=round(med_px, 3), outline_px_colmap=round(med_c, 3))
            if masks is not None:
                # the masks decide: the outline residual alone let a 1 cm offset through (screen pass,
                # IoU 0.880 against 0.979 for the right registration)
                iou_c = _iou(new_T, reg_c)
                colmap_info.update(mask_iou_boardless=mask_iou["iou_p50"], mask_iou_colmap=iou_c["iou_p50"])
                take = iou_c["iou_p50"] >= mask_iou["iou_p50"] - 0.003 and med_c <= 1.5 * med_px
                why = f"mask IoU {iou_c['iou_p50']} with COLMAP poses vs {mask_iou['iou_p50']} without"
            else:
                take = med_c <= 1.1 * med_px
                why = f"outline {med_c:.2f} px with COLMAP poses vs {med_px:.2f} px without"
            if take:
                poses.T_world_cam = new_T
                poses.source = {i: "colmap" for i in poses.frames}
                reg, med_px, pose_mode = reg_c, med_c, "boardless+colmap"
                if masks is not None:
                    mask_iou = iou_c
            else:
                colmap_info.update(used=False, reason=why)
                log.info(f"[{p.name}]  colmap not used: {why}")
        else:
            log.info(f"[{p.name}]  colmap not used: {colmap_info.get('reason')}")
    pv = list(reg.per_view.values())
    if not np.isfinite(med_px) or med_px > max_reg_px:
        raise FlawmapError("DEVICE_NOT_FOUND", f"[{p.name}] outline residual {med_px:.2f} px after registration")
    Tcad = {i: inv_T(reg.T_world_cad) @ poses.T_world_cam[i] for i in poses.frames}
    stages.at("register", p_i, 0.5)
    edges = edge_residuals(model, frames, cams, poses.T_world_cam, reg.T_world_cad, views=_spread(pose_ids, 100))
    s_p, r_p = outline_samples(model, edges)
    g_frames = {_gid(p_i, i): frames[i] for i in poses.frames}
    info = {_gid(p_i, i): {"pass": p.name, "frame": int(i), "time_s": float(cap.timestamps[i] - cap.timestamps[0])} for i in poses.frames}
    rep = {"name": p.name, "face_up": p.face_up, "root": str(p.root), "n_frames": len(cap), "n_keyframes": len(frames),
           "n_board_poses": len(board_ids), "pose_mode": pose_mode, "arkit_vs_board": poses.arkit_residual, "undistortion": undist,
           "colmap": colmap_info,
           "mask_iou": {k: v for k, v in mask_iou.items() if k != "per_frame"} if mask_iou else None,
           "registration": {"T_world_cad": reg.T_world_cad, "rounds": reg.rounds, "median_px": med_px,
                            "median_mm": float(np.median([v["median_mm"] for v in pv])) if pv else None},
           "video": {k: v for k, v in cap.info.items() if k != "ffmpeg_banner"}}
    prep = PreparedPass(p.name, p.face_up, g_frames, {_gid(p_i, i): cams[i] for i in poses.frames},
                        {_gid(p_i, i): Tcad[i] for i in poses.frames},
                        {_gid(p_i, i): quality[i]["sharpness"] for i in poses.frames}, info, p.root / "rgb.mp4", rep, s_p, r_p, pose_mode)
    prep._poses, prep._reg = poses, reg
    prep._masks = {_gid(p_i, i): m for i, m in masks.items()} if masks else None
    rep["seconds"] = round(time.time() - t0, 1)
    return prep


def prepare_uco3d_pass(seq, face_up: str, p_i: int = 0, keyframes: int = 100, n_fit_views: int = 24,
                       assumed_width_mm: float = 71.5, stages: _Stages | None = None):
    """A uCO3D sequence without a reference drawing: SfM cameras as given, a plate model fitted to the
    outline (``anchor.template``), the scene scaled so the fitted width is ``assumed_width_mm``
    (SfM has no metric scale). Returns (PreparedPass, model)."""
    from .anchor.template import axes_box, fit_dimensions, mask_iou, plate_box, template_spec
    from .io.uco3d import read_frames, read_points

    stages = stages or _Stages(1, None)
    t0 = time.time()
    stages.at("decode", p_i)
    allf = read_frames(seq, seq.frame_numbers, undistorted=False)
    quality = {fn: frame_quality(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)) for fn, img in allf.items()}
    ts = np.zeros(max(seq.frame_numbers) + 1)
    for fn, t in zip(seq.frame_numbers, seq.timestamps):
        ts[fn] = t
    keys = select_keyframes(quality, ts, target=keyframes)
    from .io.uco3d import undistort

    frames = {fn: undistort(allf[fn], seq.K[fn], seq.k1[fn]) for fn in keys}
    del allf
    stages.at("poses", p_i)
    pts = read_points(seq)
    centres = np.array([seq.T_world_cam[fn][:3, 3] for fn in keys])
    cams = {fn: seq.camera(fn) for fn in keys}
    views = _spread(keys, n_fit_views)
    stages.at("register", p_i)
    # two starts (smallest rectangle in the face plane, principal axes); the outline residual hardly
    # tells their fits apart (59015-27660-21826: 0.398 vs 0.404 px for plates 9.5 mm apart in length),
    # so the dataset's own object masks decide
    mviews = _spread(keys, 8)
    masks = read_frames(seq, mviews, masks=True)
    trials = []
    for how, box in (("rectangle", plate_box), ("axes", axes_box)):
        T0, size0 = box(pts, centres, face_up)
        k = assumed_width_mm / max(size0[1], 1e-9)               # SfM units -> assumed mm
        T_wc = {fn: seq.T_world_cam[fn].copy() for fn in keys}
        for T in T_wc.values():
            T[:3, 3] *= k
        T0 = T0.copy()
        T0[:3, 3] *= k
        L0 = size0[0] * k
        if not 1.2 <= L0 / assumed_width_mm <= 3.0:
            continue
        spec0 = template_spec(L0, assumed_width_mm, 0.052 * L0, 0.068 * L0)
        spec_t, T_wcad_t, fit_t = fit_dimensions(spec0, frames, cams, T_wc, T0, views, log=lambda m: log.info(f"[{face_up}] ({how}){m}"))
        iou = mask_iou(build_device_model(spec_t, corner_step=0.5, edge_step=2.0, profile_step=0.3), T_wcad_t, cams, T_wc, masks)
        log.info(f"[{face_up}] start '{how}': plate {spec_t.length:.1f} x {spec_t.width:.1f}, residual {fit_t['median_px']:.3f} px, mask IoU {iou:.3f}")
        trials.append((iou, how, spec_t, T_wcad_t, fit_t, T_wc, k))
    if not trials:
        raise FlawmapError("DEVICE_NOT_FOUND", f"[{seq.name}] no plate-shaped object in the point cloud")
    trials.sort(key=lambda t: -t[0])
    iou, how, spec, T_wcad, fit, T_wc, k = trials[0]
    fit = {**fit, "start": how, "mask_iou": iou, "starts": [{"start": t[1], "mask_iou": round(t[0], 3), "size_mm": [round(t[2].length, 1),
                                                                    round(t[2].width, 1)]} for t in trials]}
    ratio = spec.length / spec.width
    if not 1.2 <= ratio <= 2.6:          # phones are 1.9-2.3 times longer than wide, 4:3 tablets ~1.3-1.45
        raise FlawmapError("DEVICE_NOT_FOUND", f"[{seq.name}] fitted plate {spec.length:.1f} x {spec.width:.1f}: "
                                               f"length/width {ratio:.2f} is not a phone outline")
    k2 = assumed_width_mm / spec.width                        # keep "width = assumed" exactly
    for T in T_wc.values():
        T[:3, 3] *= k2
    T_wcad = T_wcad.copy()
    T_wcad[:3, 3] *= k2
    spec = template_spec(spec.length * k2, assumed_width_mm, spec.thickness * k2, spec.raw["corner_radius"] * k2)
    model = build_device_model(spec)
    reg = register_device(model, {i: frames[i] for i in views}, cams, T_wc, T_wcad, searches=(3.0, 2.5),
                          log=lambda m: log.info(f"[{face_up}]{m}"))
    pv = list(reg.per_view.values())
    med_px = float(np.median([v["median_px"] for v in pv])) if pv else float("inf")
    if not np.isfinite(med_px) or med_px > 2.5:
        raise FlawmapError("DEVICE_NOT_FOUND", f"[{seq.name}] plate outline residual {med_px:.2f} px")
    stages.at("register", p_i, 0.5)
    edges = edge_residuals(model, frames, cams, T_wc, reg.T_world_cad, views=_spread(keys, 60))
    s_p, r_p = outline_samples(model, edges)
    Tcad = {fn: inv_T(reg.T_world_cad) @ T_wc[fn] for fn in keys}
    t_first = float(seq.timestamps[0])
    info = {_gid(p_i, fn): {"pass": face_up, "frame": int(seq.video_index(fn)), "time_s": float(ts[fn] - t_first)} for fn in keys}
    rep = {"name": face_up, "face_up": face_up, "root": str(seq.root), "source": f"uco3d:{seq.name}", "n_frames": len(seq.frame_numbers),
           "n_keyframes": len(keys), "n_board_poses": 0, "pose_mode": "sfm", "caption": seq.caption,
           "template": {**fit, "assumed_width_mm": assumed_width_mm, "scale_mm_per_sfm_unit": k * k2},
           "registration": {"T_world_cad": reg.T_world_cad, "rounds": reg.rounds, "median_px": med_px,
                            "median_mm": float(np.median([v["median_mm"] for v in pv])) if pv else None},
           "seconds": round(time.time() - t0, 1)}
    prep = PreparedPass(face_up, face_up, {_gid(p_i, fn): frames[fn] for fn in keys}, {_gid(p_i, fn): cams[fn] for fn in keys},
                        {_gid(p_i, fn): Tcad[fn] for fn in keys}, {_gid(p_i, fn): quality[fn]["sharpness"] for fn in keys},
                        info, seq.video, rep, s_p, r_p, "sfm")
    prep._reg = reg
    return prep, model


def run_uco3d(name: str, out: str | Path, face_up: str = "screen", data_root: str | Path = "data/datasets/uco3d",
              keyframes: int = 100, progress: Progress | None = None, meta: dict | None = None) -> dict:
    """Process a public uCO3D phone video with a fitted plate model (no drawing, no metric scale)."""
    from .io.uco3d import load_sequence

    out = Path(out)
    add_file_handler(out / "run.log")
    t_start = time.time()
    seq = load_sequence(data_root, name)
    log.info(f"uCO3D {name}: {len(seq.frame_numbers)} annotated frames {seq.width}x{seq.height}; caption: {seq.caption[:120]}")
    stages = _Stages(1, progress)
    prep, model = prepare_uco3d_pass(seq, face_up, keyframes=keyframes, stages=stages)
    return process([prep], model, out, "template", stages=stages, t_start=t_start, anchor="template",
                   meta={**(meta or {}), "source": f"uco3d:{name}", "caption": seq.caption,
                         "scale": f"assumed: device width {prep.report['template']['assumed_width_mm']} mm"})


def drop_mirror_symmetric(regions: list[Region], maps: dict[str, ChartMaps], share: float = 0.3, gap_mm: float = 1.5):
    """Without a drawing, the earpiece, the home button and the display border look like defects to
    the line detector. They are left-right symmetric; damage almost never is. A screen or back region
    whose mirror image (about the device's long centre line) falls on detections, its own included,
    over ``share`` of its area is dropped. Returns (kept, dropped records)."""
    keep, dropped = [], []
    for chart in ("screen", "back"):
        rs = [r for r in regions if r.chart == chart]
        if not rs:
            continue
        g = maps[chart].geom
        union = np.zeros(g.shape, bool)
        for r in rs:
            rr, cc = np.divmod(g.flat_index[r.texels], g.shape[1])
            union[rr, cc] = True
        union = ndi.binary_dilation(union, iterations=max(int(round(gap_mm / g.res)), 1))
        for r in rs:
            rr, cc = np.divmod(g.flat_index[r.texels], g.shape[1])
            frac = float(union[rr, g.shape[1] - 1 - cc].mean())
            if frac >= share:
                uv = g.uv_of(r.texels).mean(0)
                dropped.append({"chart": chart, "kind": r.kind, "u": float(uv[0]), "v": float(uv[1]), "mirror_share": round(frac, 2)})
            else:
                keep.append(r)
    keep += [r for r in regions if r.chart not in ("screen", "back")]
    return keep, dropped


def drop_border_lines(regions: list[Region], maps: dict[str, ChartMaps], model, margin_mm: float = 10.0,
                      min_length_mm: float = 12.0, max_width_mm: float = 2.0, max_angle_deg: float = 6.0):
    """The edge of the display panel (or of a bezel print) is a long straight line parallel to the
    device outline a few millimetres inside it. Without a drawing it is not masked, and the mirror
    test misses it when only one side shows up. A screen/back line region that is long, thin,
    within ``max_angle_deg`` of a chart axis and entirely within ``margin_mm`` of the outline is
    dropped. Returns (kept, dropped records)."""
    keep, dropped = [], []
    for r in regions:
        if r.chart not in ("screen", "back") or r.kind not in ("crack", "scratch"):
            keep.append(r)
            continue
        g = maps[r.chart].geom
        uv = g.uv_of(r.texels)
        W, L = model.spec.width, model.spec.length
        c = uv - uv.mean(0)
        if len(c) < 3:
            keep.append(r)
            continue
        _, sv, Vt = np.linalg.svd(c, full_matrices=False)
        ext = c @ Vt.T
        length, width = float(np.ptp(ext[:, 0])), float(np.ptp(ext[:, 1]))
        ang = float(np.degrees(np.arctan2(abs(Vt[0, 1]), abs(Vt[0, 0]))))         # 0: along u, 90: along v
        off_axis = min(ang, 90.0 - ang)
        edge_d = np.minimum.reduce([uv[:, 0], W - uv[:, 0], uv[:, 1], L - uv[:, 1]])
        if length >= min_length_mm and width <= max_width_mm and off_axis <= max_angle_deg and edge_d.max() <= margin_mm:
            m = uv.mean(0)
            dropped.append({"chart": r.chart, "kind": r.kind, "u": float(m[0]), "v": float(m[1]), "length_mm": round(length, 1),
                            "angle_off_axis_deg": round(off_axis, 1)})
        else:
            keep.append(r)
    return keep, dropped


SIDE_KO = {"left": "왼쪽", "right": "오른쪽", "top": "윗", "bottom": "아랫"}


def coverage_advice(maps: dict[str, ChartMaps], model, cov: dict, good_views: int = 2) -> list[str]:
    """What to film again, in the seller's words: faces never filmed, faces partly filmed, frame
    sides seen in too few views (sides as seen from the screen)."""
    out = []
    bc = cov.get("by_chart", {})
    for c, how in (("screen", "화면이 위로 오게"), ("back", "뒷면이 위로 오게")):
        b = bc.get(c)
        if b is None:
            continue
        if not b["captured"]:
            out.append(f"{PART_KO[c]}을 확인하지 못했습니다. 기기를 {how} 놓고 한 번 더 찍어 주세요.")
        elif b["seen"] < 0.9:
            out.append(f"{PART_KO[c]}의 {100 - round(b['seen'] * 100)}%가 덜 찍혔습니다. 높은 각도에서 위를 한 번 더 훑어 주세요.")
    if "frame" in maps and "n_good" in maps["frame"].stats:
        g = maps["frame"].geom
        good = maps["frame"].stats["n_good"] >= good_views
        x, y = g.point[:, 0], g.point[:, 1]
        W, L = model.spec.width / 2, model.spec.length / 2
        sides = {"right": (x > W - 2.5) & (np.abs(y) < L - 10), "left": (x < -W + 2.5) & (np.abs(y) < L - 10),
                 "top": (y > L - 2.5) & (np.abs(x) < W - 10), "bottom": (y < -L + 2.5) & (np.abs(x) < W - 10)}
        weak = [k for k, m in sides.items() if m.any() and good[m].mean() < 0.8]
        if weak:
            names = ", ".join(f"{SIDE_KO[k]}{'면' if k in ('top', 'bottom') else ' 옆면'}" for k in weak)
            out.append(f"{names}이 덜 찍혔습니다. 그쪽을 낮은 각도(약 20°)에서 천천히 더 찍어 주세요.")
    return out


def run(passes: list[PassInput], device: str, out: str | Path, keyframes: int = 140, reg_views: int = 36,
        detect_cfg: dict | None = None, board_square_mm: float | None = None, poses: str = "auto",
        progress: Progress | None = None, meta: dict | None = None, board_path: str | Path | None = None,
        refine: str | None = None, sharp_views: bool = False, segment: bool = False) -> dict:
    """Process one device from Stray Scanner passes. ``progress(stage, fraction, note)`` is called
    along the way; ``meta`` (inspection id, code version, input hashes) is stored with the result.
    ``board_path``: the capture board's YAML (default the A3 board; tablets lie on the A2 mat)."""
    out = Path(out)
    add_file_handler(out / "run.log")
    t_start = time.time()
    board = BoardSpec.load(board_path, square_mm=board_square_mm)
    stages = _Stages(len(passes), progress)
    anchor, meta = "cad", dict(meta or {})
    plates: list = [None] * len(passes)
    if device == "auto":
        plates = measure_plates(passes) if poses != "board" else plates
        spec, how = estimate_device(passes[0], board, keyframes=min(keyframes, 60), poses=poses, plate=plates[0])
        meta["device_estimate"] = how
        anchor = "cad-estimated" if spec.slug != "template" else "template"
        device = spec.slug
        model = build_device_model(spec)
    else:
        model = build_device_model(DeviceSpec.by_slug(device))
    prepared = [prepare_stray_pass(p, p_i, model, board, keyframes, reg_views, poses, stages, plate=plates[p_i], refine=refine,
                                   segment=segment)
                for p_i, p in enumerate(passes)]
    for pp in prepared:
        log.info(f"[{pp.name}] done in {pp.report['seconds']:.0f}s")
    meta.setdefault("quality", {"refine": refine, "sharp_views": sharp_views, "segment": segment})
    return process(prepared, model, out, device, detect_cfg=detect_cfg, stages=stages, t_start=t_start, meta=meta, anchor=anchor,
                   sharp_views=sharp_views)


def measure_plates(passes: list[PassInput]) -> list:
    """Floor-mosaic plates of every pass (``measure_pass_plate``), in three rounds:

    1. each pass on its own; a textured face also gives the top face's height (the thickness);
    2. passes that could not measure it (black glass) again with that height, so their outline is
       taken at the top face and not spread out by parallax;
    3. the shape (length, width, corner radius) of the best fit (highest IoU) held fixed on the
       others, which only get position and heading: on the iPad back the photographer's shadow joined
       the device in the mask while the black screen pass fitted cleanly (2026-11-17)."""
    plates = [measure_pass_plate(p) for p in passes]
    tops = [pl.top_height for pl in plates if pl is not None and pl.top_height is not None]
    hint = float(np.median(tops)) if tops else None
    if hint is not None:
        for k, p in enumerate(passes):
            if plates[k] is None or plates[k].top_height is None:
                pl = measure_pass_plate(p, top_hint=hint)
                if pl is not None:
                    pl.top_height = hint
                    plates[k] = pl
    ok = [k for k, pl in enumerate(plates) if pl is not None]
    if not ok:
        return plates
    best = max(ok, key=lambda k: plates[k].iou)
    b = plates[best]
    out = []
    for k, p in enumerate(passes):
        if k == best:
            out.append(b)
            continue
        pl = measure_pass_plate(p, size=(b.length, b.width, b.corner_radius), top_hint=hint)
        if pl is not None and pl.top_height is None:
            pl.top_height = hint
        out.append(pl)
    log.info(f"floor plates: shape from pass {passes[best].name} ({b.length:.1f} x {b.width:.1f} mm, r {b.corner_radius:.1f}, "
             f"IoU {b.iou:.3f}), top {hint}; fixed-shape IoU {[round(pl.iou, 3) if pl else None for pl in out]}")
    return out


def estimate_device(p: PassInput, board: BoardSpec, keyframes: int = 60, poses: str = "auto", tol_mm: float = 1.0,
                    plate=None):
    """Model unknown: pose the pass with a plate (measured on the floor mosaic when there is one, else a
    generic phone), fit the plate's size to the outline, and take the drawing-based device whose length
    and width match within ``tol_mm`` (else the plate). Needs metric cameras (board or ARKit).
    Returns (spec, record)."""
    from .anchor.template import GENERIC_PHONE, fit_dimensions, identify, template_spec

    if plate is not None:
        # start from the measured plate: the generic phone plate never finds a tablet
        thick = plate.top_height if plate.top_height and 2.0 < plate.top_height < 15.0 else GENERIC_PHONE["thickness"]
        spec0 = template_spec(plate.length, plate.width, thick, plate.corner_radius)
    else:
        spec0 = template_spec(**GENERIC_PHONE)
    prep = prepare_stray_pass(p, 0, build_device_model(spec0), board, keyframes, 24, poses, None, max_reg_px=3.0, plate=plate)
    ids = sorted(prep.frames)
    spec, _, fit = fit_dimensions(spec0, prep.frames, prep.cams, prep.T_cad_cam, np.eye(4), _spread(ids, 24),
                                  log=lambda m: log.info(f"[{p.name}] (size){m}"))
    L, W, T = fit["size_mm"]
    hit = identify(L, W, tol_mm)
    rec = {"size_mm": [round(L, 2), round(W, 2), round(T, 2)], "corner_radius": fit["corner_radius"],
           "floor_plate": plate.info if plate is not None else None,
           "matched": hit.slug if hit else None, "matched_name": hit.name if hit else None, "tolerance_mm": tol_mm}
    log.info(f"[{p.name}] fitted plate {L:.2f} x {W:.2f} x {T:.2f} mm -> {hit.name if hit else 'no drawing within ' + str(tol_mm) + ' mm'}")
    return (hit if hit else spec), rec


def measure_pass_plate(p: PassInput, n_views: int = 60, size: tuple | None = None, top_hint: float | None = None):
    """The device's size and pose on the floor from a Stray pass with ARKit and LiDAR, before any
    model exists (``anchor.floor_outline``); None without depth or when nothing device-like is found."""
    from .anchor.floor_outline import measure_plate
    from .poses.boardless import T_W_ARK, axes_meeting_point, floor_height

    cap = read_stray(p.root)
    if not cap.has_depth or cap.T_ark_cam is None:
        return None
    n = len(cap)
    ids = sorted({int(k) for k in np.linspace(0, n - 6, n_views).round()})
    frames = dict(iter_frames(p.root / "rgb.mp4", ids, scale=0.5))
    ids = [i for i in ids if i in frames]
    cams = {i: cap.camera(i).scaled(0.5) for i in ids}
    Tw = {i: T_W_ARK @ cap.T_ark_cam[i] for i in ids}
    centre = axes_meeting_point([Tw[i] for i in ids])
    early = ids[: max(8, len(ids) * 3 // 10)]
    zf = floor_height(cap, early, Tw, centre)
    if zf is None:
        return None
    return measure_plate(frames, cams, Tw, zf, centre[:2], scale=1.0, size=size, top_hint=top_hint, log=lambda m: log.info(f"[{p.name}]{m}"))


def process(prepared: list[PreparedPass], model, out: Path, device: str, detect_cfg: dict | None = None,
            stages: _Stages | None = None, t_start: float | None = None, meta: dict | None = None,
            anchor: str = "cad", sharp_views: bool = False) -> dict:
    """The steps shared by every input: colour samples, coverage, detection, evidence, bundle.
    ``sharp_views``: statistics and detection from each chart's sharpest views (``prefer_sharp_views``)."""
    out = Path(out)
    (out / "bundle" / "evidence").mkdir(parents=True, exist_ok=True)
    (out / "bundle" / "spin").mkdir(parents=True, exist_ok=True)
    stages = stages or _Stages(len(prepared), None)
    t_start = t_start or time.time()
    geoms = {c: chart_geometry(model, c) for c in CHARTS}
    pass_maps, frames_all, cams_all, Tcad_all, sharp_all, info_all = [], {}, {}, {}, {}, {}
    for p_i, pp in enumerate(prepared):
        stages.at("samples", p_i)
        pm = collect_samples(model, {c: geoms[c] for c in pp.charts}, pp.frames, pp.cams, pp.T_cad_cam, np.eye(4),
                             masks=getattr(pp, "_masks", None))
        if sharp_views:
            sel = {c: prefer_sharp_views(cm) for c, cm in pm.items()}
            pp.report["sharp_views"] = sel
            log.info(f"[{pp.name}] sharp views: " + ", ".join(f"{c} {v['kept']}/{v['views']}" for c, v in sel.items()))
        pass_maps.append(pm)
        frames_all.update(pp.frames)
        cams_all.update(pp.cams)
        Tcad_all.update(pp.T_cad_cam)
        sharp_all.update(pp.sharpness)
        info_all.update(pp.info)
        pdir = out / "passes" / pp.name
        pdir.mkdir(parents=True, exist_ok=True)
        if getattr(pp, "_poses", None) is not None:
            write_json(pdir / "poses.json", pp._poses.to_json())
        if getattr(pp, "_reg", None) is not None:
            write_json(pdir / "registration.json", pp._reg.to_json())
    modes = sorted({pp.pose_mode for pp in prepared})
    report = {"device": device, "device_name": model.spec.name, "passes": [pp.report for pp in prepared], "timing_s": {},
              "processing": {"pose_mode": modes[0] if len(modes) == 1 else "mixed", "anchor": anchor, **(meta or {})}}
    t0 = time.time()
    stages.at("detect")
    maps = merge_maps(pass_maps)
    for cm in maps.values():
        compute_stats(cm)
    cov = coverage(maps, model, geoms)
    advice = coverage_advice(maps, model, cov)
    regions, scores = detect_all(maps, model, detect_cfg)
    regions = merge_regions(regions, maps)
    if anchor == "template":
        regions, dropped = drop_mirror_symmetric(regions, maps)
        report["suppressed_symmetric"] = dropped
        regions, dropped = drop_border_lines(regions, maps, model)
        report["suppressed_border"] = dropped
        # the frame band of a plate fitted to a top-down video is seen at grazing angles, and the
        # buttons and ports of an unknown model are not masked: on 8 uCO3D phones (2026-11-16) it gave
        # 6-52 "chips" per phone, clean ones included. Not reported for plates.
        n_frame = sum(r.chart == "frame" for r in regions)
        regions = [r for r in regions if r.chart != "frame"]
        report["processing"]["frame_checked"] = False
        report["frame_regions_not_reported"] = n_frame
    S = np.concatenate([pp.outline_S for pp in prepared]) if prepared and anchor != "template" else np.zeros(0)
    R = np.concatenate([pp.outline_R for pp in prepared]) if prepared and anchor != "template" else np.zeros(0)
    prof = outline_profile(model, S, R) if len(S) else None
    dents = outline_dents(prof) if prof is not None else []
    if dents and "frame" in maps:
        regions = merge_dents_with_chips(regions, dents_to_regions(dents, maps["frame"], model), maps)
    report["outline"] = {"n_samples": int(len(S)), "sigma_mm": prof["sigma_mm"] if prof else None, "dents": dents}
    if prof is not None:
        write_json(out / "outline_profile.json", {k: v for k, v in prof.items()})
    report["timing_s"]["maps_detect"] = round(time.time() - t0, 1)
    # --- defects with evidence
    stages.at("evidence")
    defects, region_texels = [], {}
    bdir = out / "bundle"
    for r in sorted(regions, key=lambda r: -r.score):
        cm = maps[r.chart]
        vs = view_scores(cm, r.texels, np.eye(4), Tcad_all, sharp_all)
        if not vs:
            continue
        pers = persistence(vs)
        kind = r.kind
        if kind in ("crack", "scratch"):
            kind = "crack" if (pers >= 0.6 and r.extra.get("length_mm", 0) >= 3.0) else "scratch"
        did = len(defects) + 1
        g = cm.geom
        uv = g.uv_of(r.texels)
        u, v = uv.mean(0)
        part_ids, counts = np.unique(g.part[r.texels], return_counts=True)
        part = PARTS[int(part_ids[np.argmax(counts)])]
        cen = g.point[r.texels].mean(0)
        nrm = g.normal[r.texels].mean(0)
        nrm /= np.linalg.norm(nrm) + 1e-12
        conf = "high" if (r.score >= 2 * (detect_cfg or {}).get("frame_z", 8.0) and pers >= 0.5) else ("medium" if pers >= 0.3 else "low")
        ev = []
        for k, vsc in enumerate(pick_diverse(vs, 3)):
            img = frames_all[vsc.frame]
            px = project_region(cm, r.texels, np.eye(4), cams_all[vsc.frame], Tcad_all[vsc.frame])
            boxes = render_evidence(img, px, bdir / "evidence" / f"d{did}_{k + 1}_crop.jpg", bdir / "evidence" / f"d{did}_{k + 1}_full.jpg")
            inf = info_all[vsc.frame]
            ev.append({"pass": inf["pass"], "frame": inf["frame"], "time_s": round(inf["time_s"], 2),
                       "crop": f"evidence/d{did}_{k + 1}_crop.jpg", "full": f"evidence/d{did}_{k + 1}_full.jpg",
                       "view_cos": round(vsc.cos, 3), "visible": round(vsc.visible, 3), "contrast": round(vsc.contrast, 2),
                       "score": round(vsc.score, 4), **boxes})
        channels = {"appearance": round(float(r.extra.get("appearance_score", r.score if kind != "dent" else 0.0)), 2)}
        if "outline_peak_mm" in r.extra:
            channels["outline_mm"] = round(float(r.extra["outline_peak_mm"]), 3)
        defects.append({
            "id": did, "type": kind, "label_ko": TYPE_KO.get(kind, kind), "part": part, "part_ko": PART_KO.get(part, part),
            "chart": r.chart, "score": round(r.score, 2), "confidence": conf, "persistence": round(pers, 3), "channels": channels,
            "n_views": len(vs), "centroid_mm": cen, "normal": nrm,
            "unwrap": {"chart": r.chart, "u": float(u), "v": float(v), "bbox": [float(x) for x in (*uv.min(0), *uv.max(0))]},
            "size": {"area_mm2": round(r.extra.get("area_mm2", 0.0), 2), "length_mm": round(r.extra.get("length_mm", 0.0), 2)},
            "location_ko": describe_location(model, r.chart, float(u), float(v)), "evidence": ev})
        region_texels[did] = r.texels
    # --- bundle
    stages.at("bundle")
    atlas = B.build_atlas(maps, model, defects, region_texels)
    B.save_jpg(atlas["texture"], bdir / "texture.jpg", 92)
    B.save_png(atlas["overlay"], bdir / "defects.png")
    B.save_png(atlas["idmap"], bdir / "idmap.png")
    B.write_glb(model, bdir / "model.glb")
    stages.at("bundle", None, 0.3)
    spin = write_spin(bdir, maps, defects, region_texels, frames_all, cams_all, Tcad_all, info_all)
    stages.at("bundle", None, 0.6)
    videos = {}
    for pp in prepared:
        if pp.video is None or not Path(pp.video).exists():
            continue
        dst = bdir / f"video_{pp.name}.mp4"
        try:
            B.transcode_for_browser(pp.video, dst)
            videos[pp.name] = dst.name
        except Exception as e:  # pragma: no cover
            log.warning(f"video {pp.name}: {e}")
    payload = {"device": {"slug": device, "name": model.spec.name, "length_mm": model.spec.length, "width_mm": model.spec.width,
                          "thickness_mm": model.spec.thickness},
               "capture": {"passes": [{k: rp.get(k) for k in ("name", "face_up", "n_frames", "n_keyframes", "n_board_poses", "pose_mode")}
                                      for rp in report["passes"]],
                           "videos": videos},
               "processing": report["processing"], "coverage": cov, "advice": advice, "defects": defects, "spin": spin,
               "files": {"model": "model.glb", "texture": "texture.jpg", "overlay": "defects.png", "idmap": "idmap.png"},
               "atlas": {"width": B.ATLAS_W, "height": B.ATLAS_H}}
    write_json(bdir / "defects.json", payload)
    stages.done()
    report["coverage"] = cov
    report["advice"] = advice

    report["defects"] = [{k: d[k] for k in ("id", "type", "part", "score", "confidence", "persistence", "unwrap", "size", "location_ko")} for d in defects]
    report["timing_s"].update({f"stage_{k}": v for k, v in stages.timing.items()})
    report["timing_s"]["total"] = round(time.time() - t_start, 1)
    write_json(out / "run.json", report)
    log.info(f"{len(defects)} defects; coverage {cov['overall']:.1%}; total {report['timing_s']['total']:.0f}s -> {bdir}")
    return report


def write_spin(bdir: Path, maps, defects, region_texels, frames, cams, Tcad, info, width: int = 1280, per_band: int = 60) -> dict:
    """Keyframes ordered by viewing angle around the device, with projected defect outlines."""
    out = {"passes": []}
    by_pass: dict[str, list[int]] = {}
    for gid, inf in info.items():
        by_pass.setdefault(inf["pass"], []).append(gid)
    for pname, gids in by_pass.items():
        rows = []
        for gid in gids:
            c = Tcad[gid][:3, 3]
            az = float(np.degrees(np.arctan2(c[1], c[0])) % 360)
            el = float(np.degrees(np.arctan2(abs(c[2]), np.hypot(c[0], c[1]))))
            rows.append((gid, az, el))
        bands = [(0, 35), (35, 60), (60, 91)]
        pdata = {"name": pname, "bands": []}
        for lo, hi in bands:
            sel = sorted([r for r in rows if lo <= r[2] < hi], key=lambda r: r[1])
            if not sel:
                continue
            if len(sel) > per_band:
                sel = [sel[int(k)] for k in np.linspace(0, len(sel) - 1, per_band).round()]
            band = {"elev": [lo, hi], "frames": []}
            for gid, az, el in sel:
                img = frames[gid]
                s = width / img.shape[1]
                small = cv2.resize(img, (width, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
                name = f"spin/{pname}_{info[gid]['frame']:06d}.jpg"
                B.save_jpg(small, bdir / name, 85)
                outlines = []
                for d in defects:
                    cm = maps[d["chart"]]
                    tex = region_texels[d["id"]]
                    vi = cm.samples.frames.index(gid) if gid in cm.samples.frames else None
                    if vi is None or (cm.samples.cos[vi, tex].astype(np.float32) > 0.2).mean() < 0.3:
                        continue
                    px = project_region(cm, tex, np.eye(4), cams[gid], Tcad[gid])
                    polys = region_outline(px, img.shape[:2], dilate=4)
                    outlines.append({"id": d["id"], "polys": [(p * s).round(1).tolist() for p in polys]})
                band["frames"].append({"img": name, "az": round(az, 1), "el": round(el, 1), "frame": info[gid]["frame"],
                                       "time_s": round(info[gid]["time_s"], 2), "outlines": outlines})
            pdata["bands"].append(band)
        out["passes"].append(pdata)
    return out
