"""Synthetic captures in the Stray Scanner format, with ground truth.

Until the team's real captures arrive, the whole pipeline is developed and tested on renders of
the reference model lying on the printed board, written exactly like a Stray Scanner recording
(``rgb.mp4``, ``odometry.csv`` with ARKit-convention poses in metres and per-frame intrinsics,
``camera_matrix.csv``, 256x192 ``depth/`` in mm, ``confidence/``) plus ``gt.json`` with the true
board-frame camera poses, the true device placement and the injected defects. The ARKit poses get
a slow drift, as real ARKit does over a full circle, so the board-based poses have to be used.

The shading is deliberately simple but keeps the two things that matter for the detectors:
glossy parts reflect a bright "ceiling" that moves with the viewpoint (glare that is not a defect),
and defects are painted in the device's unwrapped surface coordinates so they stay fixed on the
surface in every view. Defects:

* ``chip``   - paint chip on the frame: an irregular silver blob (stands in for 까짐 and the coating
  break of a 찍힘)
* ``crack``  - branching bright lines on glass
* ``scratch``- a faint thin line on glass that is only visible from some angles
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..anchor.parametric import CHARTS, PARTS, DeviceModel, DeviceSpec, build_device_model, rest_pose
from ..geometry.camera import Camera
from ..geometry.transforms import inv_T, look_at_opencv, make_T, rot_z, se3_exp
from ..poses.board import BoardSpec, render_board_page
from ..surface.raycast import RaycastScene
from ..utils.io import write_json


@dataclass
class Decal:
    kind: str                  # chip | crack | scratch
    chart: str                 # screen | back | frame
    u: float                   # chart coordinates (mm)
    v: float
    size: float                # chip diameter / crack or scratch length (mm)
    angle: float = 0.0         # degrees, for cracks and scratches
    seed: int = 0


@dataclass
class SynthConfig:
    device: str = "iphone-12"
    face_up: str = "back"
    width: int = 1920
    height: int = 1440
    f_px: float = 1460.0
    rings: list = field(default_factory=lambda: [(20.0, 60), (45.0, 50), (70.0, 40)])   # (elevation deg, frames)
    top_frames: int = 16
    radius_mm: float = 300.0
    placement_jitter: tuple = (1.5, 1.5, 1.5)       # dx, dy (mm), yaw (deg) of the device on the outline
    arkit_drift_mm: float = 8.0                      # end-to-end ARKit translation drift
    arkit_drift_deg: float = 0.6
    noise_sigma: float = 2.0
    blur_sigma: float = 0.6
    board_px_per_mm: int = 8
    texture_mm_per_px: float = 0.05
    decals: list = field(default_factory=list)
    dents: list = field(default_factory=list)        # [{s, v, radius, depth}] frame-chart mm: geometric dents
    seed: int = 0
    fps: float = 30.0
    background: str = "board"                        # board | wood | paper | cloth (no board: an ordinary floor)


# ----------------------------------------------------------------------------- appearance
def _chart_canvas(model: DeviceModel, chart: str, res: float) -> tuple[np.ndarray, np.ndarray]:
    w, h = model.chart_size(chart)
    W, H = int(np.ceil(w / res)) + 1, int(np.ceil(h / res)) + 1
    return np.zeros((H, W), np.float32), np.zeros((H, W), np.int16)


def _paint_decal(alpha: np.ndarray, ids: np.ndarray, d: Decal, did: int, res: float, rng: np.random.Generator) -> None:
    cu, cv = d.u / res, d.v / res
    if d.kind == "chip":
        n = 24
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rad = (d.size / 2 / res) * (0.7 + 0.5 * rng.random(n))
        pts = np.stack([cu + rad * np.cos(ang), cv + rad * np.sin(ang)], 1)
        mask = np.zeros_like(alpha, np.uint8)
        cv2.fillPoly(mask, [np.round(pts * 16).astype(np.int32)], 1, lineType=cv2.LINE_AA, shift=4)
        alpha[:] = np.maximum(alpha, mask.astype(np.float32))
        ids[mask > 0] = did
        return
    # cracks and scratches: polylines (cracks branch)
    width_mm = 0.25 if d.kind == "crack" else 0.08
    strength = 1.0 if d.kind == "crack" else 0.55
    segs = []

    def walk(x, y, ang, length, depth):
        step = 0.6
        n = max(int(length / step), 2)
        pts = [(x, y)]
        for _ in range(n):
            ang += rng.normal(0, 0.18 if d.kind == "crack" else 0.03)
            x, y = x + step * np.cos(ang), y + step * np.sin(ang)
            pts.append((x, y))
            if d.kind == "crack" and depth < 2 and rng.random() < 0.06:
                walk(x, y, ang + rng.choice([-1, 1]) * rng.uniform(0.5, 1.2), length * rng.uniform(0.2, 0.5), depth + 1)
        segs.append(np.array(pts))

    a = np.radians(d.angle)
    x0, y0 = d.u - 0.5 * d.size * np.cos(a), d.v - 0.5 * d.size * np.sin(a)
    walk(x0, y0, a, d.size, 0)
    mask = np.zeros(alpha.shape, np.uint8)
    thick = max(int(round(width_mm / res)), 1)
    for p in segs:
        q = np.round(p / res * 16).astype(np.int32)
        cv2.polylines(mask, [q], False, 255, thickness=thick, lineType=cv2.LINE_AA, shift=4)
    m = mask.astype(np.float32) / 255.0 * strength
    alpha[:] = np.maximum(alpha, m)
    ids[mask > 64] = did


def build_textures(model: DeviceModel, decals: list[Decal], res: float, seed: int):
    """Per chart: fine surface variation, defect alpha and defect ids (GT masks)."""
    rng = np.random.default_rng(seed)
    tex = {}
    for chart in CHARTS:
        alpha, ids = _chart_canvas(model, chart, res)
        var = cv2.GaussianBlur(rng.normal(0, 1, alpha.shape).astype(np.float32), (0, 0), 3.0)
        var = 1.0 + 0.03 * var / (var.std() + 1e-6)
        tex[chart] = {"var": var, "alpha": alpha, "ids": ids}
    for k, d in enumerate(decals):
        _paint_decal(tex[d.chart]["alpha"], tex[d.chart]["ids"], d, k + 1, res, np.random.default_rng(d.seed + 1000 * k))
    return tex


def _sample(img: np.ndarray, u_px: np.ndarray, v_px: np.ndarray) -> np.ndarray:
    """Bilinear lookup at continuous pixel coordinates (edge-clamped). cv2.remap is limited to
    32767 samples per row, so this is plain numpy."""
    h, w = img.shape[:2]
    x = np.clip(np.asarray(u_px, np.float64), 0, w - 1.0001)
    y = np.clip(np.asarray(v_px, np.float64), 0, h - 1.0001)
    x0, y0 = x.astype(np.int64), y.astype(np.int64)
    fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    if img.ndim == 3:
        fx, fy = fx[:, None], fy[:, None]
    a, b = img[y0, x0], img[y0, x0 + 1]
    c, d = img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


PART_SHADING = {   # ks (specular), shininess, env reflection strength
    "screen": (0.9, 200.0, 0.55), "back": (0.6, 120.0, 0.35), "frame": (0.5, 40.0, 0.25),
    "camera": (0.6, 120.0, 0.35), "lens": (0.9, 200.0, 0.5), "button": (0.5, 40.0, 0.25),
}


def _env(r: np.ndarray, lamps_only: bool = False) -> np.ndarray:
    """Bright ceiling with two long lamps, dark floor: what glossy parts reflect."""
    z = np.clip(r[:, 2], -1, 1)
    sky = 0.18 + 0.55 * np.clip(z, 0, 1) ** 1.5
    lamp = np.exp(-((r[:, 0] * 2.2 - 0.6) ** 2) / 0.02) * (z > 0.3) + np.exp(-((r[:, 0] * 2.2 + 0.9) ** 2) / 0.03) * (z > 0.4)
    if lamps_only:
        return lamp
    return np.clip(sky + 0.9 * lamp, 0, 1.5)


def apply_dents(model: DeviceModel, dents: list[dict]) -> DeviceModel:
    """Copy of the model with frame vertices pushed inwards around each dent (smooth (1-r^2)^2
    profile in frame-chart coordinates). Only the rendered device is dented; the pipeline keeps
    using the nominal model, so the dent shows up as an outline residual."""
    import copy

    from ..anchor.parametric import outward_normals

    if not dents:
        return model
    m = copy.deepcopy(model)
    frame = m.vchart == CHARTS.index("frame")
    P = m.perimeter
    nrm = outward_normals(m.outline_xy)
    s_v = m.vuv[:, 0]
    nx = np.interp(s_v, np.r_[m.outline_s, P], np.r_[nrm[:, 0], nrm[0, 0]])
    ny = np.interp(s_v, np.r_[m.outline_s, P], np.r_[nrm[:, 1], nrm[0, 1]])
    for d in dents:
        ds = (s_v - d["s"] + P / 2) % P - P / 2
        dv = m.vuv[:, 1] - d["v"]
        rv = d.get("radius_v", d["radius"])          # a drop dent crushes the whole band height
        r = np.sqrt((ds / d["radius"]) ** 2 + (dv / rv) ** 2)
        w = np.where(frame & (r < 1.0), (1 - r ** 2) ** 2, 0.0) * d["depth"]
        m.mesh.vertices[:, 0] -= w * nx
        m.mesh.vertices[:, 1] -= w * ny
    return m


# ----------------------------------------------------------------------------- trajectory and render
def floor_texture(kind: str, rng: np.random.Generator, half_mm: float = 600.0, px_per_mm: int = 2) -> np.ndarray:
    """An ordinary surface under the device instead of the printed board (float RGB 0..1, centred
    on the world origin, ``px_per_mm``): wood grain with plank joints, plain paper (almost no
    texture) or a woven cloth."""
    n = int(2 * half_mm * px_per_mm)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / px_per_mm

    def field(sigma_mm: float) -> np.ndarray:
        f = cv2.GaussianBlur(rng.normal(0, 1, (n, n)).astype(np.float32), (0, 0), sigma_mm * px_per_mm)
        return f / (f.std() + 1e-9)

    if kind == "wood":
        warp = 3.0 * field(25.0) + 0.6 * field(4.0)
        grain = 0.5 + 0.5 * np.sin(2 * np.pi * (yy + warp) / 3.2)
        tone = 1.0 + 0.10 * field(40.0) - 0.16 * grain ** 3 + 0.03 * rng.normal(0, 1, (n, n)).astype(np.float32)
        tone *= np.where((xx % 140.0) < 0.8, 0.55, 1.0)                     # plank joints
        img = tone[..., None] * np.array([0.47, 0.34, 0.22], np.float32)
    elif kind == "paper":
        tone = 1.0 + 0.012 * field(30.0) + 0.008 * rng.normal(0, 1, (n, n)).astype(np.float32)
        img = tone[..., None] * np.array([0.86, 0.85, 0.82], np.float32)
    elif kind == "cloth":
        weave = np.sin(2 * np.pi * xx / 1.3) * np.sin(2 * np.pi * yy / 1.3)
        tone = 1.0 + 0.06 * weave + 0.05 * field(15.0) + 0.02 * rng.normal(0, 1, (n, n)).astype(np.float32)
        img = tone[..., None] * np.array([0.55, 0.57, 0.60], np.float32)
    else:
        raise ValueError(f"unknown background {kind!r}")
    return np.clip(img, 0, 1)


def trajectory(cfg: SynthConfig, rng: np.random.Generator) -> list[np.ndarray]:
    poses = []
    for elev, n in cfg.rings:
        az0 = rng.uniform(0, 360)
        for i in range(n):
            az = np.radians(az0 + 360.0 * i / n + rng.normal(0, 1.0))
            e = np.radians(elev + rng.normal(0, 1.5))
            r = cfg.radius_mm * rng.uniform(0.92, 1.08)
            eye = r * np.array([np.cos(e) * np.cos(az), np.cos(e) * np.sin(az), np.sin(e)])
            target = np.array([rng.normal(0, 6), rng.normal(0, 10), 2.0])
            T = look_at_opencv(eye, target)
            T = T @ make_T(rot_z(rng.normal(0, 4.0)))             # hand roll
            poses.append(T)
    for i in range(cfg.top_frames):                               # top sweep along the device
        y = -60 + 120 * i / max(cfg.top_frames - 1, 1)
        eye = np.array([rng.normal(0, 15), y * 0.6, cfg.radius_mm * 0.95])
        T = look_at_opencv(eye, np.array([0.0, y * 0.4, 0.0]), up=np.array([0.0, 1.0, 0.0]))
        poses.append(T)
    return poses


def render_frame(model: DeviceModel, scene: RaycastScene, tex: dict, colors: dict, board_img: np.ndarray,
                 board_k: float, cam: Camera, T_wc: np.ndarray, cfg: SynthConfig, rng: np.random.Generator,
                 light_dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB image (uint8), depth (mm, camera z) and defect-id image of one view."""
    h = scene.cast(cam, T_wc, step=1)
    H, W = h.face.shape
    us, vs = cam.pixel_grid(1)
    d_cam = cam.ray_dirs_cam(us.ravel(), vs.ravel())
    d_w = d_cam / np.linalg.norm(d_cam, axis=1, keepdims=True) @ T_wc[:3, :3].T
    o = T_wc[:3, 3]
    img = np.empty((H * W, 3), np.float32)
    img[:] = np.array([0.55, 0.55, 0.57], np.float32)          # wall behind the table
    depth = h.depth.ravel().astype(np.float32).copy()
    did = np.zeros(H * W, np.int16)
    hit = h.hit.ravel()
    # --- board plane z = 0 and the table around it
    t_plane = -o[2] / np.where(np.abs(d_w[:, 2]) > 1e-9, d_w[:, 2], -1e-9)
    on_plane = (~hit) & (t_plane > 0)
    P = o + d_w * t_plane[:, None]
    bh, bw = board_img.shape[:2]
    pu = bw / 2.0 + P[:, 0] * board_k
    pv = bh / 2.0 - P[:, 1] * board_k
    inside = on_plane & (pu >= 0) & (pu < bw - 1) & (pv >= 0) & (pv < bh - 1)
    table = np.array([0.42, 0.36, 0.30], np.float32)
    img[on_plane] = table
    if inside.any():
        img[inside] = _sample(board_img, pu[inside], pv[inside]) * 0.92
    depth[on_plane] = (t_plane[on_plane] * (d_cam[on_plane, 2] / np.linalg.norm(d_cam[on_plane], axis=1))).astype(np.float32)
    # --- device
    if hit.any():
        face = h.face.ravel()[hit]
        part_ids = model.mesh.face_label[face]
        chart, uv = model.surface_coords(face, h.bary.reshape(-1, 2)[hit])
        n_w = scene.face_normals_obj[face] @ scene.T_world_obj[:3, :3].T
        v = -d_w[hit]
        n_w = np.where((np.einsum("ij,ij->i", n_w, v) < 0)[:, None], -n_w, n_w)
        base = np.zeros((len(face), 3), np.float32)
        ks = np.zeros(len(face), np.float32)
        shin = np.zeros(len(face), np.float32)
        env_k = np.zeros(len(face), np.float32)
        for pi, pname in enumerate(PARTS):
            sel = part_ids == pi
            if sel.any():
                base[sel] = np.asarray(colors.get(pname, [128, 128, 128]), np.float32) / 255.0
                ks[sel], shin[sel], env_k[sel] = PART_SHADING[pname]
        var = np.ones(len(face), np.float32)
        alpha = np.zeros(len(face), np.float32)
        dids = np.zeros(len(face), np.int16)
        res = cfg.texture_mm_per_px
        for ci, cname in enumerate(CHARTS):
            sel = chart == ci
            if not sel.any():
                continue
            t = tex[cname]
            px, py = uv[sel, 0] / res, uv[sel, 1] / res
            var[sel] = _sample(t["var"], px, py)
            alpha[sel] = _sample(t["alpha"], px, py)
            ix = np.clip(np.round(px).astype(int), 0, t["ids"].shape[1] - 1)
            iy = np.clip(np.round(py).astype(int), 0, t["ids"].shape[0] - 1)
            dids[sel] = t["ids"][iy, ix]
        col = base * var[:, None]
        diffuse = np.zeros(len(face), np.float32)
        spec = np.zeros(len(face), np.float32)
        for L in light_dirs:
            diffuse += np.clip(n_w @ L, 0, None) * 0.55
            hv = L + v
            hv /= np.linalg.norm(hv, axis=1, keepdims=True)
            spec += np.clip(np.einsum("ij,ij->i", n_w, hv), 0, None) ** shin
        r = 2 * np.einsum("ij,ij->i", n_w, v)[:, None] * n_w - v
        refl = _env(r)
        shade = col * (0.35 + diffuse)[:, None] + (ks * spec * 0.6)[:, None] + (env_k * refl)[:, None] * 0.6
        # defects: silver chip (glossy, lighter than any paint), bright cracks; scratches fade with angle
        kind = np.array([cfg.decals[i - 1].kind if i > 0 else "" for i in range(len(cfg.decals) + 1)])
        dk = kind[np.clip(dids, 0, len(kind) - 1)] if len(cfg.decals) else np.full(len(face), "")
        chip_col = np.array([0.78, 0.78, 0.80], np.float32) * (0.75 + 0.5 * diffuse)[:, None] + (0.8 * spec)[:, None] * 0.4
        line_col = np.array([0.85, 0.87, 0.90], np.float32) * (0.8 + 0.3 * refl)[:, None]
        a = alpha.copy()
        is_scratch = dk == "scratch"
        # a fine scratch only lights up when it catches a lamp's reflection (and faintly otherwise)
        if is_scratch.any():
            lamp = _env(r[is_scratch], lamps_only=True)
            a[is_scratch] *= np.clip(0.08 + 3.0 * lamp, 0, 1)
        is_chip = dk == "chip"
        shade = np.where(is_chip[:, None], shade * (1 - a[:, None]) + chip_col * a[:, None], shade * (1 - a[:, None]) + line_col * a[:, None])
        img[hit] = shade
        did[hit] = dids
    img = np.clip(img.reshape(H, W, 3), 0, 1)
    if cfg.blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), cfg.blur_sigma)
    img = img * 255.0 + rng.normal(0, cfg.noise_sigma, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), depth.reshape(H, W), did.reshape(H, W)


# ----------------------------------------------------------------------------- ARKit convention
T_ARK_WORLD_BASE = make_T(np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]))   # board z-up -> ARKit y-up
CV_TO_ARKIT_CAM = np.diag([1.0, -1.0, -1.0, 1.0])


def write_stray(out: Path, cam: Camera, frames: list[np.ndarray], depths: list[np.ndarray], T_ark_cam: list[np.ndarray],
                fps: float) -> None:
    """Write a Stray Scanner-format folder (H.264 instead of HEVC: the bundled ffmpeg encodes H.264)."""
    import imageio.v2 as iio
    from scipy.spatial.transform import Rotation

    out.mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(exist_ok=True)
    (out / "confidence").mkdir(exist_ok=True)
    w = iio.get_writer(str(out / "rgb.mp4"), fps=fps, codec="libx264", quality=None, pixelformat="yuv420p",
                       output_params=["-crf", "14", "-preset", "medium"], macro_block_size=8)
    for f in frames:
        w.append_data(f)
    w.close()
    rows = ["timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y"]
    for i, (T, dmap) in enumerate(zip(T_ark_cam, depths)):
        q = Rotation.from_matrix(T[:3, :3]).as_quat()       # x, y, z, w
        x, y, z = T[:3, 3] / 1000.0
        rows.append(f"{i / fps:.6f}, {i:06d}, {x:.6f}, {y:.6f}, {z:.6f}, {q[0]:.8f}, {q[1]:.8f}, {q[2]:.8f}, {q[3]:.8f}, "
                    f"{cam.fx:.4f}, {cam.fy:.4f}, {cam.cx:.4f}, {cam.cy:.4f}, , ")
        small = cv2.resize(dmap, (256, 192), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out / "depth" / f"{i:06d}.png"), np.clip(small, 0, 65535).astype(np.uint16))
        cv2.imwrite(str(out / "confidence" / f"{i:06d}.png"), np.full((192, 256), 2, np.uint8))
    (out / "odometry.csv").write_text("\n".join(rows) + "\n")
    np.savetxt(out / "camera_matrix.csv", cam.K, delimiter=",", fmt="%.6f")
    (out / "imu.csv").write_text("timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z\n")


def render_capture(cfg: SynthConfig, out: str | Path, progress: bool = True) -> Path:
    out = Path(out)
    rng = np.random.default_rng(cfg.seed)
    spec = DeviceSpec.by_slug(cfg.device)
    model = build_device_model(spec)
    decals = [Decal(**d) if isinstance(d, dict) else d for d in cfg.decals]
    cfg.decals = decals
    dx, dy, dyaw = (rng.normal(0, s) for s in cfg.placement_jitter)
    T_world_cad = make_T(rot_z(dyaw), [dx, dy, 0.0]) @ rest_pose(model, cfg.face_up)
    model = apply_dents(model, cfg.dents)
    scene = RaycastScene(model.mesh, T_world_cad)
    tex = build_textures(model, decals, cfg.texture_mm_per_px, cfg.seed)
    bspec = BoardSpec.load()
    if cfg.background == "board":
        bspec.px_per_mm = cfg.board_px_per_mm
        page = render_board_page(bspec, model.outline_xy, spec.name)
        board_img = np.asarray(page).astype(np.float32) / 255.0
        board_img = cv2.GaussianBlur(board_img, (0, 0), 0.6)
        board_k = bspec.px_per_mm
    else:
        board_k = 2
        board_img = floor_texture(cfg.background, np.random.default_rng(cfg.seed + 101), px_per_mm=board_k) / 0.92
    cam = Camera(np.array([[cfg.f_px, 0, (cfg.width - 1) / 2.0], [0, cfg.f_px, (cfg.height - 1) / 2.0], [0, 0, 1]]), cfg.width, cfg.height)
    light_dirs = np.array([[0.3, 0.4, 0.86], [-0.5, -0.2, 0.84]])
    light_dirs /= np.linalg.norm(light_dirs, axis=1, keepdims=True)
    poses = trajectory(cfg, rng)
    yaw0 = rng.uniform(0, 360)
    T_ark_world = make_T(None, rng.normal(0, 200, 3)) @ T_ARK_WORLD_BASE @ make_T(rot_z(yaw0))
    frames, depths, ark, gt_frames = [], [], [], []
    n = len(poses)
    for i, T_wc in enumerate(poses):
        img, dep, did = render_frame(model, scene, tex, spec.default_color, board_img, board_k, cam, T_wc, cfg, rng, light_dirs)
        frames.append(img)
        depths.append(dep)
        frac = i / max(n - 1, 1)
        drift = se3_exp(np.concatenate([np.radians(cfg.arkit_drift_deg) * frac * np.array([0.3, 0.2, 1.0]) / 1.06,
                                        cfg.arkit_drift_mm * frac * np.array([0.6, -0.5, 0.62])]))
        ark.append(T_ark_world @ drift @ T_wc @ CV_TO_ARKIT_CAM)
        vis = {int(k): int(c) for k, c in zip(*np.unique(did[did > 0], return_counts=True))}
        gt_frames.append({"frame": i, "T_world_cam": T_wc, "defect_pixels": vis})
        if progress and (i % 20 == 0 or i == n - 1):
            print(f"  rendered {i + 1}/{n}", flush=True)
    write_stray(out, cam, frames, depths, ark, cfg.fps)
    gt = {"config": {k: v for k, v in asdict(cfg).items() if k != "decals"}, "device": cfg.device, "face_up": cfg.face_up,
          "T_world_cad": T_world_cad, "decals": [asdict(d) for d in decals], "dents": list(cfg.dents), "frames": gt_frames,
          "board_square_mm": bspec.square_mm if cfg.background == "board" else None, "background": cfg.background,
          "note": "synthetic capture; T_world_* are in the board (floor) frame (mm)"}
    write_json(out / "gt.json", gt)
    return out


DEMO_DENTS = [
    {"s": 160.0, "v": 3.7, "radius": 3.0, "radius_v": 9.0, "depth": 0.8},   # near the bottom-right corner, crushing the band
]

DEMO_DECALS = [
    {"kind": "chip", "chart": "frame", "u": 30.0, "v": 3.7, "size": 1.6, "seed": 1},      # top edge, right of centre
    {"kind": "chip", "chart": "frame", "u": 120.0, "v": 3.0, "size": 1.0, "seed": 2},     # right side
    {"kind": "chip", "chart": "frame", "u": 250.0, "v": 4.5, "size": 2.2, "seed": 3},     # bottom-left area
    {"kind": "crack", "chart": "back", "u": 22.0, "v": 95.0, "size": 18.0, "angle": 35.0, "seed": 4},
    {"kind": "scratch", "chart": "back", "u": 50.0, "v": 60.0, "size": 12.0, "angle": -20.0, "seed": 5},
    {"kind": "crack", "chart": "screen", "u": 48.0, "v": 30.0, "size": 25.0, "angle": 120.0, "seed": 6},
    {"kind": "chip", "chart": "frame", "u": 160.0, "v": 3.7, "size": 1.4, "seed": 7},      # the dent also breaks the coating
]


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m flawmap.synth.render_capture")
    p.add_argument("out")
    p.add_argument("--device", default="iphone-12")
    p.add_argument("--face-up", default="back", choices=["screen", "back"])
    p.add_argument("--no-defects", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--small", action="store_true", help="fewer frames (quick tests); same as --preset small")
    p.add_argument("--preset", choices=["demo", "small", "small-v1", "dents"], default=None,
                   help="demo: 166 frames, 7 defects + dent (default); small: 32 frames, same defects; small-v1: 32 frames, "
                        "the first 5 defects, no dent (iphone12_back_small); dents: 50 frames, two dents only (iphone12_dent_small)")
    p.add_argument("--background", default="board", choices=["board", "wood", "paper", "cloth"],
                   help="what the device lies on: the printed board, or an ordinary floor without any board")
    p.add_argument("--arkit-drift-mm", type=float, default=8.0, help="end-to-end ARKit translation drift")
    p.add_argument("--arkit-drift-deg", type=float, default=0.6)
    p.add_argument("--placement-jitter", type=float, nargs=3, default=None, metavar=("DX", "DY", "YAW"),
                   help="device placement noise (mm, mm, deg); default 1.5 1.5 1.5 on the board, 15 15 180 without it (any heading)")
    a = p.parse_args(argv)
    cfg = SynthConfig(device=a.device, face_up=a.face_up, seed=a.seed, decals=[] if a.no_defects else list(DEMO_DECALS),
                      dents=[] if a.no_defects else list(DEMO_DENTS), background=a.background,
                      arkit_drift_mm=a.arkit_drift_mm, arkit_drift_deg=a.arkit_drift_deg)
    if a.placement_jitter is not None:
        cfg.placement_jitter = tuple(a.placement_jitter)
    elif a.background != "board":
        cfg.placement_jitter = (15.0, 15.0, 180.0)
    preset = a.preset or ("small" if a.small else "demo")
    if preset in ("small", "small-v1"):
        cfg.rings, cfg.top_frames = [(22.0, 16), (50.0, 12)], 4
    if preset == "small-v1" and not a.no_defects:
        cfg.decals, cfg.dents = list(DEMO_DECALS[:5]), []
    if preset == "dents":
        cfg.rings, cfg.top_frames = [(22.0, 20), (50.0, 14), (72.0, 10)], 6
        cfg.decals = []
        cfg.dents = [] if a.no_defects else [{"s": 160.0, "v": 3.7, "radius": 3.0, "radius_v": 9.0, "depth": 0.8},
                                             {"s": 300.0, "v": 3.7, "radius": 2.0, "radius_v": 9.0, "depth": 0.4}]
    render_capture(cfg, a.out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
