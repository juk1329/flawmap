"""Camera poses without the printed board: the ARKit trajectory, corrected by the device's own outline.

ARKit poses are metric (mm) but drift by millimetres to centimetres over a circle. The board used to
give drift-free poses. Without it the reference model takes the board's place: its outline is known
in 3D, so once the device is found, every keyframe's camera can be refined against the device
outline (model-based tracking) with ARKit as the motion prior.

1. **World frame:** ARKit's world (y up, gravity) turned z-up, in mm.
2. **Where is the device:** the point where the keyframes' optical axes meet (the seller keeps the
   device in the middle of the picture); the floor height from the LiDAR depth when there is any.
3. **Which way does it lie:** grid search over heading and position. The model's upper and lower
   outline loops are projected (a visible side face means the lower loop forms the silhouette,
   otherwise the upper one) and scored by the truncated distance to image edges, coarse then fine.
   The 180-degree twin has the same outline; full registration (camera bump, buttons) decides.
4. **Joint fit** of the device pose and the ARKit drift over all keyframes: the device lies flat on
   the floor (gravity from ARKit, height from LiDAR), so only its heading and position are free;
   the drift is a smooth function of time (6-DoF knots, linear in between, the first knot fixed).
   Edge-alignment rounds as in ``register.register_device`` with shrinking search windows.

Why not refine each frame's camera on its own (the first version, 2026-11-16): on a wood floor the
dark frame side hardly differs from the floor, the strongest edge is the inner one between back
glass and frame, and a free 6-DoF camera per frame happily tilts to put the outline there (cameras
6-8 mm off, 20 px model reprojection on the synthetic wood capture). With the device on the floor
and a smooth drift, a wrong edge in single views is outvoted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ..anchor.parametric import _resample_closed, outward_normals, rest_pose
from ..anchor.register import _grad, _project, mask_fields, mask_weight_for, match_edges, register_device, silhouette_points
from ..geometry.camera import Camera
from ..geometry.mesh_io import LabeledMesh
from ..geometry.transforms import inv_T, make_T, rot_z, se3_exp
from ..surface.raycast import RaycastScene
from .align import CapturePoses

T_W_ARK = make_T(np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]))     # ARKit y-up -> z-up world


@dataclass
class BoardlessResult:
    poses: CapturePoses
    T_world_cad: np.ndarray
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------- where the device is
def axes_meeting_point(Ts: list[np.ndarray]) -> np.ndarray:
    """Least-squares point closest to the optical axes (OpenCV cameras: +z looks forward)."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for T in Ts:
        c, z = T[:3, 3], T[:3, 2] / np.linalg.norm(T[:3, 2])
        P = np.eye(3) - np.outer(z, z)
        A += P
        b += P @ c
    return np.linalg.lstsq(A, b, rcond=None)[0]


def floor_height(cap, ids: list[int], T_world_cam: dict[int, np.ndarray], centre: np.ndarray, radius: float = 150.0,
                 min_conf: int = 2) -> float | None:
    """Height of the surface the device lies on: mode of LiDAR point heights near the device."""
    zs = []
    for i in ids:
        d = cap.depth(i)
        if d is None:
            continue
        conf = cap.confidence(i)
        h, w = d.shape[:2]
        K = cap.K[i]
        sx, sy = w / cap.width, h / cap.height
        fx, fy = K[0, 0] * sx, K[1, 1] * sy
        cx, cy = (K[0, 2] + 0.5) * sx - 0.5, (K[1, 2] + 0.5) * sy - 0.5
        vs, us = np.mgrid[0:h, 0:w]
        z = d.astype(np.float32)
        ok = z > 0
        if conf is not None and conf.shape == d.shape:
            ok &= conf >= min_conf
        pc = np.stack([(us - cx) / fx * z, (vs - cy) / fy * z, z], -1)[ok]
        T = T_world_cam[i]
        pw = pc @ T[:3, :3].T + T[:3, 3]
        near = np.hypot(pw[:, 0] - centre[0], pw[:, 1] - centre[1]) < radius
        zs.append(pw[near, 2])
    if not zs or sum(len(z) for z in zs) < 500:
        return None
    z = np.concatenate(zs)
    # the axes' meeting point can land far below the floor when they converge weakly (an iPad filmed
    # from 40 cm: 80 mm below; filmed from above: 270 mm below, 2026-11-17). The floor lies below
    # every camera, so search from 1 m under the lowest camera up to just under it; the mode wins
    cam_z = min(float(T_world_cam[i][2, 3]) for i in ids)
    lo, hi = min(centre[2] - 200.0, cam_z - 1000.0), max(centre[2] + 200.0, cam_z - 30.0)
    z = z[(z > lo) & (z < hi)]
    if len(z) < 500:
        return None
    hist, edges = np.histogram(z, bins=np.arange(lo, hi + 1.0, 1.0))
    hist = np.convolve(hist, [1, 2, 3, 2, 1], mode="same")
    m = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
    return float(np.median(z[np.abs(z - m) < 3.0]))


# ---------------------------------------------------------------------------- which way it lies
def _edge_distance(img: np.ndarray, scale: float, trunc: float) -> tuple[np.ndarray, np.ndarray]:
    """Truncated distance to the nearest strong edge and that edge's gradient direction (radians)."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    hi = float(np.percentile(mag, 97.0))           # strong edges only: the device outline, not floor grain
    e = cv2.Canny(g, max(hi * 0.4, 10.0), max(hi, 25.0), L2gradient=True)
    dist, lab = cv2.distanceTransformWithLabels(255 - e, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    theta = np.arctan2(gy, gx)
    ang_of_label = np.concatenate([[0.0], theta[e > 0]]).astype(np.float32)    # labels follow the raster order of edge pixels
    return np.minimum(dist, trunc).astype(np.float32), ang_of_label[np.clip(lab, 0, len(ang_of_label) - 1)]


class _Loops:
    """The model's upper and lower plan-outline loops in the rest pose (device centred over the
    origin, lying on z = 0), with the side-face normals."""

    def __init__(self, model, face_up: str, n: int = 120):
        R0 = rest_pose(model, face_up)
        xy = _resample_closed(model.outline_xy, n)
        nrm = outward_normals(xy)
        a = np.column_stack([xy, np.full(len(xy), model.z_front)])
        b = np.column_stack([xy, np.full(len(xy), model.z_back)])
        pa, pb = a @ R0[:3, :3].T + R0[:3, 3], b @ R0[:3, :3].T + R0[:3, 3]
        self.up, self.lo = (pa, pb) if pa[:, 2].mean() > pb[:, 2].mean() else (pb, pa)
        self.nrm = np.column_stack([nrm, np.zeros(len(nrm))]) @ R0[:3, :3].T
        self.R0 = R0


def _score(loops: _Loops, yaw_deg: np.ndarray, txy: np.ndarray, z0: float, views: list, centre_xy: np.ndarray,
           trunc: float, prior_px: float = 1.0, prior_mm: float = 25.0) -> np.ndarray:
    """Cost of every (yaw, translation): mean oriented chamfer distance of the projected outline
    (distance to the nearest strong edge, plus a penalty when that edge does not run along the
    outline), plus a mild prior towards the point the cameras aimed at. Plain chamfer distance is
    fooled by floors with edges everywhere (the printed board itself, 2026-11-16)."""
    out = np.zeros((len(yaw_deg), len(txy)), np.float64)
    prior = prior_px * (np.sum((txy - centre_xy) ** 2, axis=1) / prior_mm ** 2)
    for k, yaw in enumerate(yaw_deg):
        R = rot_z(yaw)
        up, lo, nr = loops.up @ R.T, loops.lo @ R.T, loops.nrm @ R.T
        mid = 0.5 * (up + lo)
        t = np.column_stack([txy, np.full(len(txy), z0)])                     # (T,3)
        acc = np.zeros(len(txy))
        for cam, Tcw, (dt, ang) in views:
            C = -Tcw[:3, :3].T @ Tcw[:3, 3]
            vis = np.einsum("nj,tnj->tn", nr, C[None, None, :] - (mid[None] + t[:, None, :])) > 0
            P = np.where(vis[..., None], lo[None], up[None]) + t[:, None, :]        # (T,N,3)
            uv = []
            for Q in (P, P + nr[None]):                                           # the point and 1 mm outwards
                pc = Q @ Tcw[:3, :3].T + Tcw[:3, 3]
                z = np.maximum(pc[..., 2], 1.0)
                uv.append((cam.fx * pc[..., 0] / z + cam.cx, cam.fy * pc[..., 1] / z + cam.cy))
            (u, v), (u2, v2) = uv
            phi = np.arctan2(v2 - v, u2 - u)                                      # outline normal in the image
            h, w = dt.shape
            ui, vi = np.clip(np.round(u).astype(int), 0, w - 1), np.clip(np.round(v).astype(int), 0, h - 1)
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            mis = 1.0 - np.abs(np.cos(ang[vi, ui] - phi))
            d = np.where(inside, np.minimum(dt[vi, ui] + 0.6 * trunc * mis, trunc), trunc)
            acc += d.mean(1)
        out[k] = acc / len(views) + prior
    return out


def find_device(model, face_up: str, frames: dict[int, np.ndarray], cams: dict[int, Camera], T_world_cam: dict[int, np.ndarray],
                centre: np.ndarray, z_floor: float, ids: list[int], radius: float = 32.0, log=print) -> list[np.ndarray]:
    """Candidate ``T_world_cad`` from a coarse-to-fine search, best first (both 180-degree twins)."""
    loops = _Loops(model, face_up)

    def views_at(scale: float, trunc: float):
        return [(cams[i].scaled(scale), inv_T(T_world_cam[i]), _edge_distance(frames[i], scale, trunc)) for i in ids]

    # coarse: quarter resolution, 6 deg, 4 mm
    vq = views_at(0.25, 8.0)
    yaws = np.arange(0.0, 360.0, 6.0)
    g = np.arange(-radius, radius + 1e-9, 4.0)
    txy = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2) + centre[:2]
    S = _score(loops, yaws, txy, z_floor, vq, centre[:2], 8.0)
    order = np.argsort(S, axis=None)
    picks = []
    for f in order:
        k, j = np.unravel_index(f, S.shape)
        y, t = yaws[k], txy[j]
        if all(min(abs(y - y2) % 180, 180 - abs(y - y2) % 180) > 15 or np.linalg.norm(t - t2) > 8 for y2, t2, _ in picks):
            picks.append((y, t, S[k, j]))
        if len(picks) >= 6:
            break
    log(f"  boardless search coarse: best {picks[0][2]:.2f} px at yaw {picks[0][0]:.0f}, xy {np.round(picks[0][1], 1)}; "
        f"next {[round(p[2], 2) for p in picks[1:4]]}")
    # fine: half resolution, 1.5 deg, 1 mm around each pick
    vh = views_at(0.5, 10.0)
    best = []
    for y0, t0, _ in picks:
        ys = y0 + np.arange(-6.0, 6.01, 1.5)
        gg = np.arange(-5.0, 5.01, 1.0)
        tt = np.stack(np.meshgrid(gg, gg), -1).reshape(-1, 2) + t0
        Sf = _score(loops, ys, tt, z_floor, vh, centre[:2], 10.0)
        k, j = np.unravel_index(np.argmin(Sf), Sf.shape)
        best.append((Sf[k, j], ys[k], tt[j]))
    best.sort(key=lambda b: b[0])
    log(f"  boardless search fine: {[(round(b[0], 2), round(float(b[1]) % 360, 1)) for b in best[:4]]}")
    out = []
    for s, y, t in best[:2]:
        for twin in (0.0, 180.0):
            out.append(make_T(rot_z(y + twin), [t[0], t[1], z_floor]) @ loops.R0)
    return out


def lens_contrast(model, frames: dict[int, np.ndarray], cams: dict[int, Camera], T_world_cam: dict[int, np.ndarray],
                  T_world_cad: np.ndarray, views: list[int], scale: float = 0.25) -> float | None:
    """Median lightness of the back glass minus that of the camera lenses, where the model puts them.
    The outline cannot tell a device from its 180-degree twin; the dark lenses can (back up only)."""
    from ..anchor.parametric import PARTS

    scene = RaycastScene(model.mesh, None)
    lens, back = PARTS.index("lens"), PARTS.index("back")
    Ls, Bs = [], []
    for i in views:
        cam = cams[i].scaled(scale)
        h = scene.cast(cam, inv_T(T_world_cad) @ T_world_cam[i], step=1)
        if not h.hit.any():
            continue
        lab = np.full(h.hit.shape, -1, int)
        lab[h.hit] = model.mesh.face_label[h.face[h.hit]]
        g = cv2.cvtColor(cv2.resize(frames[i], (h.hit.shape[1], h.hit.shape[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        if (lab == lens).sum() >= 10 and (lab == back).sum() >= 50:
            Ls.append(np.median(g[lab == lens]))
            Bs.append(np.median(g[lab == back]))
    if len(Ls) < 3:
        return None
    return float(np.median(np.array(Bs) - np.array(Ls)))


def refine_camera(scene: RaycastScene, frame_grad, cam: Camera, T_wc0: np.ndarray, T_world_cad: np.ndarray,
                  T_prior: np.ndarray, searches=(3.0, 2.0), sigma_rot_deg: float = 0.15, sigma_mm: float = 1.0,
                  max_points: int = 500, min_points: int = 40, rng=None) -> tuple[np.ndarray, dict | None]:
    """One camera pose that puts the model's silhouette on the image edges, pulled towards ``T_prior``
    (right perturbation in the camera frame; the prior residuals are the rotation vector and
    translation of ``inv(T_prior) @ T`` over their sigmas, Huber like the edge terms)."""
    rng = rng or np.random.default_rng(0)
    gx, gy = frame_grad
    T = T_wc0.copy()
    T_cad_world = inv_T(T_world_cad)
    sr, st = np.radians(sigma_rot_deg), sigma_mm
    Tp_inv = inv_T(T_prior)
    info = None
    for S in searches:
        X, n2 = silhouette_points(scene, cam, T_cad_world @ T)
        if len(X) < min_points:
            return T, None
        if len(X) > max_points:
            sel = rng.choice(len(X), max_points, replace=False)
            X, n2 = X[sel], n2[sel]
        Xw = X @ T_world_cad[:3, :3].T + T_world_cad[:3, 3]
        p, _ = _project(cam, inv_T(T), Xw)
        tgt, _ = match_edges(gx, gy, p, n2, S)
        ok = np.isfinite(tgt[:, 0])
        if ok.sum() < min_points:
            return T, None
        Xo, no, to = Xw[ok], n2[ok], tgt[ok]

        def res(xi):
            Tn = T @ se3_exp(xi)
            q, _ = _project(cam, inv_T(Tn), Xo)
            D = Tp_inv @ Tn
            return np.concatenate([np.einsum("ij,ij->i", q - to, no), Rotation.from_matrix(D[:3, :3]).as_rotvec() / sr, D[:3, 3] / st])

        sol = least_squares(res, np.zeros(6), loss="huber", f_scale=1.0, x_scale=[1e-3] * 3 + [0.2] * 3, max_nfev=40)
        T = T @ se3_exp(sol.x)
        info = {"n": int(len(Xo)), "median_px": float(np.median(np.abs(res(np.zeros(6))[: len(Xo)])))}
    return T, info


# ---------------------------------------------------------------------------- joint fit: device on the floor + smooth drift
def _drift_vec(D: np.ndarray, c: np.ndarray) -> np.ndarray:
    """6-vector of a world-frame correction: rotation vector and the displacement of point ``c``."""
    return np.concatenate([Rotation.from_matrix(D[:3, :3]).as_rotvec(), D[:3, :3] @ c + D[:3, 3] - c])


def _drift_T(v: np.ndarray, c: np.ndarray) -> np.ndarray:
    R = Rotation.from_rotvec(v[:3]).as_matrix()
    return make_T(R, c + v[3:] - R @ c)


@dataclass
class JointFit:
    T_world_cad: np.ndarray
    drift: dict[int, np.ndarray]          # frame -> world-frame correction D (camera = D @ ARKit)
    knots: np.ndarray                     # (K, 6) drift knots (rotation vector, displacement of the device centre)
    rounds: list = field(default_factory=list)
    per_view: dict = field(default_factory=dict)
    scale: np.ndarray | None = None       # (width, length, thickness) factors of a plate fitted with the drift


def joint_fit(model, face_up: str, frames: dict[int, np.ndarray], cams: dict[int, Camera], Tw: dict[int, np.ndarray],
              times: dict[int, float], T_init: np.ndarray, z_floor: float | None, views: list[int], knot_times: np.ndarray,
              knots_init: np.ndarray | None = None, searches=(12.0, 8.0, 5.0, 3.0, 2.5), huber_px: float = 1.0,
              max_points_per_view: int = 400, sigma_knot_deg=(0.5, 0.5, 3.0), sigma_knot_mm: float = 15.0,
              sigma_floor_mm: float = 1.5, masks: dict[int, np.ndarray] | None = None, mask_weight: float = 0.3,
              mask_points_per_view: int = 150, fit_scale: bool = False, scale_init: np.ndarray | None = None,
              thickness_sigma: float | None = None, T_base: np.ndarray | None = None, prior_steps: bool = False,
              quadratic_prior: bool = False, knot_bounds: tuple[float, float] | None = None, log=print) -> JointFit:
    """Device heading/position (on the floor) and a piecewise-linear ARKit drift, from image edges.
    ``T_base``: the device lies on the z = 0 plane of this frame instead of the floor (world <- plane;
    ``anchor.placement`` for a device on a stand or leaning); its heading turns about that plane's normal.
    ``prior_steps``: the knot prior is on the change from one knot to the next (a random walk: drift
    builds up) instead of on each knot. ``quadratic_prior``: the prior keeps its quadratic cost however
    far off (least_squares' Huber loss is applied to every residual; on the priors it made the cost
    grow only linearly, and on 509e5ece1b two knots over a stretch of edge-on views ran to 4 m and 80
    degrees, cancelling each other, at 0.85 px; 2026-11-18). ``knot_bounds`` = (deg, mm): each knot's
    rotation and translation are held within these (over the last seconds of black screen on 509e5ece1b
    the outline alone still let the end run to 1.4 m and 48 degrees; ARKit's true drift was below
    40 mm and 12 degrees).
    ``knot_times`` are fixed for the whole recording so that fits on growing time windows can hand
    their knots on (``knots_init``). ``masks``: device masks for the mask term (``anchor.register``).

    ``fit_scale`` (plate models, no drawing): the plate's width, length and thickness factors are
    fitted too, the floor prior following the thickness (the centre stays half a thickness above the
    floor). Cameras moved along their axes towards the device look exactly like a larger device, and
    the drift (15 mm per knot, weakly held) can do that; without the factors the drift took up the
    error of the starting plate and the size fit kept it (synthetic iPhone 12 from a plate 4 mm too
    long: 148.4 x 73.5 mm, 2026-11-17). With them a size error costs nothing in the prior, and the
    drift keeps to what the data needs. The thickness has its own factor because a plate too thick
    shows its side walls further out, which the length and width would otherwise make up for.
    ``thickness_sigma`` holds that factor at its start (log scale) for a pass whose top face had no
    texture to measure: over the iPad's black screen it fell to a 3.1 mm tablet (Apple 5.1)."""
    R0 = rest_pose(model, face_up)
    # the device: heading (deg, about the world z axis) and position offsets; its tilt stays the resting
    # tilt of ``rest_pose`` (it lies on the floor, and ARKit's gravity is the world z axis)
    Tb = np.eye(4) if T_base is None else np.asarray(T_base, float)
    Ti = inv_T(Tb) @ T_init
    c0 = Ti[:3, 3].copy()
    Rz = Ti[:3, :3] @ R0[:3, :3].T
    yaw0 = float(np.degrees(np.arctan2(Rz[1, 0], Rz[0, 0])))

    def T_dev(q):
        yaw, dx, dy, dz = q
        return Tb @ make_T(rot_z(yaw0 + yaw) @ R0[:3, :3], [c0[0] + dx, c0[1] + dy, c0[2] + dz])

    kt = np.asarray(knot_times, float)
    n_knots = len(kt)
    w_knot = {}
    for i in views:
        x = np.interp(times[i], kt, np.arange(n_knots))
        k0 = int(min(np.floor(x), n_knots - 2))
        w_knot[i] = (k0, x - k0)
    scene = RaycastScene(model.mesh, None)
    grads = {i: _grad(frames[i]) for i in views}
    fields = mask_fields(masks, cams, views)
    sr = np.radians(np.asarray(sigma_knot_deg, float))
    zf_target = None if z_floor is None else T_init[2, 3]
    ns = 3 if fit_scale else 0
    half_T = float(model.spec.thickness) / 2
    q = np.zeros(4 + 6 * (n_knots - 1) + ns)
    if knots_init is not None:
        q[4:4 + 6 * (n_knots - 1)] = np.asarray(knots_init)[1:].ravel()
    if fit_scale and scale_init is not None:
        q[-3:] = np.log(np.asarray(scale_init, float))
    rng = np.random.default_rng(0)
    fit = JointFit(T_dev(q[:4]), {}, np.zeros((n_knots, 6)))

    def unpack(q):
        knots = np.vstack([np.zeros(6), q[4:4 + 6 * (n_knots - 1)].reshape(n_knots - 1, 6)])
        return T_dev(q[:4]), knots

    def scale_of(q):
        """CAD-axis factors (x = width, y = length, z = thickness)."""
        return np.exp(q[-3:]) if fit_scale else np.ones(3)

    def drift_of(i, knots, c):
        k0, a = w_knot[i]
        return _drift_T((1 - a) * knots[k0] + a * knots[k0 + 1], c)

    for rnd, S in enumerate(searches):
        T_wcad, knots = unpack(q)
        c = T_wcad[:3, 3]
        data, mdata = [], []
        wm = mask_weight_for(S, mask_weight) if fields else 0.0
        s_now = scale_of(q)
        if fit_scale:
            # silhouettes of the plate at its current size; the points are kept in unit-plate coordinates
            scene = RaycastScene(LabeledMesh(model.mesh.vertices * s_now, model.mesh.faces, model.mesh.face_label), None)
        for i in views:
            Twc = drift_of(i, knots, c) @ Tw[i]
            X, n2 = silhouette_points(scene, cams[i], inv_T(T_wcad) @ Twc)
            if len(X) == 0:
                continue
            X = X / s_now
            if len(X) > max_points_per_view:
                sel = rng.choice(len(X), max_points_per_view, replace=False)
                X, n2 = X[sel], n2[sel]
            p, _ = _project(cams[i], inv_T(Twc), (X * s_now) @ T_wcad[:3, :3].T + T_wcad[:3, 3])
            tgt, _ = match_edges(*grads[i], p, n2, S)
            ok = np.isfinite(tgt[:, 0])
            if ok.sum() >= 20:
                data.append((i, X[ok], n2[ok], tgt[ok]))
            if wm > 0 and i in fields:
                _, inb = fields[i](p)
                Xm = X[inb]
                if len(Xm) > mask_points_per_view:
                    Xm = Xm[rng.choice(len(Xm), mask_points_per_view, replace=False)]
                if len(Xm):
                    mdata.append((i, Xm))
        if not data:
            break

        def residuals(qq):
            T_wcad_, knots_ = unpack(qq)
            sc = scale_of(qq)
            c_ = T_wcad_[:3, 3]
            out = []
            for i, X, n2, tgt in data:
                Twc = drift_of(i, knots_, c_) @ Tw[i]
                p, _ = _project(cams[i], inv_T(Twc), (X * sc) @ T_wcad_[:3, :3].T + T_wcad_[:3, 3])
                out.append(np.einsum("ij,ij->i", p - tgt, n2))
            for i, Xm in mdata:
                Twc = drift_of(i, knots_, c_) @ Tw[i]
                p, _ = _project(cams[i], inv_T(Twc), (Xm * sc) @ T_wcad_[:3, :3].T + T_wcad_[:3, 3])
                out.append(wm * fields[i](p)[0])
            kk = np.diff(knots_, axis=0) if prior_steps else knots_[1:]
            pri = [kk[:, :3].ravel() / np.tile(sr, n_knots - 1), kk[:, 3:].ravel() / sigma_knot_mm]
            if quadratic_prior:
                # Huber on z' gives z^2 beyond 1 when z' = sign(z) (z^2 + 1) / 2 (continuous, slope 1 at 1)
                pri = [np.where(np.abs(z) > 1.0, np.sign(z) * (z * z + 1.0) / 2.0, z) for z in pri]
            if zf_target is not None:
                pri.append(np.array([(T_wcad_[2, 3] - zf_target - (sc[2] - 1.0) * half_T) / sigma_floor_mm]))
            if fit_scale:
                # held (thickness_sigma) or free; a free factor still gets a weak 25 % prior: on a capture
                # filmed only from above (cc53da7c23, 2026-11-18) the side band hardly showed and the
                # factor ran to 0, and the plate model could not be built
                pri.append(np.array([qq[-1] / (thickness_sigma if thickness_sigma else 0.25)]))
            return np.concatenate(out + pri)

        n_data = sum(len(d[1]) for d in data)
        r0 = residuals(q)[:n_data]
        bounds = (-np.inf, np.inf)
        if knot_bounds is not None:
            kb = np.tile(np.r_[[np.radians(knot_bounds[0])] * 3, [knot_bounds[1]] * 3], n_knots - 1)
            lo = np.r_[np.full(4, -np.inf), -kb, np.full(ns, -np.inf)]
            hi = np.r_[np.full(4, np.inf), kb, np.full(ns, np.inf)]
            q = np.clip(q, lo + 1e-9, hi - 1e-9)
            bounds = (lo, hi)
        sol = least_squares(residuals, q, loss="huber", f_scale=huber_px, max_nfev=30, bounds=bounds,
                            x_scale=np.r_[0.05, 0.2, 0.2, 0.2, np.tile(np.r_[1e-3, 1e-3, 1e-3, 0.2, 0.2, 0.2], n_knots - 1),
                                          np.full(ns, 3e-3)])
        q = sol.x
        r1 = residuals(q)[:n_data]
        info = {"round": rnd, "search_px": S, "n_points": int(n_data), "n_views": len(data),
                "median_abs_px_before": float(np.median(np.abs(r0))), "median_abs_px_after": float(np.median(np.abs(r1)))}
        fit.rounds.append(info)
        if fit_scale:
            info["scale"] = [round(float(x), 4) for x in scale_of(q)]
        log(f"  boardless joint fit round {rnd}: {len(data)} views, {n_data} edge points, |r| median "
            f"{info['median_abs_px_before']:.2f} -> {info['median_abs_px_after']:.2f} px"
            + (f", plate x {info['scale'][1]:.4f} long, x {info['scale'][0]:.4f} wide, x {info['scale'][2]:.3f} thick" if fit_scale else ""))
    T_wcad, knots = unpack(q)
    fit.T_world_cad, fit.knots = T_wcad, knots
    fit.scale = scale_of(q) if fit_scale else None
    c = T_wcad[:3, 3]
    for i in times:   # every frame, also those not used as views
        x = np.interp(times[i], kt, np.arange(n_knots))
        k0 = int(min(np.floor(x), n_knots - 2))
        a = x - k0
        fit.drift[i] = _drift_T((1 - a) * knots[k0] + a * knots[k0 + 1], c)
    return fit


def boardless_poses(cap, frames: dict[int, np.ndarray], cams: dict[int, Camera], model, face_up: str,
                    n_search_views: int = 12, early_share: float = 0.3, n_fit_views: int = 48, polish: bool = False,
                    min_lens_contrast: float = 15.0, plate=None, masks: dict[int, np.ndarray] | None = None,
                    z_floor: float | None = None, fit_plate: bool = False, log=print) -> BoardlessResult:
    """Camera poses (mm, a z-up world tied to the first part of the recording) for ``frames``.
    ``plate`` (``anchor.floor_outline.FloorPlate``) adds the device's measured pose on the floor to the
    candidates; for a fitted plate model (no drawing) it is measured here when not given.
    ``masks`` (SAM 2, ``anchor.locate``) add the mask term to the joint fit; ``z_floor`` replaces the
    LiDAR floor (``anchor.locate.refine_floor`` measures it from the images: the LiDAR plane lay
    5-6.5 mm too low on the iPad recordings)."""
    ids = sorted(frames)
    Tw = {i: T_W_ARK @ cap.T_ark_cam[i] for i in ids}
    times = {i: float(cap.timestamps[i] - cap.timestamps[0]) for i in ids}
    info: dict = {}
    centre = axes_meeting_point([Tw[i] for i in ids])
    # the floor from the early keyframes only: the world is tied to the start of the recording, and
    # later frames carry ARKit's drift (3.3 mm too high on the synthetic wood capture when all were used)
    n_early = max(n_search_views, int(round(len(ids) * early_share)))
    early = ids[:n_early]
    if z_floor is not None:
        zf = float(z_floor)
        info["floor_from"] = "images"
    else:
        zf = floor_height(cap, early[:: max(1, len(early) // 8)], Tw, centre)
        info["floor_from"] = "lidar" if zf is not None else "axes"
    z_search = zf if zf is not None else float(centre[2] - model.spec.thickness)
    info["floor_z"] = z_search
    log(f"  boardless: optical axes meet at {np.round(centre, 1)} mm; floor z {z_search:.1f} mm ({info['floor_from']})")
    sv = [early[int(k)] for k in np.linspace(0, len(early) - 1, min(n_search_views, len(early))).round()]
    cands = find_device(model, face_up, frames, cams, Tw, centre, z_search, sv, log=log)
    if zf is not None and (plate is not None or model.spec.raw.get("template")):
        # the outline search looks within a few cm of where the cameras aimed, sized for phones;
        # the floor mosaic finds a device of any size, so its pose is tried first
        if plate is None:
            from ..anchor.floor_outline import measure_plate

            plate = measure_plate(frames, cams, Tw, zf, centre[:2], log=log)
        if plate is not None:
            R0 = rest_pose(model, face_up)              # long side along world +y before the heading
            yaw = plate.long_axis_deg - 90.0
            # only the measured pose (and its 180-degree twin): with the outline search's candidates
            # in the race, a pose turned 90 degrees onto floor-board seams won by 0.01 px on the iPad
            # screen pass (all candidates 1.83-1.87 px after the short fit, 2026-11-17)
            cands = [make_T(rot_z(yaw + tw), [plate.centre_xy[0], plate.centre_xy[1], zf]) @ R0 for tw in (0.0, 180.0)]
            info["floor_plate"] = plate.info
    fit_views = [ids[int(k)] for k in np.linspace(0, len(ids) - 1, min(n_fit_views, len(ids))).round()]
    n_knots = int(np.clip(round(len(ids) / 30), 3, 6))
    fit_scale = bool(fit_plate and model.spec.raw.get("template"))
    # the thickness is left free only where the top face's own height was measured (texture on it).
    # Elsewhere it is held: over the iPad's black screen the strongest edge all round is the glass
    # outline, which a plate without thickness explains best; a 2 % prior lost to 18,000 edge points
    # (0.43 mm, 2026-11-17)
    t_sigma = None if (plate is not None and plate.top_height is not None and not plate.info.get("top_shared")) else 1e-3
    t_lo, t_hi = min(times.values()), max(times.values())
    kt = np.linspace(t_lo, t_hi, n_knots)

    def window(frac):
        return [i for i in fit_views if times[i] <= t_lo + frac * (t_hi - t_lo) + 1e-9]

    # the candidates (both 180-degree twins of the two best headings): short fits on the early part
    trials = []
    for T0 in cands:
        f = joint_fit(model, face_up, frames, cams, Tw, times, T0, zf, window(0.4), kt, searches=(12.0, 6.0), masks=masks,
                      fit_scale=fit_scale, thickness_sigma=t_sigma, log=lambda m: None)
        trials.append((f.rounds[-1]["median_abs_px_after"] if f.rounds else np.inf, f))
    from ..anchor.parametric import PARTS

    has_lens = bool((model.mesh.face_label == PARTS.index("lens")).any())
    if face_up == "back" and has_lens:     # settle the 180-degree twins by the dark lenses (lightness difference, 0-255)
        for k, (px, f) in enumerate(trials):
            con = lens_contrast(model, frames, cams, {i: f.drift[i] @ Tw[i] for i in sv}, f.T_world_cad, sv)
            trials[k] = (px, f, con)
        best_px = min(t[0] for t in trials)
        near = [t for t in trials if t[0] <= best_px * 1.3 + 0.1]
        near.sort(key=lambda t: -(t[2] if t[2] is not None else -1e9))
        trials = near + [t for t in trials if t not in near]
        info["candidates_lens_contrast"] = [None if t[2] is None else round(t[2], 1) for t in trials]
        best_con = max((t[2] for t in trials if t[2] is not None), default=None)
        if best_con is None or best_con < min_lens_contrast:
            # no candidate puts the model's lenses on dark pixels: the device was not found (on the
            # printed board itself the search locks onto the checker pattern, contrast -47)
            from ..errors import FlawmapError

            raise FlawmapError("DEVICE_NOT_FOUND", f"boardless: no heading puts the camera lenses on dark pixels "
                                                   f"(lens contrast {info['candidates_lens_contrast']})")
    else:
        trials.sort(key=lambda x: x[0])
    info["candidates_px"] = [round(t[0], 3) for t in trials]
    log(f"  boardless: candidates (median px after a short fit) {info['candidates_px']}"
        + (f", lens contrast {info['candidates_lens_contrast']}" if "candidates_lens_contrast" in info else ""))
    # grow the time window: drift late in the recording is tens of pixels, beyond any search window,
    # so each window starts from the drift found so far (knots past the window copy the last one)
    fit = trials[0][1]
    for frac in (0.4, 0.7, 1.0):
        vs = window(frac)
        t_end = max(times[i] for i in vs)
        kn = fit.knots.copy()
        last = int(np.searchsorted(kt, t_end - 1e-9))          # first knot at or beyond the window's end
        cover = max(1, min(last, n_knots - 1))
        kn[cover + 1:] = kn[cover]
        fit = joint_fit(model, face_up, frames, cams, Tw, times, fit.T_world_cad, zf, vs, kt, knots_init=kn,
                        searches=(10.0, 6.0, 4.0) if frac < 1.0 else (6.0, 4.0, 3.0, 2.5), masks=masks,
                        fit_scale=fit_scale, scale_init=fit.scale, thickness_sigma=t_sigma, log=log)
    final = {i: fit.drift[i] @ Tw[i] for i in ids}
    if polish:
        # optional last step: each frame on its own, a few pixels from the joint fit and pulled to it.
        # Off by default: on the synthetic wood capture it raised the median model reprojection from
        # 0.21 to 1.14 px (max 2.45 -> 2.07); one frame's outline leaves weak directions.
        scene = RaycastScene(model.mesh, None)
        moved = []
        for i in ids:
            T_ref, st = refine_camera(scene, _grad(frames[i]), cams[i], final[i], fit.T_world_cad, final[i],
                                      searches=(3.0, 2.0), sigma_rot_deg=0.15, sigma_mm=1.0)
            if st is not None and st["median_px"] < 1.0:
                moved.append(np.linalg.norm(T_ref[:3, 3] - final[i][:3, 3]))
                final[i] = T_ref
        info["polished"] = len(moved)
        info["polish_move_mm_median"] = float(np.median(moved)) if moved else None
        log(f"  boardless: polished {len(moved)}/{len(ids)} frames, median move {info['polish_move_mm_median']}")
    c = fit.T_world_cad[:3, 3]
    dv = np.array([_drift_vec(fit.drift[i], c) for i in ids])
    info.update({"n_frames": len(ids), "n_knots": n_knots, "rounds": fit.rounds,
                 "drift_mm_max": float(np.linalg.norm(dv[:, 3:], axis=1).max()),
                 "drift_deg_max": float(np.degrees(np.linalg.norm(dv[:, :3], axis=1)).max()),
                 "median_px": fit.rounds[-1]["median_abs_px_after"] if fit.rounds else None,
                 "plate_scale": None if fit.scale is None else [round(float(x), 5) for x in fit.scale]})
    log(f"  boardless: ARKit drift up to {info['drift_mm_max']:.1f} mm / {info['drift_deg_max']:.2f} deg, "
        f"outline residual {info['median_px']:.2f} px")
    poses = CapturePoses(ids, final, {i: "model" for i in ids}, {}, T_W_ARK, {"mode": "boardless", **info})
    return BoardlessResult(poses, fit.T_world_cad, info)


def tilted_poses(cap, frames: dict[int, np.ndarray], cams: dict[int, Camera], model, face_up: str, inits: list[np.ndarray],
                 T_base: np.ndarray, masks: dict[int, np.ndarray] | None = None, n_fit_views: int = 48, fit_plate: bool = False,
                 knots_per_s: float = 1 / 2.5, max_knots: int = 10, windows=(0.25, 0.4, 0.55, 0.7, 0.85, 1.0), mask_weight: float = 0.6,
                 log=print) -> BoardlessResult:
    """``boardless_poses`` for a device that stands or leans (``anchor.placement``): the same joint fit of
    the device's pose, the plate's size and a piecewise-linear ARKit drift to the image edges and the
    masks, on growing time windows, but the device lies on its own plane (``T_base``: world <- plane)
    instead of the floor, and there is no floor prior. ``inits``: the 180-degree twins of the first
    ``T_world_cad``; the one with the smaller residual after a short fit on the first 40 % is kept.

    Filmed all round a stand (509e5ece1b, 20.8 s), ARKit drifted about 36 mm between the device's two
    sides; with the raw ARKit poses no plate fitted both (mask IoU p10 0.48, 2026-11-18). The floor
    path's schedule (a knot every ~4 s, windows 40/70/100 %, mask weight 0.3) left the last third
    (the screen side) 20-40 px off (IoU p10 there 0.25); a knot every 2.5 s, six windows and mask
    weight 0.6 gave 0.81 (all frames: p10 0.57 -> 0.83, edges 0.65 -> 0.64 px)."""
    ids = sorted(frames)
    Tw = {i: T_W_ARK @ cap.T_ark_cam[i] for i in ids}
    times = {i: float(cap.timestamps[i] - cap.timestamps[0]) for i in ids}
    info: dict = {"mode": "tilted"}
    fit_views = [ids[int(k)] for k in np.linspace(0, len(ids) - 1, min(n_fit_views, len(ids))).round()]
    t_lo, t_hi = min(times.values()), max(times.values())
    n_knots = int(np.clip(round((t_hi - t_lo) * knots_per_s) + 1, 3, max_knots))
    kt = np.linspace(t_lo, t_hi, n_knots)
    # the plate's size is held here (it comes from LiDAR, which is metric): without a floor, moving the
    # cameras along their axes (the drift) looks the same as a larger plate, and with the size free the
    # leaning iPad came out 289.8 x 220.2 mm (281.6 x 215.5; 6c417d24a0, 2026-11-18). The outline size
    # fit (anchor.template) measures it afterwards on the drift-corrected cameras.
    fit_scale = False
    t_sigma = 1e-3

    def window(frac):
        return [i for i in fit_views if times[i] <= t_lo + frac * (t_hi - t_lo) + 1e-9]

    # drift as a random walk (per knot step 1 / 1 / 3 deg and 15 mm), its prior quadratic however far off
    prior = dict(prior_steps=True, quadratic_prior=True, sigma_knot_deg=(1.0, 1.0, 3.0), sigma_knot_mm=15.0, knot_bounds=(15.0, 80.0))
    trials = []
    for T0 in inits:
        f = joint_fit(model, face_up, frames, cams, Tw, times, T0, None, window(windows[0]), kt, searches=(12.0, 6.0), masks=masks,
                      mask_weight=mask_weight, fit_scale=fit_scale, thickness_sigma=t_sigma, T_base=T_base, **prior, log=lambda m: None)
        trials.append((f.rounds[-1]["median_abs_px_after"] if f.rounds else np.inf, f))
    trials.sort(key=lambda x: x[0])
    info["candidates_px"] = [round(t[0], 3) for t in trials]
    log(f"  tilted: twins (median px after a short fit) {info['candidates_px']}")
    fit = trials[0][1]
    for frac in windows:
        vs = window(frac)
        t_end = max(times[i] for i in vs)
        kn = fit.knots.copy()
        last = int(np.searchsorted(kt, t_end - 1e-9))
        cover = max(1, min(last, n_knots - 1))
        kn[cover + 1:] = kn[cover]
        # every window adds views whose drift starts as the last covered knot's, so each one begins with a wide
        # search, the last one too: with 6 px there, the last seconds of 509e5ece1b (the screen side, after the
        # stand) matched other edges 80 mm off in one run (mask IoU 0.08-0.11 at frames 999-1132; screen side p50
        # 0.815) and fitted in another with a plate 0.4 mm different (0.927), 2026-11-19
        fit = joint_fit(model, face_up, frames, cams, Tw, times, fit.T_world_cad, None, vs, kt, knots_init=kn,
                        searches=(12.0, 6.0, 4.0) if frac < 1.0 else (12.0, 6.0, 4.0, 3.0, 2.5), masks=masks, mask_weight=mask_weight,
                        fit_scale=fit_scale, scale_init=fit.scale, thickness_sigma=t_sigma, T_base=T_base, **prior, log=log)
    final = {i: fit.drift[i] @ Tw[i] for i in ids}
    c = fit.T_world_cad[:3, 3]
    dv = np.array([_drift_vec(fit.drift[i], c) for i in ids])
    info.update({"n_frames": len(ids), "n_knots": n_knots, "rounds": fit.rounds,
                 "drift_mm_max": float(np.linalg.norm(dv[:, 3:], axis=1).max()),
                 "drift_deg_max": float(np.degrees(np.linalg.norm(dv[:, :3], axis=1)).max()),
                 "median_px": fit.rounds[-1]["median_abs_px_after"] if fit.rounds else None,
                 "plate_scale": None if fit.scale is None else [round(float(x), 5) for x in fit.scale]})
    log(f"  tilted: ARKit drift up to {info['drift_mm_max']:.1f} mm / {info['drift_deg_max']:.2f} deg, "
        f"outline residual {info['median_px']:.2f} px")
    poses = CapturePoses(ids, final, {i: "model" for i in ids}, {}, T_W_ARK, {"mode": "tilted", **info})
    return BoardlessResult(poses, fit.T_world_cad, info)
