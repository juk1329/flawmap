"""Reference device model built from the dimensions on the manufacturer's drawing.

Why build it ourselves: Apple publishes vector dimensional drawings (outline, corner profile, side
profile, camera plateau, buttons) but no 3D files, and third-party models need logins or carry
unknown accuracy. A model generated from a small YAML of drawing dimensions has no licence issue,
is easy to extend to another model, and its accuracy is that of the drawing (overall size to
~0.05 mm; the undimensioned roll-offs are estimated and marked as such in the YAML).

The model is a set of labelled triangle meshes (a "soup", not a boolean union): the body is a loft
of the plan outline offset by the side profile, and the camera plateau, lens rings and buttons are
separate closed pieces sitting on it. Ray casting only needs the first hit, so the overlaps do not
matter, and every vertex keeps the unwrapped surface coordinates of its chart:

* ``screen`` - front face, ``u = x + W/2``, ``v = L/2 - y`` (mm, image-like: top edge at v = 0)
* ``back``   - back face seen from behind, ``u = W/2 - x``, ``v = L/2 - y`` (camera parts project here)
* ``frame``  - the side band, ``u = s`` = arc length along the widest outline, clockwise as seen from
  the front starting at the top centre, ``v = z_front - z`` (front edge at v = 0)

Frames follow ``geometry.transforms`` (cad frame: origin at the body centre, +Y to the top edge,
+Z out of the screen, mm).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from ..geometry.mesh_io import LabeledMesh
from ..geometry.transforms import make_T, rot_y
from ..utils.io import read_yaml, repo_root

PARTS = ["screen", "back", "frame", "camera", "lens", "button"]
CHARTS = ["screen", "back", "frame"]


@dataclass
class DeviceSpec:
    name: str
    slug: str
    length: float
    width: float
    thickness: float
    corner: list
    side_profile: list
    camera: dict
    buttons: list = field(default_factory=list)
    openings: list = field(default_factory=list)
    default_color: dict = field(default_factory=dict)
    source: str = ""
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "DeviceSpec":
        d = read_yaml(path)
        keys = {k: d[k] for k in ("name", "slug", "length", "width", "thickness", "corner", "side_profile", "camera")}
        return cls(**keys, buttons=d.get("buttons", []), openings=d.get("openings", []),
                   default_color=d.get("default_color", {}), source=d.get("source", ""), raw=d)

    @classmethod
    def by_slug(cls, slug: str) -> "DeviceSpec":
        return cls.load(repo_root() / "configs" / "devices" / f"{slug}.yaml")


# ----------------------------------------------------------------------------- plan outline
def corner_curve(points: list, step: float = 0.25) -> np.ndarray:
    """Corner curve in local inset coordinates (inset from the side edge, inset from the top edge),
    from the tangent point on the side edge to the tangent point on the top edge.

    A clamped cubic spline through the drawing's points, tangent to both edges at its ends."""
    p = np.asarray(points, dtype=float)
    t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    cs = CubicSpline(t, p, bc_type=((1, np.array([0.0, -1.0])), (1, np.array([1.0, 0.0]))))
    n = max(int(np.ceil(t[-1] / step)), 4)
    out = cs(np.linspace(0.0, t[-1], n + 1))
    return np.maximum(out, 0.0)            # a spline through few points can overshoot the edges slightly


def _densify(a: np.ndarray, b: np.ndarray, step: float) -> np.ndarray:
    """Points from a to b (a included, b excluded)."""
    n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
    return a + (b - a) * (np.arange(n)[:, None] / n)


def plan_outline(spec: DeviceSpec, corner_step: float = 0.25, edge_step: float = 1.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Widest plan outline, **clockwise** seen from the front, starting at the top centre.

    Returns ``(xy (N,2), s (N,), perimeter)``; ``s`` is the arc length from the top centre (the
    point after the last one closes the loop at ``s = perimeter``)."""
    W2, L2 = spec.width / 2.0, spec.length / 2.0
    loc = corner_curve(spec.corner, corner_step)                    # side-edge tangent -> top-edge tangent
    q1 = np.stack([W2 - loc[:, 0], L2 - loc[:, 1]], axis=1)        # top-right, ccw (right edge -> top edge)
    q1_cw = q1[::-1]                                                # top edge -> right edge (clockwise)
    q4_cw = q1 * np.array([1.0, -1.0])                              # bottom-right: mirror in y (right edge -> bottom edge)
    q3_cw = q1[::-1] * np.array([-1.0, -1.0])                       # bottom-left: rotate 180 (bottom edge -> left edge)
    q2_cw = q1 * np.array([-1.0, 1.0])                              # top-left: mirror in x (left edge -> top edge)
    top_c = np.array([0.0, L2])
    pts = [
        _densify(top_c, q1_cw[0], edge_step),        # top edge, right half
        q1_cw[:-1],
        _densify(q1_cw[-1], q4_cw[0], edge_step),     # right edge
        q4_cw[:-1],
        _densify(q4_cw[-1], q3_cw[0], edge_step),     # bottom edge
        q3_cw[:-1],
        _densify(q3_cw[-1], q2_cw[0], edge_step),     # left edge
        q2_cw[:-1],
        _densify(q2_cw[-1], top_c, edge_step),        # top edge, left half
    ]
    xy = np.concatenate(pts, axis=0)
    seg = np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)
    keep = np.concatenate([[True], seg[:-1] > 1e-9])
    xy = xy[keep]
    seg = np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    return xy, s, float(seg.sum())


def outward_normals(xy: np.ndarray) -> np.ndarray:
    """Outward unit normals of a closed clockwise polygon (as seen from +z)."""
    t = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)     # clockwise polygon: left normal points outward
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def profile_samples(spec: DeviceSpec, step: float = 0.12) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Side profile densified with a shape-preserving (PCHIP) interpolation through the drawing points.

    Returns insets, z values (front to back) and the part label of each segment (len n-1)."""
    rows = spec.side_profile
    p = np.array([[float(r[0]), float(r[1])] for r in rows])
    labels = [str(r[2]) for r in rows]
    t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    fi, fz = PchipInterpolator(t, p[:, 0]), PchipInterpolator(t, p[:, 1])
    ts = [t[0]]
    seg_lab = []
    for k in range(len(t) - 1):
        n = max(int(np.ceil((t[k + 1] - t[k]) / step)), 1)
        ts.extend(np.linspace(t[k], t[k + 1], n + 1)[1:])
        seg_lab.extend([labels[k]] * n)
    ts = np.asarray(ts)
    return fi(ts), fz(ts), seg_lab


def rounded_rect(x0: float, x1: float, y0: float, y1: float, r: float, n_corner: int = 16, edge_step: float = 1.0) -> np.ndarray:
    """Clockwise rounded rectangle (seen from +z), starting at the top centre."""
    r = min(r, (x1 - x0) / 2 - 1e-6, (y1 - y0) / 2 - 1e-6)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    corners = [((x1 - r, y1 - r), 90, 0), ((x1 - r, y0 + r), 0, -90), ((x0 + r, y0 + r), -90, -180), ((x0 + r, y1 - r), 180, 90)]
    pts = []
    start = np.array([cx, y1])
    prev = start
    for (ccx, ccy), a0, a1 in corners:
        ang = np.radians(np.linspace(a0, a1, n_corner + 1))
        arc = np.stack([ccx + r * np.cos(ang), ccy + r * np.sin(ang)], axis=1)
        pts.append(_densify(prev, arc[0], edge_step))
        pts.append(arc[:-1])
        prev = arc[-1]
    pts.append(_densify(prev, start, edge_step))
    return np.concatenate(pts, axis=0)


# ----------------------------------------------------------------------------- model
@dataclass
class DeviceModel:
    spec: DeviceSpec
    mesh: LabeledMesh                 # faces labelled with PARTS
    vchart: np.ndarray                # (V,) chart index of each vertex (CHARTS)
    vuv: np.ndarray                   # (V,2) chart coordinates in mm
    outline_xy: np.ndarray            # widest plan outline, clockwise from the top centre
    outline_s: np.ndarray
    perimeter: float
    z_front: float
    z_back: float

    @property
    def face_chart(self) -> np.ndarray:
        return self.vchart[self.mesh.faces[:, 0]]

    def chart_size(self, chart: str) -> tuple[float, float]:
        if chart in ("screen", "back"):
            return self.spec.width, self.spec.length
        return self.perimeter, self.z_front - self.z_back

    def surface_coords(self, face: np.ndarray, bary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Chart index and chart coordinates (mm) of hit points given by face id and barycentric
        ``(b1, b2)`` (point = (1-b1-b2) v0 + b1 v1 + b2 v2, Open3D's convention)."""
        face = np.asarray(face)
        f = self.mesh.faces[face]
        b1, b2 = bary[..., 0], bary[..., 1]
        w0 = 1.0 - b1 - b2
        uv = w0[..., None] * self.vuv[f[..., 0]] + b1[..., None] * self.vuv[f[..., 1]] + b2[..., None] * self.vuv[f[..., 2]]
        return self.vchart[f[..., 0]], uv

    def frame_s_of_xy(self, xy: np.ndarray) -> np.ndarray:
        """Arc length ``s`` of the outline point closest to each plan point."""
        xy = np.atleast_2d(xy)
        d = np.linalg.norm(xy[:, None, :] - self.outline_xy[None, :, :], axis=2)
        return self.outline_s[np.argmin(d, axis=1)]

    def mirror_s(self, s: np.ndarray, axis: str) -> np.ndarray:
        """Frame coordinate of the mirrored position: ``x`` mirror (left/right), ``y`` mirror
        (top/bottom) or ``xy`` (180-degree rotation). Valid because the outline is symmetric."""
        P = self.perimeter
        s = np.asarray(s, dtype=float)
        if axis == "x":
            return (P - s) % P
        if axis == "y":
            return (P / 2.0 - s) % P
        if axis == "xy":
            return (s + P / 2.0) % P
        raise ValueError(axis)

    def frame_exclusions(self, margin: float = 3.0) -> list[dict]:
        """Frame-chart s ranges of buttons and openings (with ``margin`` mm), where frame-defect
        detection is switched off (a button is 0.4-0.6 mm proud of the frame and would look like
        a defect)."""
        W2, L2 = self.spec.width / 2.0, self.spec.length / 2.0
        out = []
        for item in list(self.spec.buttons) + list(self.spec.openings):
            side = item["side"]
            half = item["length"] / 2.0 + margin
            if side in ("left", "right"):
                x = -W2 if side == "left" else W2
                yc = L2 - item["y_from_top"]
                s0, s1 = self.frame_s_of_xy(np.array([[x, yc + half], [x, yc - half]]))
            elif side in ("top", "bottom"):
                y = L2 if side == "top" else -L2
                xc = item["x_from_left"] - W2
                s0, s1 = self.frame_s_of_xy(np.array([[xc - half, y], [xc + half, y]]))
            else:
                continue
            lo, hi = (s0, s1) if (s1 - s0) % self.perimeter <= self.perimeter / 2 else (s1, s0)
            out.append({"name": item["name"], "s0": float(lo), "s1": float(hi)})
        return out


def _ring_faces(n_rings: int, n_per_ring: int, base: int, closed: bool) -> np.ndarray:
    """Triangles between consecutive rings of ``n_per_ring`` vertices; outward for clockwise rings
    stacked from front (+z) to back (-z)."""
    faces = []
    m = n_per_ring if closed else n_per_ring - 1
    for k in range(n_rings - 1):
        a = base + k * n_per_ring
        b = base + (k + 1) * n_per_ring
        j = np.arange(m)
        j1 = (j + 1) % n_per_ring
        faces.append(np.stack([a + j, a + j1, b + j1], axis=1))
        faces.append(np.stack([a + j, b + j1, b + j], axis=1))
    return np.concatenate(faces, axis=0) if faces else np.zeros((0, 3), np.int64)


def _fan(center_idx: int, ring_idx: np.ndarray, up: bool) -> np.ndarray:
    """Fan triangles over a clockwise ring; normal +z if ``up`` else -z."""
    r0, r1 = ring_idx, np.roll(ring_idx, -1)
    c = np.full_like(r0, center_idx)
    return np.stack([c, r1, r0], axis=1) if up else np.stack([c, r0, r1], axis=1)


def build_device_model(spec: DeviceSpec, corner_step: float = 0.25, edge_step: float = 1.0,
                       profile_step: float = 0.12) -> DeviceModel:
    W2, L2 = spec.width / 2.0, spec.length / 2.0
    xy, s, P = plan_outline(spec, corner_step, edge_step)
    nrm = outward_normals(xy)
    insets, zs, seg_lab = profile_samples(spec, profile_step)
    z_front, z_back = float(zs[0]), float(zs[-1])

    V, F, FL, VC, VUV = [], [], [], [], []
    nv = 0

    def add(verts, faces, part, chart, uv):
        nonlocal nv
        V.append(verts); F.append(faces + nv); FL.append(np.full(len(faces), PARTS.index(part), np.int32))
        VC.append(np.full(len(verts), CHARTS.index(chart), np.int32)); VUV.append(uv)
        nv += len(verts)

    # --- side loft (chart frame). Rings get a duplicated closing vertex at s = P so UVs do not wrap.
    ring_xy = np.vstack([xy, xy[:1]])
    ring_n = np.vstack([nrm, nrm[:1]])
    ring_s = np.concatenate([s, [P]])
    n_ring = len(ring_xy)
    verts, uvs = [], []
    for ins, z in zip(insets, zs):
        p = ring_xy - ins * ring_n
        verts.append(np.column_stack([p, np.full(n_ring, z)]))
        uvs.append(np.column_stack([ring_s, np.full(n_ring, z_front - z)]))
    verts = np.concatenate(verts)
    uvs = np.concatenate(uvs)
    loft_f = _ring_faces(len(zs), n_ring, 0, closed=False)
    # part label per ring segment (each ring pair contributes 2*(n_ring-1) faces)
    loft_lab = np.repeat([PARTS.index(l) for l in seg_lab], 2 * (n_ring - 1))
    V.append(verts); F.append(loft_f); FL.append(loft_lab.astype(np.int32))
    VC.append(np.full(len(verts), CHARTS.index("frame"), np.int32)); VUV.append(uvs)
    nv += len(verts)

    # --- front and back caps (fan from the centre)
    for which in ("screen", "back"):
        ins, z = (insets[0], z_front) if which == "screen" else (insets[-1], z_back)
        p = xy - ins * nrm
        cv = np.vstack([p, [[0.0, 0.0]]])
        cv3 = np.column_stack([cv, np.full(len(cv), z)])
        uv = np.column_stack([cv[:, 0] + W2 if which == "screen" else W2 - cv[:, 0], L2 - cv[:, 1]])
        faces = _fan(len(p), np.arange(len(p)), up=(which == "screen"))
        add(cv3, faces, which, which, uv)

    # --- camera plateau (frustum loft between outer and inner rounded rectangles) on the back
    cam = spec.camera

    def back_rect(r):  # back-view (x from left, y down from top) -> cad xy
        xb0, xb1 = r["x"]
        yb0, yb1 = r["y"]
        return W2 - xb1, W2 - xb0, L2 - yb1, L2 - yb0

    def back_uv(p):
        return np.column_stack([W2 - p[:, 0], L2 - p[:, 1]])

    if cam:
        o = back_rect(cam["plateau_outer"])
        i = back_rect(cam["plateau_inner"])
        ro = rounded_rect(*o, r=cam["plateau_outer"].get("radius", 5.0), n_corner=24, edge_step=0.5)
        ri = rounded_rect(*i, r=cam["plateau_inner"].get("radius", 4.0), n_corner=24, edge_step=0.5)
        # resample both to the same count by arc length (same start point: top centre, clockwise)
        n = max(len(ro), len(ri))
        ro, ri = _resample_closed(ro, n), _resample_closed(ri, n)
        zt = z_back - cam["plateau_height"]
        verts = np.vstack([np.column_stack([ro, np.full(n, z_back + 0.02)]), np.column_stack([ri, np.full(n, zt)])])
        faces = _ring_faces(2, n, 0, closed=True)
        cap_c = np.array([[ri[:, 0].mean(), ri[:, 1].mean(), zt]])
        verts = np.vstack([verts, cap_c])
        faces = np.vstack([faces, _fan(2 * n, np.arange(n, 2 * n), up=False)])
        add(verts, faces, "camera", "back", back_uv(verts))
        for lens in cam.get("lenses", []):
            cx, cy = W2 - lens["center"][0], L2 - lens["center"][1]
            rr = lens["diameter"] / 2.0
            ang = np.radians(np.linspace(90, -270, 97)[:-1])          # clockwise from the top
            ring = np.column_stack([cx + rr * np.cos(ang), cy + rr * np.sin(ang)])
            zl = z_back - lens["height"]
            verts = np.vstack([np.column_stack([ring, np.full(len(ring), zt + 0.02)]),
                               np.column_stack([ring, np.full(len(ring), zl)]), [[cx, cy, zl]]])
            faces = np.vstack([_ring_faces(2, len(ring), 0, closed=True), _fan(2 * len(ring), np.arange(len(ring), 2 * len(ring)), up=False)])
            add(verts, faces, "lens", "back", back_uv(verts))

    # --- buttons: boxes standing proud of the side (chart frame, s from the outline)
    zmid = (z_front + z_back) / 2.0
    for b in spec.buttons:
        side = b["side"]
        if side not in ("left", "right"):
            continue
        yc = L2 - b["y_from_top"]
        x_out = (W2 + b["protrusion"]) if side == "right" else -(W2 + b["protrusion"])
        x_in = (W2 - 0.6) if side == "right" else -(W2 - 0.6)
        y0, y1 = yc - b["length"] / 2.0, yc + b["length"] / 2.0
        z0, z1 = zmid - b["z_width"] / 2.0, zmid + b["z_width"] / 2.0
        lo = np.array([min(x_in, x_out), y0, z0]); hi = np.array([max(x_in, x_out), y1, z1])
        verts, faces = _box(lo, hi)
        model_tmp_s = _s_of_xy(xy, s, verts[:, :2])
        uv = np.column_stack([model_tmp_s, z_front - verts[:, 2]])
        add(verts, faces, "button", "frame", uv)

    mesh = LabeledMesh(np.concatenate(V), np.concatenate(F), np.concatenate(FL), list(PARTS),
                       {"device": spec.slug, "units": "mm", "source": spec.source})
    return DeviceModel(spec, mesh, np.concatenate(VC), np.concatenate(VUV), xy, s, P, z_front, z_back)


def _s_of_xy(xy: np.ndarray, s: np.ndarray, q: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(q[:, None, :] - xy[None, :, :], axis=2)
    return s[np.argmin(d, axis=1)]


def _resample_closed(p: np.ndarray, n: int) -> np.ndarray:
    q = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    tt = np.linspace(0.0, t[-1], n + 1)[:-1]
    return np.column_stack([np.interp(tt, t, q[:, 0]), np.interp(tt, t, q[:, 1])])


def _box(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return v, f


# ----------------------------------------------------------------------------- placement on the board
def rest_pose(model: DeviceModel, face_up: str = "screen", com: np.ndarray | None = None) -> np.ndarray:
    """``T_world_cad`` of the device lying on the board with ``face_up`` ("screen" or "back") up,
    its top edge towards the board's "top" arrow (+Y) and its plan centre over the board origin.

    The device rests on the convex-hull plane whose support polygon contains the projection of the
    centre of mass (taken at the body centre), so a screen-up phone tilts on its camera lenses and
    the far edge of the back (about 0.7 degrees for the iPhone 12) instead of balancing on the
    lens tops."""
    import trimesh
    from scipy.spatial import ConvexHull

    R0 = np.eye(3) if face_up == "screen" else rot_y(180.0)
    v = model.mesh.vertices @ R0.T
    c = np.zeros(3) if com is None else R0 @ np.asarray(com, float)
    hull = trimesh.convex.convex_hull(v)
    best = None
    down_all = -hull.face_normals[:, 2]
    checked = []
    for i in np.argsort(-down_all):                 # most downward first; the first valid plane wins
        n, f = hull.face_normals[i], hull.faces[i]
        down = float(down_all[i])
        if down < 0.95:
            break
        off = float(n @ hull.vertices[f[0]])
        if any(abs(down - d0) < 1e-9 and abs(off - o0) < 1e-6 for d0, o0 in checked):
            continue
        checked.append((down, off))
        p0 = hull.vertices[f[0]]
        on = np.abs((hull.vertices - p0) @ n) < 1e-6
        pts = hull.vertices[on]
        # 2D frame in the plane
        a = np.cross(n, [1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.cross(n, [0.0, 1.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        q = np.column_stack([(pts - p0) @ a, (pts - p0) @ b])
        cq = np.array([(c - p0) @ a, (c - p0) @ b])
        if len(q) < 3:
            continue
        try:
            ch = ConvexHull(q)
        except Exception:
            continue
        if np.all(ch.equations[:, :2] @ cq + ch.equations[:, 2] <= 1e-9):
            best = (down, n)
            break
    if best is None:
        raise RuntimeError("no stable resting plane found")
    R1 = _rotation_between(best[1], np.array([0.0, 0.0, -1.0]))
    R = R1 @ R0
    v2 = model.mesh.vertices @ R.T
    t = np.array([0.0, 0.0, -v2[:, 2].min()])
    return make_T(R, t)


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        from scipy.spatial.transform import Rotation
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
