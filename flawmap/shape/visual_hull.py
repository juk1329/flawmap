"""The object's shape from its silhouettes alone (visual hull), for any object, not only plates.

The plate model only knows rounded plates: a camera bump, an AirPods case or a tumbler need a shape
that comes from the capture. Silhouettes do not care about reflections (glass, metal), so they are
the robust start: every voxel of a box around the object that falls outside the device mask (SAM 2)
in more than ``max_out`` of the views that see it is carved away. What stays is the visual hull: the
object where every view agrees, with concavities filled (they never form a silhouette).

The grid lives in the run's device frame (``T_world_cad`` of a finished run) so the result can be
compared with the model there; the floor (LiDAR height) bounds it from below. Outputs a height map
of the top surface above the model's top face (where a camera bump shows up), the hull's extents and
a mesh (marching cubes).

    python -m flawmap.shape.visual_hull runs/<run> --pass back=<stray folder> [--voxel 0.5]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from ..geometry.transforms import inv_T
from ..utils.io import write_json
from ..utils.logging import get_logger

log = get_logger("flawmap.shape")


def carve(grid_cad: np.ndarray, T_world_cad: np.ndarray, cams: dict, T_world_cam: dict, masks: dict,
          max_out: int = 2, min_views: int = 5, chunk: int = 2_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Occupancy (bool) of grid points (N,3, device frame) and the number of views that saw each."""
    N = len(grid_cad)
    out_votes = np.zeros(N, np.int16)
    seen = np.zeros(N, np.int16)
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        Pw = grid_cad[a:b] @ T_world_cad[:3, :3].T + T_world_cad[:3, 3]
        for i, m in masks.items():
            cam = cams[i]
            Tcw = inv_T(T_world_cam[i])
            pc = Pw @ Tcw[:3, :3].T + Tcw[:3, 3]
            z = pc[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = cam.fx * pc[:, 0] / z + cam.cx
                v = cam.fy * pc[:, 1] / z + cam.cy
            ok = (z > 1) & (u >= 0) & (v >= 0) & (u < cam.width) & (v < cam.height)
            h, w = m.shape
            mu = np.clip(((u[ok] + 0.5) * w / cam.width).astype(int), 0, w - 1)
            mv = np.clip(((v[ok] + 0.5) * h / cam.height).astype(int), 0, h - 1)
            idx = np.flatnonzero(ok)
            seen[a + idx] += 1
            out_votes[a + idx[~m[mv, mu]]] += 1
    return (out_votes <= max_out) & (seen >= min_views), seen


def main(argv: list[str] | None = None) -> int:
    from ..anchor.parametric import DeviceSpec, build_device_model
    from ..anchor.template import template_spec
    from ..io.stray import iter_frames, read_stray
    from ..segment.device_masks import box_of, device_masks, silhouette
    from ..surface.raycast import RaycastScene

    ap = argparse.ArgumentParser(prog="python -m flawmap.shape.visual_hull")
    ap.add_argument("run")
    ap.add_argument("--pass", dest="pas", required=True, help="NAME=STRAY_FOLDER")
    ap.add_argument("--voxel", type=float, default=0.5, help="voxel size in mm")
    ap.add_argument("--margin", type=float, default=12.0, help="grid margin around the model in mm")
    ap.add_argument("--max-out", type=int, default=2, help="views that may call a voxel outside before it is carved")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    run = Path(a.run)
    name, _, root = a.pas.partition("=")
    root = Path(root)
    out = Path(a.out) if a.out else run / "shape" / name
    out.mkdir(parents=True, exist_ok=True)
    rep = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if rep.get("device") == "template":
        est = rep["processing"]["device_estimate"]
        model = build_device_model(template_spec(*est["size_mm"], est["corner_radius"], edge_radius=est.get("edge_radius")))
    else:
        model = build_device_model(DeviceSpec.by_slug(rep["device"]))
    pj = json.loads((run / "passes" / name / "poses.json").read_text(encoding="utf-8"))
    Twc = {int(k): np.array(v["T_world_cam"]) for k, v in pj["poses"].items()}
    Tcad = np.array(json.loads((run / "passes" / name / "registration.json").read_text(encoding="utf-8"))["T_world_cad"])
    floor_z = (pj.get("arkit_residual") or {}).get("floor_z")
    cap = read_stray(root)
    ids = sorted(Twc)
    t0 = time.time()
    frames = dict(iter_frames(root / "rgb.mp4", ids))
    cams = {i: cap.camera(i) for i in ids}
    # SAM 2, prompted with the registered model on six keyframes
    sc = RaycastScene(model.mesh, None)
    pids = [ids[int(k)] for k in np.linspace(0, len(ids) - 1, 6).round()]
    boxes = {}
    for i in pids:
        b = box_of(silhouette(sc, cams[i], inv_T(Tcad) @ Twc[i], 0.25))
        boxes[i] = None if b is None else b / 0.25
    masks = device_masks(frames, boxes, log=lambda m: log.info(f"[{name}]{m}"))
    del frames
    # grid in the device frame; up = the direction of world +z in the device frame
    up = Tcad[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    k_up = int(np.argmax(np.abs(up)))
    sgn = np.sign(up[k_up])
    L, W, T = model.spec.length, model.spec.width, model.spec.thickness
    z_top_face = sgn * T / 2.0                                   # the model's upper face, device frame
    if floor_z is not None:
        z_floor = (inv_T(Tcad) @ np.array([0, 0, floor_z, 1.0]))[2]
    else:
        z_floor = -sgn * T / 2.0
    z_lo, z_hi = sorted([z_floor + sgn * 0.4, z_top_face + sgn * 12.0])
    v = a.voxel
    xs = np.arange(-W / 2 - a.margin, W / 2 + a.margin, v) + v / 2
    ys = np.arange(-L / 2 - a.margin, L / 2 + a.margin, v) + v / 2
    zs = np.arange(z_lo, z_hi, v) + v / 2
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    log.info(f"[{name}] grid {len(xs)} x {len(ys)} x {len(zs)} = {len(grid) / 1e6:.1f} M voxels at {v} mm; floor at z {z_floor:.2f}, "
             f"model top face at z {z_top_face:.2f} (device frame)")
    occ, seen = carve(grid, Tcad, cams, Twc, masks, max_out=a.max_out)
    occ = occ.reshape(X.shape)
    # top surface: per column, the highest occupied voxel above the floor, as height over the model's top face
    zz = zs if sgn > 0 else -zs
    colmax = np.where(occ.any(2), (np.where(occ, zz[None, None, :], -np.inf)).max(2), np.nan)
    height = colmax - (z_top_face * sgn) + v / 2                 # mm above the model's top face (+ = higher)
    foot = occ.any(2)
    ext_x = xs[foot.any(1)]
    ext_y = ys[foot.any(0)]
    hull_L, hull_W = float(ext_y.max() - ext_y.min() + v), float(ext_x.max() - ext_x.min() + v)
    top_from_floor = float(np.nanmedian(colmax[foot])) + v / 2 - sgn * z_floor          # both in the "up" coordinate
    inner = foot.copy()
    inner[:int(20 / v)], inner[-int(20 / v):], inner[:, :int(20 / v)], inner[:, -int(20 / v):] = False, False, False, False
    face_h = float(np.nanmedian(height[inner])) if inner.any() else float("nan")
    # camera corner: the 45 mm corner with the highest heights
    q = int(45 / v)
    corners = {"x-y-": height[:q, :q], "x-y+": height[:q, -q:], "x+y-": height[-q:, :q], "x+y+": height[-q:, -q:]}
    cstat = {k: float(np.nanpercentile(c[np.isfinite(c)], 90)) if np.isfinite(c).any() else None for k, c in corners.items()}
    best_c = max((k for k in cstat if cstat[k] is not None), key=lambda k: cstat[k])
    # height map image (device seen from above): blue = at the model's top face, red = 3 mm above
    hm = np.clip((np.nan_to_num(height, nan=-99) + 1.0) / 4.0, 0, 1)
    img = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~foot] = (255, 255, 255)
    img = np.ascontiguousarray(np.rot90(img))                   # x right, y up
    cv2.imwrite(str(out / "hull_height.png"), cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    # mesh
    try:
        from skimage.measure import marching_cubes
        import trimesh

        vol = np.pad(occ.astype(np.float32), 1)
        verts, faces, _, _ = marching_cubes(vol, 0.5, spacing=(v, v, v))
        verts = verts - v + np.array([xs[0], ys[0], zs[0]])
        trimesh.Trimesh(verts, faces, process=False).export(out / "hull.ply")
        n_faces = len(faces)
    except Exception as e:  # pragma: no cover
        log.info(f"  mesh failed: {e}")
        n_faces = 0
    info = {"pass": name, "voxel_mm": v, "max_out": a.max_out, "views": len(masks),
            "hull_length_mm": round(hull_L, 2), "hull_width_mm": round(hull_W, 2),
            "model_length_mm": L, "model_width_mm": W, "model_thickness_mm": T,
            "top_above_floor_mm_p50": round(abs(top_from_floor), 2),
            "face_height_over_model_top_mm_p50": round(face_h, 2),
            "corner_height_p90_mm": {k: (round(x, 2) if x is not None else None) for k, x in cstat.items()},
            "highest_corner": best_c, "mesh_faces": n_faces, "seconds": round(time.time() - t0, 1)}
    write_json(out / "hull.json", info)
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
