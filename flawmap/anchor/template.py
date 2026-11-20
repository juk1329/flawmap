"""A plate-shaped model fitted to the capture, for devices without a reference drawing.

The detectors never use the reference model's colours, only its shape as a coordinate system
(charts, mm, parts). For phones and tablets that shape is close to a rounded plate, so a plate whose
length, width, thickness and corner radius are fitted to the capture's outline can stand in for the
drawing-based model. What is lost: the camera bump, buttons and openings are unknown (no exclusion
zones, no part names for them), and without metric cameras the size is only assumed.

* ``template_spec``  - a ``DeviceSpec`` for a rounded plate with flat sides and small chamfers
* ``fit_dimensions`` - edge alignment (as ``register.register_device``) with three extra unknowns, the
  log-scales of width, length and thickness about the device centre; the model is rebuilt with
  the new size after every round, then the corner radius is chosen by a 1-D search
* ``identify``       - the drawing-based model whose length and width match the fitted plate
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from ..geometry.transforms import inv_T, se3_exp
from ..surface.raycast import RaycastScene
from .parametric import DeviceSpec, build_device_model
from .register import _grad, _project, mask_fields, match_edges, silhouette_points

GENERIC_PHONE = {"length": 146.7, "width": 71.5, "thickness": 7.6, "corner_radius": 10.0}


def template_spec(length: float, width: float, thickness: float, corner_radius: float = 10.0, name: str | None = None,
                  edge_radius: float | None = None) -> DeviceSpec:
    """A rounded plate. ``edge_radius``: the side profile's rounding where the faces meet the flat
    side (a quarter circle of that radius, mm; at most half the thickness); None keeps the small
    default chamfer (``fit_dimensions_multi`` chooses the radius from the silhouettes)."""
    r = float(np.clip(corner_radius, 1.0, min(width, length) / 2 - 1.0))
    th = np.radians(np.linspace(0.0, 90.0, 9))
    corner = [[float(r - r * np.cos(t)), float(r - r * np.sin(t))] for t in th]      # (inset from side, inset from top)
    h = thickness / 2.0
    if edge_radius is None:
        c = min(0.6, 0.12 * thickness)
        profile = [[1.0, h, "screen"], [0.45, h - 0.05, "frame"], [0.12, h - c * 0.6, "frame"], [0.0, h - c, "frame"],
                   [0.0, -(h - c), "frame"], [0.12, -(h - c * 0.6), "frame"], [0.45, -(h - 0.05), "back"], [1.0, -h, "back"]]
    else:
        e = float(np.clip(edge_radius, 0.1, h - 0.05))
        ts = np.radians([90.0, 60.0, 30.0, 0.0])
        top = [[e * (1 - np.sin(t)) + (0.05 if k == 0 else 0.0), h - e * (1 - np.cos(t)), "screen" if k == 0 else "frame"]
               for k, t in enumerate(ts)]
        top = [[float(a), float(b), lab] for a, b, lab in top]
        bot = [[a, -b, "back" if lab == "screen" else "frame"] for a, b, lab in reversed(top)]
        profile = [[max(e + 0.4, 1.0), h, "screen"]] + top + bot + [[max(e + 0.4, 1.0), -h, "back"]]
    nm = name or f"추정 판형 모델 {length:.1f} x {width:.1f} x {thickness:.1f} mm"
    return DeviceSpec(name=nm, slug="template", length=float(length), width=float(width), thickness=float(thickness),
                      corner=corner, side_profile=profile, camera={}, buttons=[], openings=[],
                      default_color={"frame": [60, 62, 66], "back": [70, 72, 76], "screen": [14, 14, 16], "camera": [40, 40, 44],
                                     "lens": [10, 10, 12], "button": [60, 62, 66]},
                      source="fitted to the capture (no reference drawing)",
                      raw={"template": True, "corner_radius": r, "edge_radius": edge_radius})


def fit_dimensions(spec: DeviceSpec, frames: dict[int, np.ndarray], cams: dict, T_world_cam: dict[int, np.ndarray],
                   T_world_cad: np.ndarray, views: list[int], searches=(10.0, 6.0, 4.0, 3.0, 2.5), fit_thickness: bool = True,
                   max_points_per_view: int = 500, huber_px: float = 1.0, radii=None, log=print):
    """Pose and plate size from image edges with cameras fixed. Returns (spec, T_world_cad, info).
    ``radii``: corner radii tried at the end; default 6-12 mm for phone-sized plates, and 0.6-1.5x
    the starting radius for larger ones (a 13-inch tablet's corner is ~14 mm)."""
    L, W, T = spec.length, spec.width, spec.thickness
    r = spec.raw.get("corner_radius", GENERIC_PHONE["corner_radius"])
    if radii is None:
        radii = (6.0, 8.0, 10.0, 12.0) if r <= 12.0 else tuple(round(r * k, 1) for k in (0.6, 0.8, 1.0, 1.25, 1.5))
    grads = {i: _grad(frames[i]) for i in views}
    rng = np.random.default_rng(0)
    rounds = []
    Tw = T_world_cad.copy()

    def collect(spec_, Tw_, S):
        model = build_device_model(spec_, corner_step=0.5, edge_step=2.0, profile_step=0.3)
        scene = RaycastScene(model.mesh, None)
        data = []
        for i in views:
            X, n2 = silhouette_points(scene, cams[i], inv_T(Tw_) @ T_world_cam[i])
            if len(X) == 0:
                continue
            if len(X) > max_points_per_view:
                sel = rng.choice(len(X), max_points_per_view, replace=False)
                X, n2 = X[sel], n2[sel]
            p, _ = _project(cams[i], inv_T(T_world_cam[i]), X @ Tw_[:3, :3].T + Tw_[:3, 3])
            tgt, _ = match_edges(*grads[i], p, n2, S)
            ok = np.isfinite(tgt[:, 0])
            if ok.sum() >= 20:
                data.append((i, X[ok], n2[ok], tgt[ok], inv_T(T_world_cam[i])))
        return data

    fit_T = [bool(fit_thickness)]

    def make_res(data, Tw_, with_scale):
        def res(x):
            xi = x[:6]
            sc = np.exp(np.r_[x[6:9], 0.0][:3]) if with_scale else np.ones(3)
            if with_scale and not fit_T[0]:
                sc[2] = 1.0
            Tn = Tw_ @ se3_exp(xi)            # update in the device frame (rotation about the device centre)
            out = []
            for i, X, n2, tgt, Tcw in data:
                p, _ = _project(cams[i], Tcw, (X * sc) @ Tn[:3, :3].T + Tn[:3, 3])
                out.append(np.einsum("ij,ij->i", p - tgt, n2))
            return np.concatenate(out)
        return res

    for rnd, S in enumerate(searches):
        cur = template_spec(L, W, T, r, spec.name if not spec.raw.get("template") else None)
        data = collect(cur, Tw, S)
        if not data:
            break
        res = make_res(data, Tw, True)
        x0 = np.zeros(9)
        r0 = res(x0)
        sol = least_squares(res, x0, loss="huber", f_scale=huber_px, max_nfev=60,
                            x_scale=[1e-3] * 3 + [0.1] * 3 + [3e-3] * 3)
        x = sol.x
        Tw = Tw @ se3_exp(x[:6])
        sc = np.exp(x[6:9])
        W, L = W * sc[0], L * sc[1]
        if fit_T[0]:
            # as in fit_dimensions_multi: out of half to twice the start, the start is kept
            T_new = T * sc[2]
            if 0.5 * spec.thickness < T_new < 2.0 * spec.thickness:
                T = float(T_new)
            else:
                T, fit_T[0] = float(spec.thickness), False
        r1 = res(x)
        rounds.append({"round": rnd, "search_px": S, "n_points": int(len(r0)), "median_px_before": float(np.median(np.abs(r0))),
                       "median_px_after": float(np.median(np.abs(r1))), "size_mm": [round(L, 3), round(W, 3), round(T, 3)]})
        log(f"  template fit round {rnd}: {len(r0)} edge points, |r| {rounds[-1]['median_px_before']:.2f} -> "
            f"{rounds[-1]['median_px_after']:.2f} px, size {L:.2f} x {W:.2f} x {T:.2f}")
    # corner radius: the radius whose model leaves the smallest residual (pose refit for each)
    best = None
    for rr in radii:
        cur = template_spec(L, W, T, rr)
        data = collect(cur, Tw, searches[-1])
        if not data:
            continue
        res = make_res(data, Tw, False)
        sol = least_squares(res, np.zeros(6), loss="huber", f_scale=huber_px, max_nfev=30, x_scale=[1e-3] * 3 + [0.1] * 3)
        med = float(np.median(np.abs(res(sol.x))))
        if best is None or med < best[0]:
            best = (med, rr, Tw @ se3_exp(sol.x))
    if best is not None:
        r, Tw = best[1], best[2]
    out = template_spec(L, W, T, r)
    info = {"rounds": rounds, "size_mm": [L, W, T], "corner_radius": r, "median_px": best[0] if best else None}
    log(f"  template: {L:.2f} x {W:.2f} x {T:.2f}, corner radius {r:.0f}, residual {info['median_px']}")
    return out, Tw, info


def fit_dimensions_multi(spec: DeviceSpec, passes: list[dict], searches=(10.0, 6.0, 4.0, 3.0, 2.5), fit_thickness: bool = True,
                         max_points_per_view: int = 400, huber_px: float = 1.0, radii=None, mask_weight: float = 0.0,
                         mask_points_per_view: int = 120, mask_until_px: float = 6.0, edge_radii=(None, 0.5, 1.0, 1.5, 2.0), log=print, thickness_measured: bool = True):
    """``fit_dimensions`` over several passes of the same device: one plate size (length, width,
    thickness, corner radius) shared by all, one pose per pass. ``passes``: dicts with ``frames``,
    ``cams``, ``T_world_cam``, ``T_world_cad``, ``views`` and optionally ``masks``. The masks choose
    the shape (corner and edge radius, ``shape_score``); the size itself is fitted to the edges only
    by default (``mask_weight`` 0): SAM 2's masks run about a pixel wide, and with them in the first
    rounds the synthetic iPhone 12 came out 148.4 x 73.6 mm against 147.6 x 72.3 mm without them from
    the same start (146.7 x 71.5 true), so it was no longer recognised (2026-11-17). With
    ``mask_weight`` > 0 they join while the window is at least ``mask_until_px``. Returns (spec,
    [T_world_cad per pass], info).

    Two passes see the device from both faces: the screen pass's outline is the front edge and the
    back pass's the rear edge, so the thickness and the width/length are tied by both sides at once
    (2026-11-17: the iPad's black screen gives no texture to measure the top face, the silver back
    does)."""
    L, W, T = spec.length, spec.width, spec.thickness
    r = spec.raw.get("corner_radius", GENERIC_PHONE["corner_radius"])
    if radii is None:
        radii = (6.0, 8.0, 10.0, 12.0) if r <= 12.0 else tuple(round(r * k, 1) for k in (0.6, 0.8, 1.0, 1.25, 1.5))
    P = len(passes)
    grads = [{i: _grad(ps["frames"][i]) for i in ps["views"]} for ps in passes]
    fields = [mask_fields(ps.get("masks"), ps["cams"], ps["views"]) for ps in passes]
    Tws = [ps["T_world_cad"].copy() for ps in passes]
    rng = np.random.default_rng(0)
    rounds = []

    def collect(spec_, Tws_, S, with_mask):
        model = build_device_model(spec_, corner_step=0.5, edge_step=2.0, profile_step=0.3)
        scene = RaycastScene(model.mesh, None)
        data, mdata = [], []
        for k, ps in enumerate(passes):
            for i in ps["views"]:
                Twc = ps["T_world_cam"][i]
                X, n2 = silhouette_points(scene, ps["cams"][i], inv_T(Tws_[k]) @ Twc)
                if len(X) == 0:
                    continue
                if len(X) > max_points_per_view:
                    sel = rng.choice(len(X), max_points_per_view, replace=False)
                    X, n2 = X[sel], n2[sel]
                p, _ = _project(ps["cams"][i], inv_T(Twc), X @ Tws_[k][:3, :3].T + Tws_[k][:3, 3])
                tgt, _ = match_edges(*grads[k][i], p, n2, S)
                ok = np.isfinite(tgt[:, 0])
                if ok.sum() >= 20:
                    data.append((k, i, X[ok], n2[ok], tgt[ok], inv_T(Twc)))
                if with_mask and i in fields[k]:
                    _, inb = fields[k][i](p)
                    Xm = X[inb]
                    if len(Xm) > mask_points_per_view:
                        Xm = Xm[rng.choice(len(Xm), mask_points_per_view, replace=False)]
                    if len(Xm):
                        mdata.append((k, i, Xm, inv_T(Twc)))
        return data, mdata

    def make_res(data, mdata, Tws_, with_scale, wm):
        def res(x):
            sc = np.exp(x[6 * P:6 * P + 3]) if with_scale else np.ones(3)
            if with_scale and not fit_T[0]:
                sc = np.array([sc[0], sc[1], 1.0])
            Tn = [Tws_[k] @ se3_exp(x[6 * k:6 * k + 6]) for k in range(P)]
            out = []
            for k, i, X, n2, tgt, Tcw in data:
                p, _ = _project(passes[k]["cams"][i], Tcw, (X * sc) @ Tn[k][:3, :3].T + Tn[k][:3, 3])
                out.append(np.einsum("ij,ij->i", p - tgt, n2))
            for k, i, Xm, Tcw in mdata:
                p, _ = _project(passes[k]["cams"][i], Tcw, (Xm * sc) @ Tn[k][:3, :3].T + Tn[k][:3, 3])
                out.append(wm * fields[k][i](p)[0])
            return np.concatenate(out)
        return res

    xs_pose = [1e-3] * 3 + [0.1] * 3
    edge = spec.raw.get("edge_radius")
    T_start = float(spec.thickness)
    thickness_clamped = [False]
    fit_T = [bool(fit_thickness)]

    def size_rounds(srch, L, W, T, r, edge, Tws):
        for rnd, S in enumerate(srch):
            cur = template_spec(L, W, T, r, edge_radius=edge)
            use_mask = mask_weight > 0 and S >= mask_until_px and any(fields)
            wm = mask_weight * min(1.0, S / 6.0) if use_mask else 0.0
            data, mdata = collect(cur, Tws, S, use_mask)
            if not data:
                break
            res = make_res(data, mdata, Tws, True, wm)
            x0 = np.zeros(6 * P + 3)
            n_e = sum(len(d[2]) for d in data)
            r0 = res(x0)[:n_e]
            sol = least_squares(res, x0, loss="huber", f_scale=huber_px, max_nfev=60, x_scale=xs_pose * P + [3e-3] * 3)
            x = sol.x
            Tws = [Tws[k] @ se3_exp(x[6 * k:6 * k + 6]) for k in range(P)]
            sc = np.exp(x[6 * P:6 * P + 3])
            W, L = W * sc[0], L * sc[1]
            if fit_T[0]:
                # the outline barely constrains the thickness when the side band is never seen (a capture
                # filmed from above only, cc53da7c23, 2026-11-18: it ran to 0 mm and registration failed).
                # When it runs out of half to twice the start, the start is kept for the rest of the fit:
                # it is the capture's own measurement (the top face's height above the floor; cc53: 4.84 mm,
                # where the outline went to the 2.42 mm bound)
                # Without a measured start (a generic plate: the device stood or leaned, nothing measured its
                # top face's height), the last thickness inside the band is kept instead (6c417d24a0: 7.6 ->
                # 6.1 ... 4.2 mm, then out with a 0.5 mm edge radius; the iPad is 5.1 mm)
                T_new = T * sc[2]
                if not (0.5 * T_start < T_new < 2.0 * T_start):
                    thickness_clamped[0], fit_T[0] = True, False
                    T = T_start if thickness_measured else T
                else:
                    T = float(T_new)
            r1 = res(x)[:n_e]
            rounds.append({"round": len(rounds), "search_px": S, "n_points": int(n_e), "mask_points": int(sum(len(d[2]) for d in mdata)),
                           "mask_weight": round(wm, 3), "median_px_before": float(np.median(np.abs(r0))),
                           "median_px_after": float(np.median(np.abs(r1))), "size_mm": [round(L, 3), round(W, 3), round(T, 3)],
                           "corner_radius": r, "edge_radius": edge})
            log(f"  template fit round {rounds[-1]['round']} ({P} passes): {n_e} edge points"
                + (f" + {rounds[-1]['mask_points']} mask points" if mdata else "")
                + f", |r| {rounds[-1]['median_px_before']:.2f} -> {rounds[-1]['median_px_after']:.2f} px, size {L:.2f} x {W:.2f} x {T:.2f}"
                + f" (corner r {r:.1f})")
        return L, W, T, Tws

    def shape_score(spec_, Tws_):
        """Pose refit and score of one plate shape: median edge residual, plus half the median
        distance of the silhouette to the device masks when there are masks. Edges alone cannot
        choose a corner radius: where the radius is wrong the corner's points find no edge inside
        the 2.5 px window and simply drop out (the iPad's corners came out at 8.7 mm against
        13-15 mm in the masks' footprint, and the model's corners lay on the floor, 2026-11-17)."""
        data, mdata = collect(spec_, Tws_, searches[-1], any(fields))
        if not data:
            return None
        res = make_res(data, [], Tws_, False, 0.0)
        sol = least_squares(res, np.zeros(6 * P + 3), loss="huber", f_scale=huber_px, max_nfev=30, x_scale=xs_pose * P + [1.0] * 3)
        med = float(np.median(np.abs(res(sol.x))))
        Tn = [Tws_[k] @ se3_exp(sol.x[6 * k:6 * k + 6]) for k in range(P)]
        mm = None
        if mdata:
            vals = []
            for k, i, Xm, Tcw in mdata:
                pp, _ = _project(passes[k]["cams"][i], Tcw, Xm @ Tn[k][:3, :3].T + Tn[k][:3, 3])
                d, ok = fields[k][i](pp)
                vals.append(np.abs(d[ok]))
            mm = float(np.median(np.concatenate(vals))) if vals else None
        return med + (0.5 * mm if mm is not None else 0.0), med, mm, Tn

    # size and shape in turns: the size rounds fit the outline with the current corner radius, and a
    # wrong radius makes them compensate in length and width (the synthetic iPhone 12 started from the
    # footprint's 14.8 mm corners and ended at 148.4 x 73.7 mm, true 146.7 x 71.5, 2026-11-17)
    radius_scores, best, best_e = [], None, None
    for it in range(2):
        L, W, T, Tws = size_rounds(searches if it == 0 else tuple(x for x in searches if x <= 6.0), L, W, T, r, edge, Tws)
        cands = radii if it == 0 else tuple(sorted({round(r * k, 1) for k in (0.85, 1.0, 1.15)}))
        best = None
        for rr in cands:
            sc = shape_score(template_spec(L, W, T, rr, edge_radius=edge), Tws)
            if sc is None:
                continue
            radius_scores.append({"iteration": it, "radius": rr, "edge_px": round(sc[1], 3), "mask_px": None if sc[2] is None else round(sc[2], 2)})
            if best is None or sc[0] < best[0]:
                best = (sc[0], rr, sc[3], sc[1])
        if best is not None:
            r, Tws = best[1], best[2]
        # the side profile's rounding: seen from low, a rounded edge moves the silhouette inwards
        best_e = None
        for e in edge_radii:
            if e is not None and e >= T / 2 - 0.1:
                continue
            sc = shape_score(template_spec(L, W, T, r, edge_radius=e), Tws)
            if sc is None:
                continue
            if best_e is None or sc[0] < best_e[0] - 0.005:
                best_e = (sc[0], e, sc[3], sc[1])
        if best_e is not None:
            edge, Tws = best_e[1], best_e[2]
        log(f"  template shape {it}: corner radius {r:.1f}, edge radius {edge}, size {L:.2f} x {W:.2f} x {T:.2f}")
    out = template_spec(L, W, T, r, edge_radius=edge)
    info = {"rounds": rounds, "size_mm": [L, W, T], "corner_radius": r, "edge_radius": edge, "radius_scores": radius_scores,
            "median_px": best_e[3] if best_e else (best[3] if best else None), "passes": P,
            "thickness_from_outline": not thickness_clamped[0]}
    if thickness_clamped[0]:
        log(f"  template: the outline did not fix the thickness (left half to twice the start {T_start:.2f} mm); kept "
            + (f"the measured start" if thickness_measured else f"the last value inside, {T:.2f} mm"))
    log(f"  template ({P} passes): {L:.2f} x {W:.2f} x {T:.2f}, corner radius {r:.1f}, edge radius {edge}, residual {info['median_px']}")
    return out, Tws, info



def model_of_run(rep: dict):
    """The device model a finished run used: the drawing-based one by its slug, or the plate the run
    measured (``processing.device_estimate``) when no drawing matched."""
    from .parametric import DeviceSpec, build_device_model

    if rep.get("device") == "template":
        est = rep.get("processing", {}).get("device_estimate")
        if est is None:       # a public video (pipeline.prepare_uco3d_pass): the fit scaled to the assumed width
            t = rep["passes"][0]["template"]
            k = t["assumed_width_mm"] / t["size_mm"][1]
            return build_device_model(template_spec(t["size_mm"][0] * k, t["assumed_width_mm"], t["size_mm"][2] * k, t["corner_radius"] * k))
        return build_device_model(template_spec(*est["size_mm"], est["corner_radius"], edge_radius=est.get("edge_radius")))
    return build_device_model(DeviceSpec.by_slug(rep["device"]))


def identify(length: float, width: float, tol_mm: float = 1.0, scale_tol: float = 0.01, ratio_tol: float = 0.004) -> DeviceSpec | None:
    """The drawing-based device whose length and width are both within ``tol_mm`` of the fit, or,
    failing that, whose length and width differ from the fit by one common scale (within
    ``ratio_tol`` of each other and ``scale_tol`` of 1).

    The second rule is for ARKit's metric scale, which is off by 0.5-0.9 % per recording: the two
    iPad recordings of 2026-11-17 gave the same device as 280.2 x 214.9 and 279.0 x 213.5 mm on
    their own (Apple: 281.6 x 215.5). A common scale error keeps the length/width ratio; another
    model of a similar size does not (iPhone 11 vs 12: 2.9 % and 5.9 %)."""
    from ..utils.io import repo_root

    best = None
    for p in sorted((repo_root() / "configs" / "devices").glob("*.yaml")):
        try:
            s = DeviceSpec.load(p)
        except Exception:
            continue
        d = max(abs(s.length - length), abs(s.width - width))
        kl, kw = length / s.length, width / s.width
        by_scale = abs(kl - kw) <= ratio_tol and abs(0.5 * (kl + kw) - 1.0) <= scale_tol
        if d <= tol_mm or by_scale:
            key = d if d <= tol_mm else 100.0 + abs(0.5 * (kl + kw) - 1.0) * 1000.0
            if best is None or key < best[0]:
                best = (key, s)
    return best[1] if best else None


def plate_box(points: np.ndarray, cam_centres: np.ndarray, face_up: str = "screen") -> tuple[np.ndarray, np.ndarray]:
    """Initial ``T_world_cad`` and (length, width, thickness) of a plate from object points.

    The face plane is fitted robustly (SVD, points beyond 3 MAD of the plane dropped, 4 rounds), and
    length and width come from the smallest rectangle around the in-plane points (2 % farthest from the
    centre dropped). Principal axes of all points were used first; on uCO3D 29776-61880-12178 the
    segmented cloud carries off-plane noise, the longest axis came out across the phone (length/width
    0.57) and the fit settled on a 40 x 71 mm plate. Cad z (screen normal) points to the cameras when
    the screen is up."""
    import cv2

    c = np.median(points, 0)
    q = points - c
    if len(q) > 60000:
        q = q[np.random.default_rng(0).choice(len(q), 60000, replace=False)]
    keep = np.ones(len(q), bool)
    cc = np.zeros(3)
    for _ in range(4):
        cc = np.median(q[keep], 0)
        _, _, Vt = np.linalg.svd(q[keep] - cc, full_matrices=False)
        d = (q - cc) @ Vt[2]
        mad = np.median(np.abs(d[keep])) + 1e-12
        keep = np.abs(d) < 3.0 * mad
    a1, a2, e_n = Vt[0], Vt[1], Vt[2]
    uv = np.stack([(q - cc) @ a1, (q - cc) @ a2], 1)[keep]
    r = np.linalg.norm(uv - np.median(uv, 0), axis=1)
    uv = uv[r <= np.percentile(r, 98)]
    (rx, ry), (w, h), ang = cv2.minAreaRect(uv.astype(np.float32))
    t = np.radians(ang)
    dir_w = np.cos(t) * a1 + np.sin(t) * a2           # direction of the rectangle side with size w
    dir_h = -np.sin(t) * a1 + np.cos(t) * a2
    e_len, L, W = (dir_w, w, h) if w >= h else (dir_h, h, w)
    toward = (cam_centres - (c + cc)).mean(0)
    if (e_n @ toward > 0) != (face_up == "screen"):
        e_n = -e_n
    e_wid = np.cross(e_len, e_n)                       # x = y cross z keeps the frame right-handed
    R = np.column_stack([e_wid, e_len, e_n])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c + cc + rx * a1 + ry * a2
    thick = float(2.0 * np.percentile(np.abs(d[keep]), 98))
    return T, np.array([float(L), float(W), thick])


def axes_box(points: np.ndarray, cam_centres: np.ndarray, face_up: str = "screen") -> tuple[np.ndarray, np.ndarray]:
    """The first initialisation: principal axes of all points, 1st-99th percentile extents. Kept as a
    second start next to ``plate_box`` (on uCO3D 59015-27660-21826 it gave the better plate)."""
    c = np.median(points, 0)
    q = points - c
    _, _, Vt = np.linalg.svd(q[np.random.default_rng(0).choice(len(q), min(len(q), 20000), replace=False)], full_matrices=False)
    e_len, e_n = Vt[0], Vt[2]
    toward = (cam_centres - c).mean(0)
    if (e_n @ toward > 0) != (face_up == "screen"):
        e_n = -e_n
    e_wid = np.cross(e_len, e_n)
    R = np.column_stack([e_wid, e_len, e_n])
    proj = q @ R
    lo, hi = np.percentile(proj, 1, axis=0), np.percentile(proj, 99, axis=0)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c + R @ ((lo + hi) / 2)
    size = hi - lo
    return T, np.array([size[1], size[0], size[2]])


def mask_iou(model, T_world_cad: np.ndarray, cams: dict, T_world_cam: dict, masks: dict, scale: float = 0.25) -> float:
    """Median IoU between the model's silhouette and given object masks (bool, full resolution)."""
    import cv2

    scene = RaycastScene(model.mesh, None)
    vals = []
    for i, m in masks.items():
        h = scene.cast(cams[i].scaled(scale), inv_T(T_world_cad) @ T_world_cam[i], step=1)
        mm = cv2.resize(m.astype(np.uint8), (h.hit.shape[1], h.hit.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        u = (h.hit | mm).sum()
        if u:
            vals.append((h.hit & mm).sum() / u)
    return float(np.median(vals)) if vals else 0.0


pca_box = plate_box     # the first name, kept for callers
