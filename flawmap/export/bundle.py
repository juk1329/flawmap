"""The viewer bundle: everything the web viewer needs for one inspected device.

    bundle/
      model.glb            reference model with UVs into the texture atlas (cad frame, metres)
      texture.jpg          atlas of the median appearance (unseen areas hatched grey)
      defects.png          RGBA overlay of defect regions (red) in the same atlas
      idmap.png            defect id per atlas texel (R channel), for picking with the mouse
      defects.json         defects, coverage, device and capture information (see README)
      spin/<pass>_<k>.jpg  keyframes ordered by viewing angle ("실사 360 보기") + spin.json outlines
      evidence/*.jpg       close-ups with a red outline and full-frame thumbnails
      video_<pass>.mp4     H.264 copy of each pass for the browser ("영상에서 보기")

Atlas layout (pixels in a 4096 x 2048 image): the screen chart at 0.1 mm/px at (0, 0), the back
chart at 0.1 mm/px at (760, 0), and the whole frame band in one strip at 0.12 mm/px at (0, 1520),
so no face ever spans two atlas regions. glTF UV (0, 0) is the top-left of the image.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from ..anchor.parametric import CHARTS, DeviceModel
from ..surface.maps import ChartMaps

ATLAS_W, ATLAS_H = 4096, 2048
ATLAS = {"screen": (0, 0, 0.1), "back": (760, 0, 0.1), "frame": (0, 1520, 0.12)}   # x0, y0, mm per px (phones)


def atlas_layout(model: DeviceModel) -> dict[str, tuple[int, int, float]]:
    """Where each chart sits in the 4096 x 2048 atlas: (x0, y0, mm per pixel). Screen and back side
    by side on top, the frame band below. Phones get the fixed layout above (0.1 / 0.12 mm per px);
    a larger device gets coarser pixels so it still fits (a 13-inch tablet: 0.15 mm on the faces;
    the fixed layout overflowed the atlas, 2026-11-17)."""
    fw, fh = model.chart_size("screen")
    bw, bh = model.chart_size("frame")
    r_band = max(0.12, bw / (ATLAS_W - 8))
    band_px = int(np.ceil(bh / r_band))
    r_face = max(0.1, 2 * fw / (ATLAS_W - 45), fh / (ATLAS_H - 53 - band_px - 8))
    r_face = float(np.ceil(r_face / 0.01 - 1e-9) * 0.01)
    if r_face <= 0.1 + 1e-9 and r_band <= 0.12 + 1e-9:
        return dict(ATLAS)
    w_px, h_px = int(np.ceil(fw / r_face)), int(np.ceil(fh / r_face))
    return {"screen": (0, 0, r_face), "back": (w_px + 45, 0, r_face), "frame": (0, h_px + 53, float(r_band))}
UNSEEN_RGB = (150, 150, 150)


def atlas_uv(model: DeviceModel) -> np.ndarray:
    uv = np.zeros((len(model.vuv), 2))
    lay = atlas_layout(model)
    for ci, c in enumerate(CHARTS):
        x0, y0, res = lay[c]
        sel = model.vchart == ci
        uv[sel, 0] = (x0 + model.vuv[sel, 0] / res) / ATLAS_W
        uv[sel, 1] = (y0 + model.vuv[sel, 1] / res) / ATLAS_H
    return uv


def _chart_to_atlas(cm: ChartMaps | None, chart: str, model: DeviceModel, values: np.ndarray | None,
                    fill: np.ndarray, interp=cv2.INTER_AREA) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x0, y0, res = atlas_layout(model)[chart]
    w_mm, h_mm = model.chart_size(chart)
    W, H = int(np.ceil(w_mm / res)), int(np.ceil(h_mm / res))
    if cm is None or values is None:
        img = np.broadcast_to(fill, (H, W) + fill.shape).copy()
    else:
        src = cm.geom.to_image(values, fill=0)
        img = cv2.resize(src.astype(np.float32), (W, H), interpolation=interp)
    return img, (x0, y0, W, H)


def hatch(h: int, w: int, period: int = 14, width: int = 2) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx + yy) % period) < width


def chart_colour(cm: ChartMaps, chart: str, decomp: dict | None, albedo: dict | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Colour of every texel for the bundle: the shading-flattened surface colour of the
    decomposition (``texture.decompose``) where it has samples, else the keyframe median; and the
    reflection strength (mean excess lightness) where known. ``albedo``: chart images already
    flattened (``albedo_images``)."""
    rgb = np.nan_to_num(cm.stats["med_rgb"].astype(np.float32), nan=UNSEEN_RGB[0])
    if not decomp or chart not in decomp:
        return rgb, None
    dec = decomp[chart]
    g = cm.geom
    got = dec.n_bins > 0
    col = dec.diffuse.astype(np.float32)
    if chart in ("screen", "back") and got.any():
        alb = (albedo or {}).get(chart)
        if alb is None:
            alb = albedo_images(cm, {chart: dec})[chart]
        col = alb.reshape(-1, 3)[g.flat_index].astype(np.float32)
    return np.where(got[:, None], col, rgb), dec.spec_mean.astype(np.float32)


def albedo_images(maps_or_cm, decomp: dict) -> dict[str, np.ndarray]:
    """Shading-flattened surface colour (chart image, uint8 RGB) of every face chart with a decomposition."""
    from ..texture.decompose import flatten_shading

    out = {}
    for c, dec in decomp.items():
        if c not in ("screen", "back"):
            continue
        cm = maps_or_cm if isinstance(maps_or_cm, ChartMaps) else maps_or_cm.get(c)
        if cm is None:
            continue
        g = cm.geom
        got = dec.n_bins > 0
        alb, _ = flatten_shading(g.to_image(dec.diffuse), g.valid & g.to_image(got, fill=False), g.res)
        out[c] = alb
    return out


def _region_mask(cm: ChartMaps, chart: str, model: DeviceModel, texels: np.ndarray, grow: int = 2) -> np.ndarray:
    x0, y0, res = atlas_layout(model)[chart]
    uv = cm.geom.uv_of(texels)
    px = np.stack([x0 + uv[:, 0] / res, y0 + uv[:, 1] / res], 1)
    m = np.zeros((ATLAS_H, ATLAS_W), np.uint8)
    q = np.round(px).astype(int)
    m[np.clip(q[:, 1], 0, ATLAS_H - 1), np.clip(q[:, 0], 0, ATLAS_W - 1)] = 255
    return cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=grow)


def build_atlas(maps: dict[str, ChartMaps], model: DeviceModel, defects: list[dict], region_texels: dict,
                good_views: int = 2, decomp: dict | None = None, design: list | None = None,
                albedo: dict | None = None) -> dict[str, np.ndarray]:
    """Texture (surface colour), defect overlays, defect id map and, with a decomposition, the
    reflection-strength map (``specular``: mean excess lightness x 8, 0-255) and a roughness map
    (``roughness``: 0.9 - excess / 25, 0-255) of the atlas.

    The defects are drawn twice: ``overlay`` only outlines them (a white halo and a red line), so the
    real surface stays visible under them; ``overlay_fill`` also fills them (the highlighted view).
    ``design``: [(chart, texels)] of the regions set aside as design, print or stains, outlined in
    blue on ``design_overlay``."""
    tex = np.full((ATLAS_H, ATLAS_W, 3), 40, np.uint8)
    ovl = np.zeros((ATLAS_H, ATLAS_W, 4), np.uint8)
    ovl_fill = np.zeros((ATLAS_H, ATLAS_W, 4), np.uint8)
    dsg = np.zeros((ATLAS_H, ATLAS_W, 4), np.uint8)
    idm = np.zeros((ATLAS_H, ATLAS_W, 3), np.uint8)
    spec = np.zeros((ATLAS_H, ATLAS_W), np.uint8)
    rough = np.full((ATLAS_H, ATLAS_W), int(0.6 * 255), np.uint8)
    for c in CHARTS:
        cm = maps.get(c)
        fill = np.array(UNSEEN_RGB, np.float32)
        if cm is not None:
            rgb, sp = chart_colour(cm, c, decomp, albedo)
            if sp is not None:
                sp_img, (sx0, sy0, sW, sH) = _chart_to_atlas(cm, c, model, np.clip(sp * 8.0, 0, 255)[:, None], np.zeros(1, np.float32))
                spec[sy0:sy0 + sH, sx0:sx0 + sW] = np.clip(sp_img.reshape(sH, sW), 0, 255).astype(np.uint8)
                rg = np.clip(0.9 - sp / 25.0, 0.12, 0.9) * 255
                rg_img, _ = _chart_to_atlas(cm, c, model, rg[:, None], np.zeros(1, np.float32))
                rough[sy0:sy0 + sH, sx0:sx0 + sW] = np.clip(rg_img.reshape(sH, sW), 0, 255).astype(np.uint8)
            seen = (cm.stats["n_good"] >= good_views).astype(np.float32)
            img, (x0, y0, W, H) = _chart_to_atlas(cm, c, model, rgb, fill)
            seen_img, _ = _chart_to_atlas(cm, c, model, seen[:, None], np.zeros(1, np.float32), cv2.INTER_LINEAR)
            valid_img, _ = _chart_to_atlas(cm, c, model, np.ones((len(cm.geom.point), 1), np.float32), np.zeros(1, np.float32), cv2.INTER_LINEAR)
            seen_img = seen_img.reshape(H, W) > 0.5
            unseen = ~seen_img
            hat = hatch(H, W)
            img[unseen] = np.array(UNSEEN_RGB, np.float32)
            img[unseen & hat] = np.array([115, 115, 115], np.float32)
            img[valid_img.reshape(H, W) < 0.5] = np.array(UNSEEN_RGB, np.float32)
        else:
            img, (x0, y0, W, H) = _chart_to_atlas(None, c, model, None, fill)
            img[hatch(H, W)] = np.array([115, 115, 115], np.float32)
        tex[y0:y0 + H, x0:x0 + W] = np.clip(img, 0, 255).astype(np.uint8)
    for d in defects:
        c = d["chart"]
        cm = maps[c]
        m = _region_mask(cm, c, model, region_texels[d["id"]])
        ring = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=3)
        cs, _ = cv2.findContours(ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ovl, cs, -1, (255, 255, 255, 220), 4)
        cv2.drawContours(ovl, cs, -1, (230, 30, 30, 255), 2)
        ovl_fill[m > 0] = (235, 30, 30, 160)
        cv2.drawContours(ovl_fill, cs, -1, (255, 0, 0, 255), 2)
        # generous pick area so a 1 mm chip is clickable
        mp = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=8)
        idm[mp > 0, 0] = d["id"]
    for c, texels in (design or []):
        if c not in maps:
            continue
        m = _region_mask(maps[c], c, model, texels, grow=3)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(dsg, cs, -1, (40, 120, 255, 230), 2)
    return {"texture": tex, "overlay": ovl, "overlay_fill": ovl_fill, "design_overlay": dsg, "idmap": idm, "specular": spec,
            "roughness": rough}


def write_glb(model: DeviceModel, path: Path) -> None:
    """Reference model in metres (three.js convention) with atlas UVs; the texture is loaded
    separately by the viewer so the plain/overlay/id images can be swapped."""
    import trimesh
    from PIL import Image

    uv = atlas_uv(model)
    # trimesh keeps UVs with a bottom-left origin and writes 1 - v into glTF (checked on the exported
    # file, 2026-11-15), so hand it the flipped value to end up with the top-left glTF convention
    uv = np.column_stack([uv[:, 0], 1.0 - uv[:, 1]])
    mesh = trimesh.Trimesh(vertices=model.mesh.vertices / 1000.0, faces=model.mesh.faces, process=False)
    placeholder = Image.new("RGB", (4, 4), (128, 128, 128))
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=placeholder)
    mesh.export(str(path), file_type="glb")


def transcode_for_browser(src: Path, dst: Path, height: int = 720) -> None:
    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [exe, "-y", "-hide_banner", "-loglevel", "error", "-noautorotate", "-i", str(src), "-vf", f"scale=-2:{height}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "veryfast", "-movflags", "+faststart", "-an", str(dst)]
    subprocess.run(cmd, check=True)


def save_png(img: np.ndarray, path: Path) -> None:
    if img.shape[-1] == 4:
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
    else:
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def save_jpg(img: np.ndarray, path: Path, q: int = 90) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
