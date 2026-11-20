"""A contact sheet per recording: one row per candidate, with a held-out frame and its render, the
middle of the device in both at twice the size (print, logo, reflections), and the same four stress
views (a turn at 60 degrees, grazing, the unfilmed side, a low ring), so the candidates can be judged
side by side at a glance (``flawmap compare-sheet``)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

STRESS = ["ring60_el60_az0.jpg", "graze_el8_az60.jpg", "under_el-40_az0.jpg", "ring10_el10_az90.jpg"]


def write_sheet(rec_dir: str | Path, out: str | Path | None = None, frame: int | None = None, cell: tuple[int, int] = (320, 240)) -> Path:
    rec_dir = Path(rec_dir)
    idx = json.loads((rec_dir / "index.json").read_text())
    cands = idx.get("candidates", [])
    W, H = cell
    rows = []
    # the held-out frame shown: the median-PSNR frame of the first candidate, or the one asked for
    for c in cands:
        hdir = rec_dir / c["id"] / "renders" / "heldout"
        files = sorted(hdir.glob("*.jpg")) if hdir.exists() else []
        if not files:
            continue
        if frame is None:
            m = json.loads((rec_dir / c["id"] / "metrics.json").read_text())
            pf = sorted(m["heldout"].get("per_frame", []), key=lambda r: r["psnr_g"])
            frame = pf[len(pf) // 2]["frame"] if pf else int(files[len(files) // 2].stem)
        break
    box = None
    for c in cands:
        d = rec_dir / c["id"] / "renders"
        pair = cv2.imread(str(d / "heldout" / f"{frame:05d}.jpg"))
        tiles = []
        if pair is not None:
            h, w = pair.shape[:2]
            real, ren = pair[:, : w // 2], pair[:, w // 2:]
            if box is None:
                box = _detail_box(ren, cell)
            tiles += [cv2.resize(real, cell), cv2.resize(ren, cell)]
            if box is not None:
                y0, y1, x0, x1 = box
                tiles += [cv2.resize(real[y0:y1, x0:x1], cell, interpolation=cv2.INTER_AREA),
                          cv2.resize(ren[y0:y1, x0:x1], cell, interpolation=cv2.INTER_AREA)]
            else:
                tiles += [np.full((H, W, 3), 235, np.uint8)] * 2
        else:
            tiles += [np.full((H, W, 3), 235, np.uint8)] * 4
        for n in STRESS:
            im = cv2.imread(str(d / "stress" / n))
            tiles.append(cv2.resize(im, cell) if im is not None else np.full((H, W, 3), 235, np.uint8))
        row = np.concatenate(tiles, 1)
        s = c.get("summary", {})
        def f(k, nd):
            v = s.get(k)
            return "-" if v is None else f"{v:.{nd}f}"

        txt = (f"{c['id']}  PSNR {f('psnr_g', 2)}  detail {f('hf_ncc', 2)}  LPIPS {f('lpips_g', 3)}  "
               f"float {f('turn30_float', 2)}  hole {f('under_hole', 2)}")
        cv2.rectangle(row, (0, 0), (W * 3, 26), (255, 255, 255), -1)
        cv2.putText(row, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        rows.append(row)
    head = np.full((30, rows[0].shape[1], 3), 255, np.uint8) if rows else np.full((30, 8 * W, 3), 255, np.uint8)
    labels = [f"real frame {frame}", "render (gain fitted)", "real, middle x2", "render, middle x2", "turn 60 deg", "grazing 8 deg",
              "unfilmed side", "low ring 10 deg"]
    for k, t in enumerate(labels):
        cv2.putText(head, t, (k * W + 6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)
    sheet = np.concatenate([head] + rows, 0)
    out = Path(out) if out else rec_dir / "sheet.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def _detail_box(render: np.ndarray, cell: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """The middle of the device in a held-out render (composited over 50 % grey): half its bounding
    box's width, at the cell's aspect ratio."""
    dev = (np.abs(render.astype(np.int16) - 128) > 6).any(-1)
    ys, xs = np.nonzero(dev)
    if len(xs) < 500:
        return None
    x0, x1, y0, y1 = np.percentile(xs, 2), np.percentile(xs, 98), np.percentile(ys, 2), np.percentile(ys, 98)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = max(40.0, (x1 - x0) / 2)
    h = w * cell[1] / cell[0]
    H, W = render.shape[:2]
    xa, xb = int(max(0, cx - w / 2)), int(min(W, cx + w / 2))
    ya, yb = int(max(0, cy - h / 2)), int(min(H, cy + h / 2))
    return (ya, yb, xa, xb) if xb - xa > 20 and yb - ya > 20 else None
