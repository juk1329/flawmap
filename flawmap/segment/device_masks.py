"""Where the device is in each keyframe, independently of the registration (SAM 2 video predictor).

Until now every stage judged itself only by the outline residual, and on a wood floor a board seam
can fit an outline as well as the device edge does: a candidate turned 90 degrees won by 0.01 px,
a COLMAP re-registration ended 1 cm off, and the texture took in floor (iPad, 2026-11-17). A mask
of the device in each frame is an independent check:

* ``device_masks``: SAM 2 tracks the device through the keyframes. It is prompted automatically with
  boxes from a pose that does not come from the registration being checked (the floor-mosaic plate,
  or any pose the caller trusts), on several keyframes, and propagated forwards and backwards.
* ``silhouette`` / ``mask_iou``: the registered model projected into a frame and compared with the
  mask; ``iou_report`` summarises a pass (median, worst frames).
* ``inside``: whether projected texels fall inside the (eroded) mask, so sampling never takes floor
  pixels even when a frame is slightly misregistered.

SAM 2 predicts masks at 256 x 256 internally (about 1 mm per mask pixel for a tablet at 40 cm), so
the masks judge and gate; sub-millimetre edges still come from image edges (``anchor.register``).

Needs ``torch`` (2.7.0+cu128) and Meta's ``sam2`` (github.com/facebookresearch/sam2, Apache-2.0) with
the SAM 2.1 hiera-small checkpoint in ``data/models/sam2.1_hiera_small.pt`` (``docs/project_guide.md``
6.13 says where to get both).
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from ..geometry.camera import Camera
from ..geometry.transforms import inv_T
from ..surface.raycast import RaycastScene
from ..utils.io import data_root

CHECKPOINT = "models/sam2.1_hiera_small.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"


def available() -> bool:
    try:
        import sam2  # noqa: F401
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available() and (data_root() / CHECKPOINT).exists()


def silhouette(scene: RaycastScene, cam: Camera, T_cad_cam: np.ndarray, scale: float = 0.25) -> np.ndarray:
    """Boolean mask of the model seen from a camera (``T_cad_cam``: camera pose in the model frame),
    at ``scale`` of the frame size."""
    return scene.cast(cam.scaled(scale), T_cad_cam, step=1).hit


def box_of(mask: np.ndarray, pad: float = 0.04) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    w, h = xs.max() - xs.min(), ys.max() - ys.min()
    return np.array([xs.min() - pad * w, ys.min() - pad * h, xs.max() + pad * w, ys.max() + pad * h], np.float32)


def device_masks(frames: dict[int, np.ndarray], prompt_boxes: dict[int, np.ndarray], max_side: int = 1024,
                 out_scale: float = 0.25, log=print) -> dict[int, np.ndarray]:
    """SAM 2 masks (bool, ``out_scale`` of the frame size: SAM 2 predicts at 256 x 256, and full-size
    masks of two 100-frame passes ran a 32 GB PC into the memory guard) for every frame in
    ``frames`` (time order = key order), prompted with boxes (x0, y0, x1, y1 in full-frame pixels)
    on the frames in ``prompt_boxes``."""
    prompts = {i: {"box": b} for i, b in prompt_boxes.items() if b is not None}
    return track_device(frames, prompts, max_side=max_side, out_scale=out_scale, log=log)


def track_device(frames: dict[int, np.ndarray], prompts: dict[int, dict], max_side: int = 1024, out_scale: float = 0.25,
                 log=print) -> dict[int, np.ndarray]:
    """SAM 2 video masks of one object (the device) for every frame in ``frames`` (time order = key
    order). ``prompts[frame]`` holds any of ``box`` (x0, y0, x1, y1), ``points`` (n, 2) and ``labels``
    (n,: 1 inside the device, 0 outside), all in full-frame pixels, or a ``mask`` (bool, any size).
    Propagated forwards and backwards from the prompted frames; a frame's mask is the union of the
    two directions."""
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    ids = sorted(frames)
    h, w = frames[ids[0]].shape[:2]
    s = min(1.0, max_side / max(h, w))
    work = Path(tempfile.mkdtemp(prefix="flawmap_sam2_"))
    try:
        for j, i in enumerate(ids):
            img = frames[i] if s == 1.0 else cv2.resize(frames[i], (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(work / f"{j:05d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        predictor = build_sam2_video_predictor(CONFIG, str(data_root() / CHECKPOINT), device="cuda")
        out: dict[int, np.ndarray] = {}
        n_prompted = 0
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=str(work), offload_video_to_cpu=True, offload_state_to_cpu=True)
            pos = {i: j for j, i in enumerate(ids)}
            for i, pr in prompts.items():
                if i not in pos or not pr:
                    continue
                if pr.get("mask") is not None:
                    predictor.add_new_mask(state, frame_idx=pos[i], obj_id=1, mask=np.asarray(pr["mask"], bool))
                    n_prompted += 1
                    continue
                kw = {}
                if pr.get("box") is not None:
                    kw["box"] = np.asarray(pr["box"], np.float32) * s
                if pr.get("points") is not None and len(pr["points"]):
                    kw["points"] = np.asarray(pr["points"], np.float32) * s
                    kw["labels"] = np.asarray(pr["labels"], np.int32)
                if not kw:
                    continue
                predictor.add_new_points_or_box(state, frame_idx=pos[i], obj_id=1, **kw)
                n_prompted += 1
            for reverse in (False, True):
                for j, _, logits in predictor.propagate_in_video(state, reverse=reverse):
                    m = (logits[0, 0] > 0.0).float().cpu().numpy()
                    m = cv2.resize(m, (int(round(w * out_scale)), int(round(h * out_scale))), interpolation=cv2.INTER_LINEAR)
                    i = ids[j]
                    m = m > 0.5
                    out[i] = m if i not in out else (out[i] | m) if reverse else m
            predictor.reset_state(state)
        del predictor
        torch.cuda.empty_cache()
        log(f"  sam2: masks for {len(out)}/{len(ids)} frames, prompted on {n_prompted}")
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def object_hypotheses(frames: dict[int, np.ndarray], points: dict[int, np.ndarray], max_side: int = 1024) -> dict[int, list]:
    """SAM 2's nested answers (part, larger part, whole object) to one positive point per frame:
    ``{frame: [(mask bool at the frame's size, predicted IoU), ...]}``. A single point is ambiguous
    (a point on the iPad's logo gave the logo, 32 x 31 mm, 2026-11-17); the caller picks the
    hypothesis whose size makes sense."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(CONFIG, str(data_root() / CHECKPOINT), device="cuda")
    pred = SAM2ImagePredictor(model)
    out = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, pt in points.items():
            img = frames[i]
            h, w = img.shape[:2]
            s = min(1.0, max_side / max(h, w))
            small = img if s == 1.0 else cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
            pred.set_image(small)
            masks, scores, _ = pred.predict(point_coords=np.asarray(pt, np.float32).reshape(1, 2) * s,
                                            point_labels=np.ones(1, np.int32), multimask_output=True)
            out[i] = [(cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0, float(sc))
                      for m, sc in zip(masks, scores)]
    del pred, model
    torch.cuda.empty_cache()
    return out


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def iou_report(model, masks: dict[int, np.ndarray], cams: dict[int, Camera], T_cad_cam: dict[int, np.ndarray],
               scale: float = 0.25) -> dict:
    """IoU of the projected model and the mask per frame, and a summary."""
    scene = RaycastScene(model.mesh, None)
    per = {}
    for i, m in masks.items():
        if i not in T_cad_cam:
            continue
        sil = silhouette(scene, cams[i], T_cad_cam[i], scale)
        per[i] = mask_iou(sil, m)
    v = np.array(list(per.values())) if per else np.zeros(0)
    worst = sorted(per, key=per.get)[:5]
    return {"frames": len(per), "iou_p50": round(float(np.median(v)), 4) if len(v) else None,
            "iou_p10": round(float(np.percentile(v, 10)), 4) if len(v) else None,
            "worst": {int(i): round(per[i], 3) for i in worst}, "per_frame": {int(i): round(x, 4) for i, x in per.items()}}


def inside(mask: np.ndarray, u: np.ndarray, v: np.ndarray, erode_px: int = 0) -> np.ndarray:
    """Whether pixel positions (u, v) fall inside ``mask`` (optionally eroded by ``erode_px``)."""
    m = mask
    if erode_px > 0:
        m = cv2.erode(mask.astype(np.uint8), np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)) > 0
    h, w = m.shape
    ui = np.clip(np.round(u).astype(int), 0, w - 1)
    vi = np.clip(np.round(v).astype(int), 0, h - 1)
    return m[vi, ui]


def mask_lookup(mask: np.ndarray, frame_w: int, frame_h: int, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Whether full-frame pixel positions (u, v) fall inside a mask of any (smaller) size."""
    h, w = mask.shape
    ui = np.clip(np.floor((u + 0.5) * w / frame_w).astype(int), 0, w - 1)
    vi = np.clip(np.floor((v + 0.5) * h / frame_h).astype(int), 0, h - 1)
    return mask[vi, ui]
