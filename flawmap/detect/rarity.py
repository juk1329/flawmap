"""Self-referential rarity of a face (DINOv2 patch features): how unlike the rest of its own face each
small patch looks.

A clean face is one or two materials repeated over its whole area; a scratch, a stain, a logo or a
printed line is rare there. Every patch of the shading-flattened surface colour (``export.bundle.
albedo_images``) gets a DINOv2 feature (ViT-S/14, public weights ``facebook/dinov2-small``, Apache-2.0,
read from the local Hugging Face cache only), and its rarity is one minus the mean cosine similarity
of its ``k`` most similar patches elsewhere on the same face (patches next to it left out, since they
share its context). Scores are robust z-scores within the face.

No reference images and no training: the face is its own reference, like the multi-view detector.
It is an auxiliary signal (plan E-5, 2026-11-17): recorded with every candidate and design element,
not used to decide."""
from __future__ import annotations

import numpy as np

MODEL_ID = "facebook/dinov2-small"
PATCH = 14
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_MODEL: dict = {}


def available() -> bool:
    """True when torch, transformers and the cached weights are all here (nothing is downloaded)."""
    try:
        import torch  # noqa: F401
        from huggingface_hub import try_to_load_from_cache
        from transformers import AutoModel  # noqa: F401
    except Exception:
        return False
    try:
        return isinstance(try_to_load_from_cache(MODEL_ID, "config.json"), str)
    except Exception:
        return False


def _model(device: str):
    if device not in _MODEL:
        from transformers import AutoModel

        _MODEL[device] = AutoModel.from_pretrained(MODEL_ID, local_files_only=True).to(device).eval()
    return _MODEL[device]


def patch_features(img: np.ndarray, tile_patches: int = 37, device: str | None = None) -> np.ndarray:
    """(rows, cols, C) L2-normalised DINOv2 patch features of an RGB uint8 image, one per 14 x 14 px
    patch (the image is cut to whole patches), computed tile by tile."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = _model(device)
    H, W = img.shape[0] // PATCH, img.shape[1] // PATCH
    x = (img[: H * PATCH, : W * PATCH].astype(np.float32) / 255.0 - _MEAN) / _STD
    out = None
    t = tile_patches
    with torch.no_grad():
        for r0 in range(0, H, t):
            for c0 in range(0, W, t):
                r1, c1 = min(H, r0 + t), min(W, c0 + t)
                tile = torch.from_numpy(np.ascontiguousarray(x[r0 * PATCH: r1 * PATCH, c0 * PATCH: c1 * PATCH].transpose(2, 0, 1)))[None].to(device)
                h = m(pixel_values=tile).last_hidden_state[0, 1:]           # drop the class token
                h = torch.nn.functional.normalize(h.float(), dim=1).reshape(r1 - r0, c1 - c0, -1).cpu().numpy()
                if out is None:
                    out = np.zeros((H, W, h.shape[2]), np.float32)
                out[r0:r1, c0:c1] = h
    return out


def rarity_map(feat: np.ndarray, valid: np.ndarray, k: int = 8, exclude: int = 2, chunk: int = 2048,
               device: str | None = None) -> np.ndarray:
    """Robust z-score of rarity per patch (NaN where not ``valid``): 1 - mean of the ``k`` largest cosine
    similarities to valid patches farther than ``exclude`` patches away (Chebyshev distance)."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rr, cc = np.nonzero(valid)
    out = np.full(valid.shape, np.nan, np.float32)
    if len(rr) <= k + (2 * exclude + 1) ** 2:
        return out
    F = torch.from_numpy(feat[rr, cc]).to(device)
    R = torch.from_numpy(rr).to(device)
    C = torch.from_numpy(cc).to(device)
    score = torch.empty(len(rr), device=device)
    for a in range(0, len(rr), chunk):
        b = min(len(rr), a + chunk)
        S = F[a:b] @ F.T
        near = ((R[a:b, None] - R[None]).abs() <= exclude) & ((C[a:b, None] - C[None]).abs() <= exclude)
        S[near] = -2.0
        score[a:b] = 1.0 - S.topk(k, dim=1).values.mean(1)
    s = score.cpu().numpy()
    med = float(np.median(s))
    mad = 1.4826 * float(np.median(np.abs(s - med))) + 1e-6
    out[rr, cc] = (s - med) / mad
    return out


def face_rarity(img: np.ndarray, valid: np.ndarray, res_mm: float, patch_mm: float = 1.4, min_valid: float = 0.9,
                **kw) -> np.ndarray:
    """Rarity z-score per texel of a chart image (``res_mm`` per texel; NaN outside the face). The
    chart is resampled so that one patch covers ``patch_mm`` of surface (none at 0.1 mm texels)."""
    import cv2

    s = res_mm * PATCH / patch_mm
    work = img if abs(s - 1.0) < 1e-3 else cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    vw = valid.astype(np.float32) if abs(s - 1.0) < 1e-3 else cv2.resize(valid.astype(np.float32), (work.shape[1], work.shape[0]),
                                                                          interpolation=cv2.INTER_AREA)
    feat = patch_features(work)
    H, W = feat.shape[:2]
    share = vw[: H * PATCH, : W * PATCH].reshape(H, PATCH, W, PATCH).mean((1, 3))
    z = rarity_map(feat, share >= min_valid, **kw)
    # back to texels: each texel takes its patch's value
    rows = np.minimum((np.arange(img.shape[0]) * s / PATCH).astype(int), H - 1)
    cols = np.minimum((np.arange(img.shape[1]) * s / PATCH).astype(int), W - 1)
    full = z[rows[:, None], cols[None, :]]
    covered = ((np.arange(img.shape[0]) * s) < H * PATCH)[:, None] & ((np.arange(img.shape[1]) * s) < W * PATCH)[None, :]
    full[~(valid & covered)] = np.nan
    return full


def region_rarity(rar_img: np.ndarray, geom, texels: np.ndarray) -> float | None:
    """Highest rarity z-score over a region's texels (None when none of them has a score)."""
    r, c = np.divmod(geom.flat_index[texels], geom.shape[1])
    v = rar_img[r, c]
    v = v[np.isfinite(v)]
    return round(float(v.max()), 2) if len(v) else None
