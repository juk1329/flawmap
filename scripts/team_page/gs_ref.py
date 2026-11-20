"""Reference renders for checking the page's splat renderer (gs_lite.js) with gsplat, the renderer the
splats were trained with: the page's own .splat file (cut, base colour only) and the full model
(every splat, view-dependent colour), through one test camera of the recording.

    python scripts/team_page/gs_ref.py <splat file> <full ply> <compare index.json> <test view no> <out_dir> [width height]

With a size, the camera is scaled to it (the page's canvas in device pixels), so the two renders can be
compared pixel for pixel.
"""
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization

from flawmap.recon.candidates.c5_free_gs import read_ply


def read_bytes(p: Path) -> bytes:
    """A page file, stored as base64 text beside its name (build.py) or as it is."""
    b64 = Path(str(p) + ".b64.txt")
    return base64.b64decode(b64.read_text()) if b64.exists() else p.read_bytes()


splat, ply, index, k, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
out.mkdir(parents=True, exist_ok=True)
tv = json.loads(index.read_text())["test_views"][k]
K = np.asarray(tv["K"], np.float32)
T = np.asarray(tv["T_cad_cam"], np.float64)
T[:3, 3] *= 1e-3
view = np.linalg.inv(T).astype(np.float32)
W, H = int(round(2 * K[0, 2])), int(round(2 * K[1, 2]))
H = int(round(W * 3 / 4))
if len(sys.argv) > 7:
    W2, H2 = int(sys.argv[6]), int(sys.argv[7])
    s = H2 / (2 * K[1, 2])                      # the page's camera: this focal, the centre as principal point
    K = np.array([[K[0, 0] * s, 0, W2 / 2], [0, K[1, 1] * s, H2 / 2], [0, 0, 1]], np.float32)
    W, H = W2, H2
dev = torch.device("cuda")
vm, Ks = torch.tensor(view, device=dev)[None], torch.tensor(K, device=dev)[None]

rec = np.frombuffer(read_bytes(splat), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
q = (rec["r"].astype(np.float32) - 128) / 128
q /= np.linalg.norm(q, axis=1, keepdims=True)
t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
img, alpha, _ = rasterization(t(rec["p"]), t(q), t(rec["s"]), t(rec["c"][:, 3] / 255.0), t(rec["c"][:, :3] / 255.0), vm, Ks, W, H)
cv2.imwrite(str(out / f"ref_cut_dc_{k}.png"), cv2.cvtColor((img[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))

d = read_ply(ply)
deg = int(round(np.sqrt(d["shN"].shape[1] + 1))) - 1
colors = t(np.concatenate([d["sh0"], d["shN"]], 1))
img, _, _ = rasterization(t(d["means"]), t(d["quats"] / np.linalg.norm(d["quats"], axis=1, keepdims=True)), t(np.exp(d["scales"])),
                          t(1 / (1 + np.exp(-d["opacities"]))), colors, vm, Ks, W, H, sh_degree=deg)
cv2.imwrite(str(out / f"ref_full_sh{deg}_{k}.png"), cv2.cvtColor((img[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
img, _, _ = rasterization(t(d["means"]), t(d["quats"] / np.linalg.norm(d["quats"], axis=1, keepdims=True)), t(np.exp(d["scales"])),
                          t(1 / (1 + np.exp(-d["opacities"]))), colors[:, :1], vm, Ks, W, H, sh_degree=0)
cv2.imwrite(str(out / f"ref_full_dc_{k}.png"), cv2.cvtColor((img[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
print(W, H, "deg", deg, "splats", len(rec), len(d["means"]))

# the page's file with its view-dependent colour, as the page decodes it
ex = json.loads((splat.parent.parent / "examples.json").read_text())
meta = next(f["gs"] for e in ex for f in e["faces"] if f.get("gs") and f["gs"]["file"] == splat.name and splat.parent.name == e["id"])
shq = np.frombuffer(read_bytes(splat.parent / meta["sh"]), np.uint8).reshape(len(rec), meta["sh_rest"], 3)
sh = (shq.astype(np.float32) - 128) / 127 * meta["sh_scale"]
dc = (rec["c"][:, :3].astype(np.float32) / 255 - 0.5) / 0.28209479177387814
cols = t(np.concatenate([dc[:, None, :], sh], 1))
deg = int(round(np.sqrt(meta["sh_rest"] + 1))) - 1
img, _, _ = rasterization(t(rec["p"]), t(q), t(rec["s"]), t(rec["c"][:, 3] / 255.0), cols, vm, Ks, W, H, sh_degree=deg)
cv2.imwrite(str(out / f"ref_cut_sh_{k}.png"), cv2.cvtColor((img[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
