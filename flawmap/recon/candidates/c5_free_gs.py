"""C5, Gaussian splatting with geometry learned freely from the capture (the user's experiment), and
the usual fixes for what breaks when the model is turned.

Kerbl et al. 2023 (3D Gaussian splatting), the gsplat library (Ye et al. 2024, Apache-2.0) for the
rasteriser and densification, Yu et al. 2024 (Mip-Splatting: the antialiased mode), Kheradmand et al.
2024 (3DGS as MCMC: a capped, relocated set), Turkulainen et al. (DN-Splatter: depth priors from a
phone's depth sensor), Huang et al. 2024 (2DGS: flat surfels, depth distortion, normal consistency).

Variants, one trainer:
- C5a (vanilla): the whole frame (device and room), no mask: what a splatting app does.
- C5b (+ masks): only the device. Colour loss inside the plate's silhouette (shrunk 3 px), opacity
  pushed to 1 there and to 0 outside it (grown 6 px), random background colour, initial points inside
  the device box, splats outside the box (+5 mm) pruned at the end.
- C5c (+ depth, antialiasing, SH <= 2): C5b plus the LiDAR depth (relative L1 on opaque device
  pixels) and the antialiased rasteriser (MCMC was tried and dropped, see the class).
- C5d (2DGS): C5b with 2D Gaussians (surfels), normal-consistency and depth-distortion losses.

Training frames: every 2nd frame of the training split at 960 x 720 (the context's cache); cameras
in the device frame, metres. 15k iterations (about 5-10 min on an RTX 3080 Ti). Export: a standard
3DGS PLY (positions, SH, opacity logits, log scales, quaternions) in the device frame, metres.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np

from ...utils.logging import get_logger
from ..context import View
from .base import Candidate

log = get_logger("flawmap.recon")
C0 = 0.28209479177387814          # SH band 0 constant


def _ssim(a, b, win=11, sigma=1.5):
    import torch
    import torch.nn.functional as F

    g = torch.arange(win, device=a.device, dtype=a.dtype) - win // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    w = (g[:, None] * g[None, :])[None, None].expand(a.shape[1], 1, win, win).contiguous()
    mu_a = F.conv2d(a, w, padding=win // 2, groups=a.shape[1])
    mu_b = F.conv2d(b, w, padding=win // 2, groups=a.shape[1])
    s_aa = F.conv2d(a * a, w, padding=win // 2, groups=a.shape[1]) - mu_a ** 2
    s_bb = F.conv2d(b * b, w, padding=win // 2, groups=a.shape[1]) - mu_b ** 2
    s_ab = F.conv2d(a * b, w, padding=win // 2, groups=a.shape[1]) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))


class _FreeGS(Candidate):
    family = "free"
    renderer = "splat"
    research = ["3DGS (Kerbl 2023)", "gsplat (2024, Apache-2.0)"]
    # variant switches
    masked = False
    depth = False
    mcmc = False
    antialias = False
    two_d = False
    sh_max = 3
    steps = 15000
    cap = 400_000

    # ------------------------------------------------------------------ data
    def _device_box(self, grow_mm: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
        s = self.ctx.model.spec
        h = np.array([s.width / 2, s.length / 2, max(abs(self.ctx.model.z_front), abs(self.ctx.model.z_back))]) + grow_mm
        return -h * 1e-3, h * 1e-3

    init_from_plate = False

    def _plate_points(self, spacing_mm: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Points on the fitted plate's surface (all charts) with the inspection texture's colours: an
        initialisation only (the splats stay free to move)."""
        import cv2

        from ...export.bundle import atlas_layout
        from ..render import atlas_sampler

        m = self.ctx.model
        V, F = np.asarray(m.mesh.vertices, float), np.asarray(m.mesh.faces)
        a = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
        n = int(a.sum() / spacing_mm ** 2)
        rng = np.random.default_rng(0)
        fi = rng.choice(len(F), n, p=a / a.sum())
        r1, r2 = rng.random(n), rng.random(n)
        s = np.sqrt(r1)
        b = np.stack([1 - s, s * (1 - r2), s * r2], 1)
        P = (V[F[fi]] * b[..., None]).sum(1)
        tex = cv2.cvtColor(cv2.imread(str(self.ctx.base_run / "bundle" / "texture.jpg")), cv2.COLOR_BGR2RGB)
        ch, uv = m.surface_coords(fi, b[:, 1:])
        R = int(np.ceil(n / 4096))
        pad = R * 4096 - n
        C = atlas_sampler(tex, atlas_layout(m), m)(np.concatenate([ch, np.full(pad, ch[0])]).reshape(R, 4096),
                                                   np.concatenate([uv, np.repeat(uv[:1], pad, 0)]).reshape(R, 4096, 2)).reshape(-1, 3)[:n]
        return (P * 1e-3).astype(np.float32), C.astype(np.float32)

    def _init_points(self, frames: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Back-projected LiDAR points (device frame, metres) with colours from the frames."""
        if self.init_from_plate:
            return self._plate_points()
        ctx = self.ctx
        lo, hi = self._device_box(20.0)
        P, C = [], []
        for i in frames[:: max(1, len(frames) // 40)]:
            d, conf = ctx.depth(i)
            if d is None:
                continue
            h, w = d.shape
            cam = ctx.camera(i)
            s = w / cam.width
            fx, fy, cx, cy = cam.fx * s, cam.fy * s, (cam.cx + 0.5) * s - 0.5, (cam.cy + 0.5) * s - 0.5
            v, u = np.nonzero((conf >= 2) & (d > 50) & (d < 3000))
            z = d[v, u]
            pc = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], 1)
            T = ctx.T_cad_cam[i]
            pw = (pc @ T[:3, :3].T + T[:3, 3]) * 1e-3
            img = ctx.image(i)
            sc = img.shape[1] / w
            col = img[np.clip(((v + 0.5) * sc).astype(int), 0, img.shape[0] - 1), np.clip(((u + 0.5) * sc).astype(int), 0, img.shape[1] - 1)]
            keep = np.linalg.norm(pw, axis=1) < 1.2
            if self.masked:
                keep &= np.all((pw > lo) & (pw < hi), axis=1)
            P.append(pw[keep])
            C.append(col[keep] / 255.0)
        if not P and hasattr(ctx, "sfm_points"):        # no LiDAR (a public video): its SfM points
            pw = ctx.sfm_points() * 1e-3
            keep = np.linalg.norm(pw, axis=1) < 1.2
            if self.masked:
                keep &= np.all((pw > lo) & (pw < hi), axis=1)
            P.append(pw[keep])
            C.append(np.full((int(keep.sum()), 3), 0.5))
        P, C = np.concatenate(P), np.concatenate(C)
        # 2 mm voxels (device) / 5 mm (room)
        vox = 0.002 if self.masked else 0.005
        key = np.floor(P / vox).astype(np.int64)
        _, first = np.unique(key, axis=0, return_index=True)
        P, C = P[first], C[first]
        if self.masked and len(P) < 20000:
            # LiDAR barely resolves a 5 mm slab: add random points in the box
            rng = np.random.default_rng(0)
            n = 60000
            Q = rng.uniform(lo, hi, (n, 3))
            P = np.concatenate([P, Q])
            C = np.concatenate([C, np.full((n, 3), 0.6)])
        return P.astype(np.float32), C.astype(np.float32)

    # ------------------------------------------------------------------ fit
    def fit(self) -> dict:
        import torch
        from gsplat import rasterization, rasterization_2dgs
        from gsplat.strategy import DefaultStrategy, MCMCStrategy

        ctx = self.ctx
        dev = "cuda"
        frames = ctx.train_frames
        t0 = time.time()
        imgs, Ks, vms, masks, depths = [], [], [], [], []
        for i in frames:
            v = ctx.view(i)
            imgs.append(torch.from_numpy(ctx.image(i)))
            Ks.append(torch.from_numpy(v.K.astype(np.float32)))
            T = np.asarray(v.T_cad_cam, float).copy()
            T[:3, 3] *= 1e-3
            vms.append(torch.from_numpy(np.linalg.inv(T).astype(np.float32)))
            if self.masked:
                inner = ctx.silhouette(v, -3.0)
                outer = ~ctx.silhouette(v, 6.0)
                masks.append(torch.from_numpy((inner * 2 + outer * 1).astype(np.uint8)))
            if self.depth:
                d, conf = ctx.depth(i)
                if d is None:
                    d, conf = np.zeros((192, 256), np.float32), np.zeros((192, 256), np.uint8)
                depths.append(torch.from_numpy(np.where(conf >= 2, d * 1e-3, 0).astype(np.float32)))
        imgs = torch.stack(imgs).to(dev)
        Ks, vms = torch.stack(Ks).to(dev), torch.stack(vms).to(dev)
        masks = torch.stack(masks).to(dev) if masks else None
        depths = torch.stack(depths).to(dev) if depths else None
        H, W = imgs.shape[1:3]
        P, C = self._init_points(frames)
        n0 = len(P)
        # scene scale: camera distance to the device
        cams = np.stack([np.linalg.inv(vm.cpu().numpy())[:3, 3] for vm in vms])
        scene_scale = float(np.median(np.linalg.norm(cams, axis=1)))
        pts = torch.from_numpy(P).to(dev)
        # initial scale: mean distance to 3 nearest neighbours
        from scipy.spatial import cKDTree

        dist, _ = cKDTree(P).query(P, k=4)
        s0 = np.log(np.clip(dist[:, 1:].mean(1), 1e-4, 0.02)).astype(np.float32)
        N = len(P)
        sh_n = (self.sh_max + 1) ** 2 - 1
        params = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(pts),
            "scales": torch.nn.Parameter(torch.from_numpy(np.repeat(s0[:, None], 3, 1)).to(dev)),
            "quats": torch.nn.Parameter(torch.nn.functional.normalize(torch.randn(N, 4, device=dev), dim=-1)),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.5, device=dev))),
            "sh0": torch.nn.Parameter(((torch.from_numpy(C).to(dev) - 0.5) / C0)[:, None, :]),
            "shN": torch.nn.Parameter(torch.zeros(N, sh_n, 3, device=dev)),
        })
        lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
        opts = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items()}
        sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / self.steps))
        if self.mcmc:
            strategy = MCMCStrategy(cap_max=self.cap, refine_stop_iter=int(self.steps * 0.8))
        else:
            # the periodic opacity reset (against floaters) is off for the masked variants: their opacity is
            # already pushed to 0 off the device, and with the reset the model collapsed on cc53da7c23 to
            # 255 splats (2026-11-18)
            strategy = DefaultStrategy(refine_stop_iter=int(self.steps * 0.8), reset_every=(self.steps + 1) if self.masked else 3000,
                                       key_for_gradient="gradient_2dgs" if self.two_d else "means2d")
        strategy.check_sanity(params, opts)
        state = strategy.initialize_state(scene_scale=scene_scale) if not self.mcmc else strategy.initialize_state()
        rng = np.random.default_rng(0)
        mode = "antialiased" if self.antialias else "classic"
        losses = []
        for step in range(self.steps):
            k = int(rng.integers(len(frames)))
            gt = imgs[k].float()[None] / 255.0
            sh_deg = min(step // 1000, self.sh_max)
            colors = torch.cat([params["sh0"], params["shN"]], 1)
            bg = torch.rand(1, 3, device=dev) if self.masked else torch.zeros(1, 3, device=dev)
            if self.two_d:
                out = rasterization_2dgs(params["means"], params["quats"], torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
                                         colors, vms[k:k + 1], Ks[k:k + 1], W, H, sh_degree=sh_deg, backgrounds=bg,
                                         render_mode="RGB+ED", distloss=True)
                rgbd, alpha, rn, sn, distort, _, info = out
            else:
                rgbd, alpha, info = rasterization(params["means"], params["quats"], torch.exp(params["scales"]),
                                                  torch.sigmoid(params["opacities"]), colors, vms[k:k + 1], Ks[k:k + 1], W, H,
                                                  sh_degree=sh_deg, backgrounds=bg, render_mode="RGB+ED", rasterize_mode=mode,
                                                  absgrad=False, packed=False)
            rgb, dep = rgbd[..., :3], rgbd[..., 3:4]
            strategy.step_pre_backward(params, opts, state, step, info)
            if masks is not None:
                m = masks[k][None, ..., None]
                inner, outer = (m == 2).float(), (m == 1).float()
                target = gt * inner + bg[:, None, None, :] * outer
                wmask = inner + outer
                l1 = (torch.abs(rgb - target) * wmask).sum() / (wmask.sum() * 3 + 1e-6)
                ss = _ssim((rgb * wmask).permute(0, 3, 1, 2), (target * wmask).permute(0, 3, 1, 2)).mean()
                a = alpha.clamp(1e-5, 1 - 1e-5)
                la = -(inner * torch.log(a)).sum() / (inner.sum() + 1) - (outer * torch.log(1 - a)).sum() / (outer.sum() + 1)
                loss = 0.8 * l1 + 0.2 * (1 - ss) + 0.1 * la
            else:
                l1 = torch.abs(rgb - gt).mean()
                ss = _ssim(rgb.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)).mean()
                loss = 0.8 * l1 + 0.2 * (1 - ss)
            if depths is not None and step > 500:
                # expected depth is undefined where nothing is drawn: only opaque device pixels count
                dl = depths[k][None, None]
                dr = torch.nan_to_num(torch.nn.functional.interpolate(dep.permute(0, 3, 1, 2), size=dl.shape[-2:], mode="area"), 0.0)
                ar = torch.nn.functional.interpolate(alpha.permute(0, 3, 1, 2), size=dl.shape[-2:], mode="area")
                mi = torch.nn.functional.interpolate((masks[k][None, None] == 2).float(), size=dl.shape[-2:], mode="area") > 0.99
                ok = mi & (dl > 0) & (ar > 0.5) & torch.isfinite(dr)
                if ok.any():
                    loss = loss + 0.1 * (torch.abs(dr - dl) / dl)[ok].clamp(max=1.0).mean()
            if self.two_d and step > 3000:
                loss = loss + 0.05 * (1 - (rn * sn).sum(-1)).mean() + 0.01 * distort.mean()
            if self.mcmc:
                loss = loss + 0.01 * torch.sigmoid(params["opacities"]).mean() + 0.01 * torch.exp(params["scales"]).mean()
            loss.backward()
            if self.mcmc:
                strategy.step_post_backward(params, opts, state, step, info, lr=sched.get_last_lr()[0])
            else:
                strategy.step_post_backward(params, opts, state, step, info, packed=False)
            for o in opts.values():
                o.step()
                o.zero_grad(set_to_none=True)
            sched.step()
            if step % 500 == 0:
                losses.append(float(loss.item()))
                if step % 2500 == 0:
                    log.info(f"[{ctx.rec}] {self.id} step {step}: loss {loss.item():.4f}, {len(params['means'])} splats")
        with torch.no_grad():
            keep = torch.ones(len(params["means"]), dtype=torch.bool, device=dev)
            if self.masked:
                lo, hi = (torch.tensor(x, device=dev) for x in self._device_box(5.0))
                keep &= ((params["means"] > lo) & (params["means"] < hi)).all(1)
            keep &= torch.sigmoid(params["opacities"]) > 0.005
            data = {k: v[keep].detach().cpu().numpy() for k, v in params.items()}
        write_ply(self.art / "splat.ply", data, two_d=self.two_d)
        return {"steps": self.steps, "splats": int(keep.sum()), "init_points": n0, "frames": len(frames), "loss": losses[-5:],
                "train_s": round(time.time() - t0, 1)}

    # ------------------------------------------------------------------ render
    def load(self) -> None:
        import torch

        d = read_ply(self.art / "splat.ply")
        dev = "cuda"
        self.g = {k: torch.from_numpy(v).to(dev) for k, v in d.items()}
        self.deg = int(round(math.sqrt(self.g["shN"].shape[1] + 1))) - 1

    def render(self, view: View):
        import torch
        from gsplat import rasterization, rasterization_2dgs

        if len(self.g["means"]) == 0:                       # the fit lost every splat: nothing is drawn
            return np.zeros((view.H, view.W, 3), np.float32), np.zeros((view.H, view.W), np.float32)

        T = np.asarray(view.T_cad_cam, float).copy()
        T[:3, 3] *= 1e-3
        vm = torch.from_numpy(np.linalg.inv(T).astype(np.float32)).cuda()[None]
        K = torch.from_numpy(np.asarray(view.K, np.float32)).cuda()[None]
        g = self.g
        colors = torch.cat([g["sh0"], g["shN"]], 1)
        with torch.no_grad():
            if self.two_d:
                out = rasterization_2dgs(g["means"], g["quats"], torch.exp(g["scales"]), torch.sigmoid(g["opacities"]), colors, vm, K,
                                         view.W, view.H, sh_degree=self.deg)
                rgb, alpha = out[0], out[1]
            else:
                rgb, alpha, _ = rasterization(g["means"], g["quats"], torch.exp(g["scales"]), torch.sigmoid(g["opacities"]), colors, vm, K,
                                              view.W, view.H, sh_degree=self.deg,
                                              rasterize_mode="antialiased" if self.antialias else "classic")
        a = alpha[0, ..., 0].clamp(0, 1).cpu().numpy()
        c = rgb[0, ..., :3].clamp(0, 1).cpu().numpy()
        # rendered colour is premultiplied over black: un-premultiply where opaque enough
        c = np.where(a[..., None] > 1e-3, c / np.maximum(a[..., None], 1e-3), 0).clip(0, 1)
        return c.astype(np.float32), a.astype(np.float32)

    # ------------------------------------------------------------------ web
    def export_web(self, web: Path) -> dict:
        import shutil

        web.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.art / "splat.ply", web / "splat.ply")
        return self.write_manifest(web, {"files": {"splat": "splat.ply"}, "frame": "device, metres",
                                         "splats": self.fit_stats().get("splats")})


def write_ply(path: Path, d: dict, two_d: bool = False) -> None:
    """Standard 3DGS PLY (binary little endian): x y z nx ny nz f_dc_0-2 f_rest_* opacity scale_0-2 rot_0-3.
    2D splats get a tiny third scale (log 1e-7 m) so viewers draw them as flat discs."""
    n = len(d["means"])
    # channel-major, like the reference implementation (explicit width: a fit that lost every splat has n = 0)
    rest = d["shN"].transpose(0, 2, 1).reshape(n, 3 * d["shN"].shape[1])
    scales = d["scales"].copy()
    if two_d and scales.shape[1] == 2:
        scales = np.concatenate([scales, np.full((n, 1), np.log(1e-7), np.float32)], 1)
    elif two_d:
        scales[:, 2] = np.log(1e-7)
    cols = [d["means"], np.zeros((n, 3), np.float32), d["sh0"].reshape(n, 3), rest, d["opacities"][:, None], scales,
            d["quats"]]
    arr = np.concatenate([c.astype(np.float32) for c in cols], 1)
    names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(rest.shape[1])]
             + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    head = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" + "".join(f"property float {k}\n" for k in names) + "end_header\n"
    with open(path, "wb") as f:
        f.write(head.encode())
        f.write(arr.astype("<f4").tobytes())


def read_ply(path: Path) -> dict:
    with open(path, "rb") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            head += f.readline()
        names = [ln.split()[-1] for ln in head.decode().splitlines() if ln.startswith("property")]
        n = int([ln for ln in head.decode().splitlines() if ln.startswith("element vertex")][0].split()[-1])
        arr = np.frombuffer(f.read(), dtype="<f4").reshape(n, len(names))
    col = {k: i for i, k in enumerate(names)}
    rest = [k for k in names if k.startswith("f_rest_")]
    nrest = len(rest) // 3
    shN = arr[:, [col[k] for k in rest]].reshape(n, 3, nrest).transpose(0, 2, 1) if nrest else np.zeros((n, 0, 3), np.float32)
    return {"means": arr[:, [col["x"], col["y"], col["z"]]].copy(), "sh0": arr[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]][:, None, :].copy(),
            "shN": np.ascontiguousarray(shN), "opacities": arr[:, col["opacity"]].copy(),
            "scales": arr[:, [col["scale_0"], col["scale_1"], col["scale_2"]]].copy(),
            "quats": arr[:, [col["rot_0"], col["rot_1"], col["rot_2"], col["rot_3"]]].copy()}


class C5aVanilla(_FreeGS):
    id = "c5a"
    label_ko = "자유 가우시안: 기본 (방 전체, 사용자 실험 재현)"


class C5bMasked(_FreeGS):
    id = "c5b"
    label_ko = "자유 가우시안: 기기 마스크 + 배경 투명"
    masked = True
    research = _FreeGS.research + ["물체 마스크·알파 손실 (객체 중심 GS 관행)"]


class C5cDepth(_FreeGS):
    id = "c5c"
    label_ko = "자유 가우시안: 마스크 + LiDAR 깊이 + 안티앨리어싱 + SH 2차"
    # MCMC (Kheradmand et al. 2024) was tried and dropped: with masks, depth and antialiasing together its
    # relocation hit a CUDA assert after many splats went transparent (gsplat 1.5.3, 2026-11-18)
    masked = depth = antialias = True
    sh_max = 2
    research = _FreeGS.research + ["DN-Splatter (휴대폰 깊이 사전)", "Mip-Splatting (CVPR 2024)"]


class C5d2DGS(_FreeGS):
    id = "c5d"
    label_ko = "자유 가우시안: 2D 서펠 (2DGS) + 마스크"
    masked = two_d = True
    research = _FreeGS.research + ["2DGS (SIGGRAPH 2024)"]


class C5ePlateInit(_FreeGS):
    id = "c5e"
    label_ko = "자유 가우시안: 맞춘 판에서 시작 + 마스크 (판에 묶지는 않음)"
    masked = True
    init_from_plate = True
    steps = 20000
    research = _FreeGS.research + ["초기값만 판 표면 (SuGaR류 표면 정렬의 가벼운 판)"]
