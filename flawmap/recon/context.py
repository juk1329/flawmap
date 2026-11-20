"""One recording prepared once for every reconstruction candidate.

Inputs come from a finished inspection run made with ``flawmap run ... --holdout blocks``:
the keyframe poses and the device registration of its pass (``passes/<pass>/``), the fitted device
model (``run.json``), the defects (``bundle/defects.json``) and the held-out split (``holdout.json``).
The ARKit pose of every frame is corrected like the keyframes (``texture.allframes.all_frame_poses``),
so every frame has a camera in the device frame: mm, OpenCV axes (x right, y down, z forward).

Frames are kept at the working scale (0.5: 960 x 720): training frames as JPEG (q 95), test frames
as PNG, decoded once into ``runs/compare/<recording>/ctx/``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..eval.heldout import Split
from ..geometry.camera import Camera
from ..geometry.transforms import inv_T
from ..utils.logging import get_logger

log = get_logger("flawmap.recon")

SCALE = 0.5
SATURATED = 248          # any channel at or above this (8-bit, half resolution) counts as clipped
OCC_MARGIN_MM = 12.0     # LiDAR this much nearer than the plate along the view axis: something is in front of it


@dataclass
class View:
    """A pinhole view of the device frame: ``K`` for an image of ``W`` x ``H``; ``T_cad_cam`` maps
    camera to device coordinates (mm, OpenCV camera axes)."""
    K: np.ndarray
    T_cad_cam: np.ndarray
    W: int
    H: int
    frame: int | None = None
    name: str = ""

    @property
    def cam(self) -> Camera:
        return Camera(self.K, self.W, self.H)

    @property
    def center(self) -> np.ndarray:
        return np.asarray(self.T_cad_cam)[:3, 3]

    def to_json(self) -> dict:
        return {"K": np.asarray(self.K).tolist(), "T_cad_cam": np.asarray(self.T_cad_cam).tolist(), "W": self.W, "H": self.H,
                "frame": self.frame, "name": self.name}


@dataclass
class CaptureContext:
    rec: str                       # recording id (Stray folder name)
    root: Path                     # Stray folder
    base_run: Path                 # the inspection run the cameras and defects come from
    out: Path                      # runs/compare/<rec>
    face: str                      # "back" | "screen": the face that lay up
    cap: object                    # io.stray capture
    split: Split
    T_cad_cam: dict[int, np.ndarray]
    model: object                  # anchor.parametric.DeviceModel (the fitted plate)
    run_report: dict
    defects: dict                  # bundle/defects.json
    scale: float = SCALE
    train_step: int = 2            # every 2nd training frame is cached (30 fps)
    test_step: int = 3             # every 3rd test frame is scored
    pass_index: int = 0            # which pass of the run this recording is
    _sil_pts: np.ndarray | None = field(default=None, repr=False)
    _raster: object = field(default=None, repr=False)

    # ------------------------------------------------------------------ building
    @classmethod
    def from_run(cls, base_run: str | Path, out_root: str | Path = "runs/compare", scale: float = SCALE, pass_index: int = 0,
                 split: Split | None = None) -> "CaptureContext":
        """``pass_index``: which pass of the run (one recording each); ``split``: the frames' roles
        when the run was not made with ``--holdout blocks`` (the service's realistic view,
        ``recon.service``), else read from ``holdout.json``."""
        from ..anchor.template import model_of_run
        from ..io.stray import read_stray
        from ..texture.allframes import all_frame_poses

        base_run = Path(base_run)
        rep = json.loads((base_run / "run.json").read_text())
        p = rep["passes"][pass_index]
        root = Path(p["root"])
        cap = read_stray(root)
        pdir = base_run / "passes" / p["name"]
        poses = json.loads((pdir / "poses.json").read_text())
        key_T = {int(k): np.asarray(v["T_world_cam"] if isinstance(v, dict) else v, float) for k, v in poses["poses"].items()}
        T_world_cad = np.asarray(json.loads((pdir / "registration.json").read_text())["T_world_cad"], float)
        allT = all_frame_poses(np.asarray(cap.T_ark_cam), key_T, len(cap))
        Tinv = inv_T(T_world_cad)
        if split is None:
            split = Split.from_json(json.loads((base_run / "holdout.json").read_text())[p["name"]], len(cap))
        ctx = cls(rec=root.name, root=root, base_run=base_run, out=Path(out_root) / root.name, face=p["face_up"], cap=cap,
                  split=split, T_cad_cam={i: Tinv @ T for i, T in allT.items()},
                  model=model_of_run(rep), run_report=rep, defects=json.loads((base_run / "bundle" / "defects.json").read_text()),
                  scale=scale, pass_index=pass_index)
        ctx.out.mkdir(parents=True, exist_ok=True)
        return ctx

    # ------------------------------------------------------------------ frames
    @property
    def n_video(self) -> int:
        """Frames the video really has (the odometry can list one more than the decoder returns)."""
        if not hasattr(self, "_n_video"):
            import cv2 as _cv

            n = int(_cv.VideoCapture(str(self.root / "rgb.mp4")).get(_cv.CAP_PROP_FRAME_COUNT))
            self._n_video = min(n, len(self.cap)) - 1          # the last one is unreliable: leave it out
        return self._n_video

    @property
    def train_frames(self) -> list[int]:
        return [i for i in self.split.train[:: self.train_step] if i < self.n_video]

    @property
    def test_frames(self) -> list[int]:
        return [i for i in self.split.test[:: self.test_step] if i < self.n_video]

    @property
    def val_frames(self) -> list[int]:
        """Validation frames: for choosing a candidate's settings (never the test frames)."""
        return [i for i in self.split.val[:: self.test_step] if i < self.n_video]

    def _path(self, i: int) -> Path:
        held = i in self._held
        return self.out / "ctx" / ("test" if held else "frames") / f"{i:05d}.{'png' if held else 'jpg'}"

    @property
    def _held(self) -> set:
        if not hasattr(self, "_held_set"):
            self._held_set = set(self.split.test) | set(self.split.val)
        return self._held_set

    def ensure_frames(self) -> None:
        """Decode the training, validation and test frames once at the working scale."""
        want = sorted(set(self.train_frames) | set(self.test_frames) | set(self.val_frames))
        missing = [i for i in want if not self._path(i).exists()]
        if not missing:
            return
        (self.out / "ctx" / "frames").mkdir(parents=True, exist_ok=True)
        (self.out / "ctx" / "test").mkdir(parents=True, exist_ok=True)
        for i, img in self.full_frames(missing, scale=self.scale):
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if i in self._held:
                cv2.imwrite(str(self._path(i)), bgr)
            else:
                cv2.imwrite(str(self._path(i)), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        log.info(f"[{self.rec}] cached {len(missing)} frames at scale {self.scale}")

    def full_frames(self, frames: list[int], scale: float = 1.0):
        """Yield ``(i, RGB uint8)`` for the frames ``i`` (sorted), at ``scale`` of the full resolution."""
        from ..io.stray import iter_frames

        yield from iter_frames(self.root / "rgb.mp4", frames, scale=scale)

    def camera(self, i: int) -> Camera:
        """The full-resolution camera of frame ``i``."""
        return self.cap.camera(i)

    def image(self, i: int) -> np.ndarray:
        """RGB uint8 frame at the working scale."""
        p = self._path(i)
        if not p.exists():
            self.ensure_frames()
        return cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    def view(self, i: int) -> View:
        cam = self.camera(i).scaled(self.scale)
        return View(cam.K, self.T_cad_cam[i], cam.width, cam.height, frame=i, name=f"frame {i}")

    # ------------------------------------------------------------------ masks
    def silhouette(self, view: View, grow_px: float = 0.0, visible: bool = True) -> np.ndarray:
        """Where the fitted plate covers the image: the convex hull of its projected vertices (the
        plate is convex), optionally grown (> 0) or shrunk (< 0) by ``grow_px``. For a recorded frame
        and ``grow_px`` <= 0 (the device's own pixels: what candidates learn from and are scored on),
        what stands in front of the plate is taken out (``occluded``). A grown silhouette (the zone
        outside of which nothing may be drawn) keeps it."""
        if self._sil_pts is None:
            v = np.asarray(self.model.mesh.vertices, float)
            self._sil_pts = v[np.random.default_rng(0).choice(len(v), min(len(v), 6000), replace=False)]
        m = plate_silhouette(self._sil_pts, view, grow_px)
        if visible and grow_px <= 0 and view.frame is not None:
            m &= ~self.occluded(view.frame)
        return m

    def occluded(self, i: int) -> np.ndarray:
        """Working-scale mask of frame ``i`` where something is between the camera and the fitted plate:
        the LiDAR depth is more than OCC_MARGIN_MM nearer than the plate along the view axis (a stand's
        lip, a hand), opened against single noisy depth pixels and grown by 4 px. Nobody marks it (the
        user's premise: stands are handled automatically). All False without depth. Cached in ctx/occ/."""
        p = self.out / "ctx" / "occ" / f"{i:05d}.png"
        if p.exists():
            return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0
        v = self.view(i)
        occ = np.zeros((v.H, v.W), bool)
        d, c = self.depth(i)
        if d is not None:
            if self._raster is None:
                from .render import PlateRaster

                self._raster = PlateRaster(self.model)
            h = self._raster.hits(v)
            T = np.asarray(v.T_cad_cam, float)
            zp = np.where(h.hit, np.einsum("hwk,k->hw", np.nan_to_num(h.point) - T[:3, 3], T[:3, 2]), np.inf)
            dd = cv2.resize(d, (v.W, v.H), interpolation=cv2.INTER_NEAREST)
            cc = cv2.resize(c, (v.W, v.H), interpolation=cv2.INTER_NEAREST)
            # high confidence only, not at the depth map's border, and a margin that grows with distance:
            # a close-up of the flat iPad (cc53da7c23 frame 869, plate about 25 cm away) flagged 3.4 % of the
            # device at the image border and on the camera bump with confidence >= 1 and a fixed 12 mm
            bm = int(round(0.06 * min(v.W, v.H)))
            inner = np.zeros_like(h.hit)
            inner[bm:-bm, bm:-bm] = True
            zh = np.where(h.hit, zp, 0.0)
            o = h.hit & inner & (cc >= 2) & (dd > 0) & (dd < zh - np.maximum(OCC_MARGIN_MM, 0.05 * zh))
            o = cv2.morphologyEx(o.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            occ = cv2.dilate(o, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), occ.astype(np.uint8) * 255)
        return occ

    def saturated(self, img: np.ndarray, grow_px: int = 2) -> np.ndarray:
        s = (img >= SATURATED).any(-1)
        return cv2.dilate(s.astype(np.uint8), np.ones((2 * grow_px + 1,) * 2, np.uint8)) > 0 if grow_px else s

    # ------------------------------------------------------------------ depth
    def depth(self, i: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """LiDAR depth (mm, float32, 192 x 256) and confidence (0/1/2) of frame ``i``; (None, None) when
        the recording has no depth for that frame (Stray drops one now and then: 29d760c369 frame 380)."""
        raw = cv2.imread(str(self.root / "depth" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None, None
        d = raw.astype(np.float32)
        cp = self.root / "confidence" / f"{i:06d}.png"
        c = cv2.imread(str(cp), cv2.IMREAD_UNCHANGED) if cp.exists() else np.full(d.shape, 2, np.uint8)
        return d, c

    @property
    def faces(self) -> list[str]:
        """The faces this recording textured: the one that was up, and the other when the run used it
        (``pipeline.other_face_seen``). Candidates model the view-dependent look on ``face``; the other
        face shows the inspection texture."""
        other = self.run_report["passes"][self.pass_index].get("other_face") or {}
        return [self.face] + ([other["face"]] if other.get("use") else [])

    # ------------------------------------------------------------------ summary for the comparison page
    def to_json(self) -> dict:
        spec = self.model.spec
        pins = [{"id": d["id"], "label": d.get("label_ko", ""), "where": d.get("location_ko", ""), "chart": d.get("chart"),
                 "centroid_mm": d.get("centroid_mm"), "normal": d.get("normal")} for d in self.defects.get("defects", [])]
        return {"rec": self.rec, "face": self.face, "faces": self.faces, "base_run": str(self.base_run),
                "device": {"length_mm": spec.length, "width_mm": spec.width, "thickness_mm": spec.thickness,
                           "z_front": self.model.z_front, "z_back": self.model.z_back},
                "frames": {"total": len(self.cap), "train_cached": len(self.train_frames), "test_scored": len(self.test_frames),
                           "held_out": len(self.split.excluded)},
                "defects": pins}


def plate_silhouette(pts: np.ndarray, view: View, grow_px: float = 0.0) -> np.ndarray:
    T = inv_T(np.asarray(view.T_cad_cam, float))
    pc = pts @ T[:3, :3].T + T[:3, 3]
    m = np.zeros((view.H, view.W), np.uint8)
    front = pc[:, 2] > 1.0
    if front.sum() < 3:
        return m > 0
    pc = pc[front]
    uv = np.stack([view.K[0, 0] * pc[:, 0] / pc[:, 2] + view.K[0, 2], view.K[1, 1] * pc[:, 1] / pc[:, 2] + view.K[1, 2]], 1)
    hull = cv2.convexHull(np.round(uv * 4).astype(np.int32))
    cv2.fillConvexPoly(m, hull, 1, lineType=cv2.LINE_8, shift=2)
    if grow_px > 0:
        r = int(np.ceil(grow_px))
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    elif grow_px < 0:
        r = int(np.ceil(-grow_px))
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return m > 0


# ---------------------------------------------------------------------- public videos (uCO3D)
@dataclass
class Uco3dCapture:
    """What the recon code reads of a capture, for a uCO3D sequence: frames are its annotated ones
    (index ``i`` = position in ``seq.frame_numbers``)."""
    seq: object

    def __len__(self) -> int:
        return len(self.seq.frame_numbers)

    @property
    def timestamps(self) -> np.ndarray:
        return np.asarray(self.seq.timestamps, float)

    def camera(self, i: int) -> Camera:
        return self.seq.camera(self.seq.frame_numbers[i])


@dataclass
class Uco3dContext(CaptureContext):
    """A public phone video (uCO3D, processed by ``pipeline.run_uco3d``): SfM cameras scaled to mm the
    way the run did (``scale_mm_per_sfm_unit``), frames undistorted (radial k1), no LiDAR. What stands
    in front of the device (the hand holding it) comes from the dataset's own object masks instead:
    plate pixels outside the object's mask are occluded; a frame whose mask covers less than half of
    the plate keeps nothing out (the mask missed the device)."""

    @property
    def n_video(self) -> int:
        return len(self.cap)

    def full_frames(self, frames: list[int], scale: float = 1.0, chunk: int = 24):
        from ..io.uco3d import read_frames

        seq = self.cap.seq
        frames = sorted(int(i) for i in frames)
        for a in range(0, len(frames), chunk):
            part = frames[a:a + chunk]
            imgs = read_frames(seq, [seq.frame_numbers[i] for i in part], undistorted=True)
            for i in part:
                img = imgs[seq.frame_numbers[i]]
                if scale != 1.0:
                    img = cv2.resize(img, (int(round(img.shape[1] * scale)), int(round(img.shape[0] * scale))),
                                     interpolation=cv2.INTER_AREA)
                yield i, img

    def depth(self, i: int):
        return None, None

    def occluded(self, i: int) -> np.ndarray:
        p = self.out / "ctx" / "occ" / f"{i:05d}.png"
        if p.exists():
            return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0
        from ..io.uco3d import read_frames

        v = self.view(i)
        occ = np.zeros((v.H, v.W), bool)
        seq = self.cap.seq
        try:
            m = read_frames(seq, [seq.frame_numbers[i]], undistorted=True, masks=True)[seq.frame_numbers[i]]
            m = cv2.resize(m.astype(np.uint8), (v.W, v.H), interpolation=cv2.INTER_NEAREST) > 0
            sil = self.silhouette(v, 0.0, visible=False)
            if sil.sum() and (m & sil).sum() >= 0.5 * sil.sum():
                o = sil & ~cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                o = cv2.morphologyEx(o.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
                occ = cv2.dilate(o, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
        except Exception as e:      # no mask for this frame: nothing kept out
            log.warning(f"[{self.rec}] frame {i}: no object mask ({type(e).__name__})")
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), occ.astype(np.uint8) * 255)
        return occ

    @classmethod
    def from_uco3d_run(cls, base_run: str | Path, out_root: str | Path, split: Split | None = None,
                       scale: float = SCALE) -> "Uco3dContext":
        from ..anchor.template import model_of_run
        from ..io.uco3d import load_sequence

        base_run = Path(base_run)
        rep = json.loads((base_run / "run.json").read_text())
        p = rep["passes"][0]
        name = str(p["source"]).split(":", 1)[1]
        root = Path(p["root"])
        seq = load_sequence(root.parent.parent.parent, name)
        k = float(p["template"]["scale_mm_per_sfm_unit"])
        T_world_cad = np.asarray(json.loads((base_run / "passes" / p["name"] / "registration.json").read_text())["T_world_cad"], float)
        Tinv = inv_T(T_world_cad)
        T_cad_cam = {}
        for i, fn in enumerate(seq.frame_numbers):
            T = np.asarray(seq.T_world_cam[fn], float).copy()
            T[:3, 3] *= k
            T_cad_cam[i] = Tinv @ T
        cap = Uco3dCapture(seq)
        if split is None:
            from .service import service_split

            split = service_split(cap.timestamps)
        ctx = cls(rec=name, root=root, base_run=base_run, out=Path(out_root) / name, face=p["face_up"], cap=cap, split=split,
                  T_cad_cam=T_cad_cam, model=model_of_run(rep), run_report=rep,
                  defects=json.loads((base_run / "bundle" / "defects.json").read_text()), scale=scale, train_step=1)
        ctx.sfm_scale, ctx.T_world_cad = k, T_world_cad
        ctx.out.mkdir(parents=True, exist_ok=True)
        return ctx

    def sfm_points(self, segmented: bool = False) -> np.ndarray:
        """The dataset's SfM point cloud in the device frame, mm (no LiDAR here: what a splatting
        candidate starts from, ``c5_free_gs``). Without the full cloud (``point_cloud.ply`` is not
        among the downloaded files), the positions of the dataset's own splat fit of the scene."""
        from ..io.uco3d import read_points, read_splat_means

        seq = self.cap.seq
        if segmented or (seq.root / "point_cloud.ply").exists():
            P = read_points(seq, segmented=segmented)
        else:
            P = read_splat_means(seq)
        P = P * self.sfm_scale
        T = inv_T(self.T_world_cad)
        return P @ T[:3, :3].T + T[:3, 3]


def context_for_pass(base_run: str | Path, pass_index: int, out_root: str | Path, split: Split | None = None) -> CaptureContext:
    """The capture context of one pass of a run: a Stray Scanner recording, or a public uCO3D video
    (``source: uco3d:<name>``)."""
    rep = json.loads((Path(base_run) / "run.json").read_text())
    p = rep["passes"][pass_index]
    if str(p.get("source", "")).startswith("uco3d:"):
        return Uco3dContext.from_uco3d_run(base_run, out_root, split=split)
    if split is None:
        from ..io.stray import read_stray
        from .service import service_split

        split = service_split(read_stray(Path(p["root"])).timestamps)
    return CaptureContext.from_run(base_run, out_root=out_root, pass_index=pass_index, split=split)
