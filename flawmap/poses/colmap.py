"""Camera poses refined by structure from motion (COLMAP) on the keyframes, in the pass's own world.

The boardless path ties every keyframe to the device outline through ARKit plus a smooth drift; the
outline fits to ~0.2 mm, but single frames stay off by millimetres inside the faces (median 2.1 mm,
p90 7 mm on the first real iPad capture, measured by aligning rectified views). Bundle adjustment on
the floor's texture fixes each frame on its own: on that capture COLMAP registered all 101 keyframes
(36,882 points, 0.76 px) and the sharp views agreed to 2.6 mm p90 instead of 7.0 mm.

COLMAP's scene has its own scale and frame, so it is mapped onto the input poses by the similarity
that best matches the camera centres (Umeyama). The device is registered again afterwards by the
caller. Nothing changes when COLMAP registers too few frames or the fit is poor.

Needs ``pycolmap`` (4.2.1 on the team server and the Windows env).
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from ..geometry.camera import Camera


def similarity(A: np.ndarray, B: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """(s, R, t) minimising |s R A + t - B| over point pairs (Umeyama)."""
    ma, mb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((B - mb).T @ (A - ma) / len(A))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / ((A - ma) ** 2).sum(1).mean())
    return s, R, mb - s * R @ ma


def refine_with_colmap(frames: dict[int, np.ndarray], cams: dict[int, Camera], T_world_cam: dict[int, np.ndarray],
                       min_registered: float = 0.8, max_reproj_px: float = 1.5, max_centre_mm: float = 30.0,
                       threads: int = 4, matching: str = "sequential", overlap: int = 12,
                       log=print) -> tuple[dict[int, np.ndarray] | None, dict]:
    """New ``T_world_cam`` for the frames COLMAP registered (others keep theirs), or None with the
    reason in the info when the reconstruction does not pass the checks."""
    try:
        import pycolmap
    except ImportError:
        return None, {"used": False, "reason": "pycolmap not installed"}
    ids = sorted(frames)
    work = Path(tempfile.mkdtemp(prefix="flawmap_colmap_"))     # ASCII path: COLMAP and non-ASCII folders
    try:
        img_dir = work / "images"
        img_dir.mkdir()
        for i in ids:
            cv2.imwrite(str(img_dir / f"{i:06d}.jpg"), cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        K = np.median(np.stack([cams[i].K for i in ids]), 0)
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "PINHOLE"
        reader.camera_params = f"{K[0, 0]},{K[1, 1]},{K[0, 2]},{K[1, 2]}"
        db = work / "db.db"
        # few threads and capped features: with the pass's keyframes already in memory, COLMAP's default
        # (one thread per core, each with its own image pyramid) took a 32 GB PC under the memory
        # guard's floor (2026-11-17)
        ext = pycolmap.FeatureExtractionOptions()
        ext.num_threads = threads
        ext.sift.max_num_features = 8192
        pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                                  extraction_options=ext)
        mat = pycolmap.FeatureMatchingOptions()
        mat.num_threads = threads
        if matching == "exhaustive":
            pycolmap.match_exhaustive(db, matching_options=mat)
        else:
            # keyframes are in time order: each is matched with its neighbours and, through the
            # quadratic overlap, with frames 2, 4, 8, ... apart, which closes the orbit's loop. The
            # exhaustive pairs (5,050 for 101 keyframes) took 2-5 min per block of 34 on 4 threads.
            pair = pycolmap.SequentialPairingOptions()
            pair.overlap = overlap
            pair.quadratic_overlap = True
            pycolmap.match_sequential(db, matching_options=mat, pairing_options=pair)
        opts = pycolmap.IncrementalPipelineOptions()
        opts.ba_refine_focal_length = False
        opts.ba_refine_principal_point = False
        opts.ba_refine_extra_params = False
        recs = pycolmap.incremental_mapping(db, img_dir, work / "sparse", options=opts)
        if not recs:
            return None, {"used": False, "reason": "no reconstruction"}
        rec = max(recs.values(), key=lambda r: r.num_reg_images())
        cm = {}
        for im in rec.images.values():
            cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
            T = np.eye(4)
            T[:3, :4] = cfw.matrix()
            cm[int(Path(im.name).stem)] = np.linalg.inv(T)
        reproj = float(rec.compute_mean_reprojection_error())
        info = {"used": False, "registered": len(cm), "keyframes": len(ids), "points": int(rec.num_points3D()),
                "reproj_px": round(reproj, 3)}
        if len(cm) < min_registered * len(ids):
            info["reason"] = f"registered {len(cm)}/{len(ids)}"
            return None, info
        if reproj > max_reproj_px:
            info["reason"] = f"reprojection {reproj:.2f} px"
            return None, info
        common = sorted(set(cm) & set(T_world_cam))
        A = np.array([cm[i][:3, 3] for i in common])
        B = np.array([T_world_cam[i][:3, 3] for i in common])
        s, R, t = similarity(A, B)
        res = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)
        info.update(scale_mm_per_unit=round(s, 3), centre_residual_mm_p50=round(float(np.median(res)), 2),
                    centre_residual_mm_p90=round(float(np.percentile(res, 90)), 2))
        if np.median(res) > max_centre_mm:
            info["reason"] = f"camera centres disagree with the input poses by {np.median(res):.1f} mm"
            return None, info
        out = dict(T_world_cam)
        for i in common:
            T = cm[i].copy()
            T[:3, :3] = R @ T[:3, :3]
            T[:3, 3] = s * R @ cm[i][:3, 3] + t
            out[i] = T
        info["used"] = True
        log(f"  colmap: {len(cm)}/{len(ids)} keyframes, {info['points']} points, {reproj:.2f} px; "
            f"camera centres moved p50 {info['centre_residual_mm_p50']} mm, p90 {info['centre_residual_mm_p90']} mm")
        return out, info
    except Exception as e:      # COLMAP failing must never stop the pass: the input poses stay
        return None, {"used": False, "reason": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)
