"""Taking in an uploaded capture: save it, unpack it safely, find the Stray Scanner folder, check it.

The seller compresses the Stray Scanner recording folder in the iOS Files app and uploads the zip.
Everything here runs before the job is queued and takes seconds, so a wrong file is rejected while
the seller is still on the page, with a hint on how to film again.
"""
from __future__ import annotations

import hashlib
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

from ..errors import FlawmapError

MAX_UPLOAD_BYTES = 3 * 1024 ** 3
MAX_UNPACKED_BYTES = 4 * 1024 ** 3
MAX_FILES = 20_000
MIN_FREE_BYTES = 80 * 1024 ** 3          # 50 GB floor + 30 GB working margin (project disk rule)
STRAY_FILES = ("rgb.mp4", "odometry.csv", "camera_matrix.csv")
MIN_FRAMES, WARN_SECONDS = 16, 30.0


def check_disk(path: Path, need: int = 0) -> None:
    free = shutil.disk_usage(path).free
    if free - need < MIN_FREE_BYTES:
        raise FlawmapError("DISK_FULL", f"{free / 1e9:.0f} GB free")


def save_stream(src, dst: Path, max_bytes: int = MAX_UPLOAD_BYTES, chunk: int = 4 << 20) -> tuple[int, str]:
    """Copy a file-like object to ``dst`` with a size cap; returns (bytes, sha256 hex)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    h, n = hashlib.sha256(), 0
    with open(dst, "wb") as f:
        while True:
            b = src.read(chunk)
            if not b:
                break
            n += len(b)
            if n > max_bytes:
                f.close()
                dst.unlink(missing_ok=True)
                raise FlawmapError("UPLOAD_INVALID", f"upload larger than {max_bytes / 1e9:.1f} GB")
            h.update(b)
            f.write(b)
    return n, h.hexdigest()


def sha256_file(path: Path, chunk: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _member_ok(info: zipfile.ZipInfo) -> bool:
    name = info.filename
    p = PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or (len(name) > 1 and name[1] == ":"):
        return False
    mode = (info.external_attr >> 16) & 0xFFFF
    return not stat.S_ISLNK(mode)


def safe_extract(zip_path: Path, dst: Path, max_bytes: int = MAX_UNPACKED_BYTES, max_files: int = MAX_FILES) -> int:
    """Unpack ``zip_path`` into ``dst``; refuses absolute paths, ``..``, symlinks and archives that
    unpack to more than ``max_bytes`` or ``max_files``. macOS metadata entries are skipped."""
    try:
        zf = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError) as e:
        raise FlawmapError("UPLOAD_INVALID", f"not a zip file: {e}") from e
    with zf:
        infos = [i for i in zf.infolist() if not i.filename.startswith("__MACOSX/") and not i.filename.endswith(".DS_Store")]
        bad = [i.filename for i in infos if not _member_ok(i)]
        if bad:
            raise FlawmapError("UPLOAD_INVALID", f"unsafe paths in zip: {bad[:3]}")
        if len(infos) > max_files:
            raise FlawmapError("UPLOAD_INVALID", f"{len(infos)} files in zip (max {max_files})")
        total = sum(i.file_size for i in infos)
        if total > max_bytes:
            raise FlawmapError("UPLOAD_INVALID", f"zip unpacks to {total / 1e9:.1f} GB (max {max_bytes / 1e9:.1f})")
        dst.mkdir(parents=True, exist_ok=True)
        root = dst.resolve()
        for i in infos:
            target = (dst / i.filename).resolve()
            if root != target and root not in target.parents:
                raise FlawmapError("UPLOAD_INVALID", f"path escapes the folder: {i.filename}")
            if i.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(i) as s, open(target, "wb") as d:
                shutil.copyfileobj(s, d, 4 << 20)
    return total


def find_stray_root(folder: Path, max_depth: int = 3) -> Path:
    """The folder holding ``rgb.mp4``, ``odometry.csv`` and ``camera_matrix.csv`` (the zip may
    contain the recording folder itself or its contents)."""
    folder = Path(folder)
    cands = [folder] + [p for p in folder.rglob("*") if p.is_dir() and len(p.relative_to(folder).parts) <= max_depth]
    for c in cands:
        if all((c / f).is_file() for f in STRAY_FILES):
            return c
    found = sorted({f.name for f in folder.rglob("*") if f.is_file()})[:8]
    raise FlawmapError("NOT_STRAY", f"no folder with {', '.join(STRAY_FILES)} (found {found})")


def quick_check(root: Path) -> dict:
    """Seconds-long check of a Stray folder: odometry rows, intrinsics, video size, duration."""
    from ..io.stray import read_stray

    try:
        cap = read_stray(root)
    except FlawmapError:
        raise
    except Exception as e:
        raise FlawmapError("NOT_STRAY", f"cannot read the recording: {type(e).__name__}: {e}") from e
    n = len(cap)
    dur = float(cap.timestamps[-1] - cap.timestamps[0]) if n > 1 else 0.0
    if n < MIN_FRAMES:
        raise FlawmapError("VIDEO_SHORT", f"{n} frames, {dur:.1f} s")
    if not cap.width or not cap.height:
        raise FlawmapError("NOT_STRAY", "rgb.mp4 has no decodable video stream")
    orbit = orbit_coverage(cap.T_ark_cam)
    warnings = []
    if cap.info.get("warning"):
        warnings.append(cap.info["warning"])
    if orbit < 0.75:
        warnings.append(f"기기 둘레의 {round(orbit * 100)}%만 돌며 찍었습니다. 찍지 않은 쪽은 확인하지 못한 곳으로 남습니다.")
    if dur < WARN_SECONDS:
        warnings.append(f"짧은 촬영({dur:.0f}초)입니다. 흔들린 장면이 많으면 일부 부위가 덜 찍힐 수 있습니다.")
    return {"n_frames": n, "duration_s": round(dur, 1), "width": cap.width, "height": cap.height,
            "has_depth": bool(cap.has_depth), "codec": cap.info.get("codec"), "orbit": round(orbit, 2), "warnings": warnings}


def orbit_coverage(T_ark_cam, bins: int = 12) -> float:
    """Share of 30-degree sectors around the device the camera passed through. ARKit's world y
    axis is up, so the orbit lies in x-z; the device sits where the optical axes meet."""
    import numpy as np

    T = np.asarray(T_ark_cam)
    if len(T) < 3:
        return 0.0
    c, z = T[:, :3, 3], T[:, :3, 2]            # camera centres and optical axes (OpenCV convention)
    A, b = np.zeros((3, 3)), np.zeros(3)
    for ci, zi in zip(c, z / np.linalg.norm(z, axis=1, keepdims=True)):
        P = np.eye(3) - np.outer(zi, zi)
        A += P
        b += P @ ci
    try:
        centre = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        centre = c.mean(0)
    d = c - centre
    az = np.degrees(np.arctan2(d[:, 2], d[:, 0])) % 360
    return float(len(np.unique((az // (360 / bins)).astype(int))) / bins)


def take_zip(upload, dst_dir: Path) -> dict:
    """Save an uploaded zip (file-like ``upload``), unpack it under ``dst_dir`` and check it.
    Returns the pass record stored with the inspection."""
    check_disk(dst_dir.parent if dst_dir.parent.exists() else Path("."))
    z = dst_dir.with_suffix(".zip")
    nbytes, digest = save_stream(upload, z)
    try:
        safe_extract(z, dst_dir)
        root = find_stray_root(dst_dir)
        chk = quick_check(root)
    except Exception:
        shutil.rmtree(dst_dir, ignore_errors=True)
        raise
    finally:
        z.unlink(missing_ok=True)
    # the report shows the hash of the video itself (independent of how the folder was zipped)
    return {"root": str(root), "bytes": nbytes, "zip_sha256": digest, "sha256": sha256_file(root / "rgb.mp4"), "check": chk}


def take_folder(src: Path, dst_dir: Path) -> dict:
    """Copy a capture folder that is already on the server (demo samples) and check it."""
    check_disk(dst_dir.parent if dst_dir.parent.exists() else Path("."))
    root = find_stray_root(src)
    shutil.copytree(root, dst_dir)
    chk = quick_check(dst_dir)
    return {"root": str(dst_dir), "bytes": sum(f.stat().st_size for f in dst_dir.rglob("*") if f.is_file()),
            "sha256": sha256_file(dst_dir / "rgb.mp4"), "check": chk, "sample": src.name}
