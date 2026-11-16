"""Public dataset downloads with a disk floor (``python -m flawmap.io.download <dataset>``).

Only data that can be fetched without a login (or with a key the provider publishes) is handled
here. Every download checks free disk space first and between files: the user allows collecting
until 50 GB remain, and 30 GB more is kept for processing (``FLAWMAP_DISK_START_GB`` default 80,
``FLAWMAP_DISK_STOP_GB`` default 60). Archives are treated as untrusted input: tar members are
extracted with the ``data`` filter and zip members through ``zipfile`` (which drops absolute and
``..`` paths), and archives are deleted after extraction.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import time
import zipfile
from pathlib import Path

from ..utils.io import data_root, write_json
from ..utils.logging import get_logger

log = get_logger("flawmap.download")

START_GB = float(os.environ.get("FLAWMAP_DISK_START_GB", 80))
STOP_GB = float(os.environ.get("FLAWMAP_DISK_STOP_GB", 60))


class DiskFloor(RuntimeError):
    pass


def free_gb(path: Path) -> float:
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


def check_space(path: Path, expected_gb: float = 0.0, start: bool = False) -> float:
    free = free_gb(path)
    limit = START_GB if start else STOP_GB
    if free - expected_gb < limit:
        raise DiskFloor(f"{free:.1f} GB free, {expected_gb:.1f} GB more expected: below the {limit:.0f} GB floor")
    return free


def http_download(url: str, dest: Path, expected_bytes: int | None = None, chunk: int = 8 << 20, retries: int = 8) -> Path:
    """Resumable streaming download (HTTP Range)."""
    import requests

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and (expected_bytes is None or dest.stat().st_size == expected_bytes):
        return dest
    for attempt in range(retries):
        done = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
                if r.status_code == 416:      # already complete
                    break
                r.raise_for_status()
                mode = "ab" if (done and r.status_code == 206) else "wb"
                total = expected_bytes or (int(r.headers.get("Content-Length", 0)) + (done if mode == "ab" else 0))
                t0, last = time.time(), 0.0
                with open(part, mode) as f:
                    for block in r.iter_content(chunk):
                        f.write(block)
                        if time.time() - last > 30:
                            last = time.time()
                            check_space(dest.parent)
                            got = part.stat().st_size
                            log.info(f"{dest.name}: {got/1e9:.2f}/{total/1e9:.2f} GB ({got/1e6/max(time.time()-t0,1):.1f} MB/s)")
            break
        except DiskFloor:
            raise
        except Exception as e:  # network hiccup: resume
            log.warning(f"{dest.name}: attempt {attempt + 1} failed: {e}")
            time.sleep(min(60, 5 * 2 ** attempt))
    if expected_bytes is not None and part.stat().st_size != expected_bytes:
        raise RuntimeError(f"{dest.name}: size {part.stat().st_size} != expected {expected_bytes}")
    part.replace(dest)
    return dest


def safe_extract_tar(archive: Path, out: Path) -> None:
    with tarfile.open(archive) as t:
        t.extractall(out, filter="data")


def safe_extract_zip(archive: Path, out: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        z.extractall(out)


def record(out: Path, entry: dict) -> None:
    """Append a provenance record next to the data."""
    path = Path(out) / "_downloads.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    rows.append(entry)
    write_json(path, rows)


# ----------------------------------------------------------------------------- uCO3D
UCO3D_REPO = "https://huggingface.co/datasets/facebook/uco3d/resolve/main"
UCO3D_INDEX = "https://raw.githubusercontent.com/facebookresearch/uco3d/main/dataset_download/category_to_archives.json"


def uco3d(category: str = "cellular_telephone", super_category: str = "communication_devices",
          kinds: tuple[str, ...] = ("rgb_videos", "mask_videos", "gaussian_splats", "segmented_point_clouds"),
          metadata: bool = True, out: Path | None = None, sequences: list[str] | None = None) -> None:
    """Extract one uCO3D category from the super-category archives with HTTP range requests
    (the official downloader only fetches whole super-categories: 88 GB of video for phones' group).

    ``sequences`` limits the extraction to those sequence names (the demo needs five); then only
    ``metadata.sqlite`` is unpacked from the metadata archive (the 9.6 GB ``set_lists`` are not
    used). The metadata archive (4.9 GB) is skipped when ``metadata.sqlite`` is already there."""
    import requests
    from remotezip import RemoteZip

    out = Path(out or data_root() / "datasets" / "uco3d")
    out.mkdir(parents=True, exist_ok=True)
    idx_path = out / "category_to_archives.json"
    if not idx_path.exists():
        idx_path.write_text(requests.get(UCO3D_INDEX, timeout=60).text)
    index = json.loads(idx_path.read_text())
    check_space(out, 60.0 if not sequences else 25.0, start=True)
    prefix = f"{super_category}/{category}/"
    prefixes = tuple(f"{prefix}{n}/" for n in sequences) if sequences else (prefix,)
    if metadata and (out / "metadata.sqlite").exists():
        log.info("uco3d metadata.sqlite already there: metadata archive skipped")
    elif metadata:
        meta = index["metadata"]
        zpath = http_download(f"{UCO3D_REPO}/{meta['filename']}", out / "_archives" / meta["filename"], meta["file_size_bytes"])
        if sequences:
            with zipfile.ZipFile(zpath) as z:
                want = [i for i in z.infolist() if i.filename.endswith("metadata.sqlite")]
                for i in want:
                    z.extract(i, out)
                if not want:
                    z.extractall(out)
        else:
            safe_extract_zip(zpath, out)
        zpath.unlink()
        record(out, {"what": "uco3d metadata.zip", "sha256": meta["sha256sum"], "bytes": meta["file_size_bytes"],
                     "date": time.strftime("%Y-%m-%d %H:%M"), "free_gb_after": round(free_gb(out), 1)})
    for kind in kinds:
        archives = index[kind][super_category]
        n_files, n_bytes = 0, 0
        for name in archives:
            url = f"{UCO3D_REPO}/{name}"
            for attempt in range(6):
                try:
                    with RemoteZip(url) as z:
                        members = [i for i in z.infolist() if i.filename.startswith(prefixes) and not i.is_dir()]
                        for k, info in enumerate(members):
                            target = out / info.filename
                            if target.exists() and target.stat().st_size == info.file_size:
                                continue
                            if k % 50 == 0:
                                check_space(out)
                            z.extract(info, out)
                            n_files += 1
                            n_bytes += info.file_size
                    break
                except DiskFloor:
                    raise
                except Exception as e:
                    log.warning(f"{name}: attempt {attempt + 1} failed: {e}")
                    time.sleep(min(120, 10 * 2 ** attempt))
            log.info(f"uco3d {kind} {name}: done ({n_files} files, {n_bytes/1e9:.2f} GB so far, {free_gb(out):.0f} GB free)")
        record(out, {"what": f"uco3d {kind} {category}" + (f" ({len(sequences)} sequences)" if sequences else ""),
                     "archives": list(archives), "files": n_files, "bytes": n_bytes,
                     "date": time.strftime("%Y-%m-%d %H:%M"), "free_gb_after": round(free_gb(out), 1)})


# ----------------------------------------------------------------------------- OmniObject3D
# Public Google Drive files of the official release (videos_processed and raw_scans per category).
OMNI_DRIVE = {
    "remote_control": ("1VhJjNble2YGnxbgcxGxQWINroSQO-f26", "1Wzxrza_Uzw0M8frER5SX9ticvEE5TVWW"),
    "calculator": ("1ZMPFJ0iDP85-ASnTl_nQyzh_jVLI4_SE", "1wPbwvBcdSTW4YqUJCSfl7psEHS-6Opyy"),
    "mouse": ("1y1AqyfQ8WvEKGqCOoO_QN5za1Z4pyqvj", "1PHFWzvZYA-JAniyVAXTwc6MgbaLUXdlw"),
    "keyboard": ("1Sq6o4K7Hp6D7WZVPsBtmyNYaFOu3KgnL", "1T5qgKjjaAPsSyNofe3lOdQa7ifYcJt10"),
    "watch": ("1jkUFxzXdxfbmjRcRLE3_UnDyU4KFaySK", "1kFPeIDY5mFa0oJOwaSLSFwQ-MaLQx2-I"),
    "laptop": ("1QRUZP2BNl5fImZHU_m1OpIPEONg79ryZ", "1IHhZVOE2iJ3nHMfuEfRYoP4jt262A2HU"),
    "monitor": ("1M0sf4oW8d13V6fSfjbkv5zgdgeF4BPRb", "1lPpuT1Cr5RM8pRBBIf9_Z-egrG8BBTtA"),
}


def omniobject3d(categories: list[str] | None = None, out: Path | None = None, attempts: int = 5) -> None:
    import gdown

    out = Path(out or data_root() / "datasets" / "omniobject3d")
    out.mkdir(parents=True, exist_ok=True)
    check_space(out, 20.0, start=True)
    for cat in categories or list(OMNI_DRIVE):
        for kind, fid in zip(("videos_processed", "raw_scans"), OMNI_DRIVE[cat]):
            done = out / kind / f".{cat}.done"
            if done.exists():
                continue
            check_space(out, 4.0)
            arc = out / "_archives" / f"{kind}_{cat}.tar.gz"
            arc.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(attempts):
                try:
                    gdown.download(id=fid, output=str(arc), quiet=True, resume=True)
                    break
                except Exception as e:  # Drive quota ("too many users") or network
                    msg = str(e).splitlines()[0] if str(e) else repr(e)
                    log.warning(f"omniobject3d {kind}/{cat}: attempt {attempt + 1} failed: {msg}")
                    if attempt + 1 < attempts:
                        time.sleep(min(600, 30 * 2 ** attempt))
            if not arc.exists():
                log.error(f"omniobject3d {kind}/{cat}: not downloaded")
                continue
            size = arc.stat().st_size
            (out / kind).mkdir(parents=True, exist_ok=True)
            safe_extract_tar(arc, out / kind)
            arc.unlink()
            done.write_text(time.strftime("%Y-%m-%d %H:%M"))
            record(out, {"what": f"omniobject3d {kind} {cat}", "drive_id": fid, "bytes": size,
                         "date": time.strftime("%Y-%m-%d %H:%M"), "free_gb_after": round(free_gb(out), 1)})
            log.info(f"omniobject3d {kind}/{cat}: {size/1e9:.2f} GB, {free_gb(out):.0f} GB free")


# ----------------------------------------------------------------------------- MobileBrick
MOBILEBRICK_URL = "https://www.robots.ox.ac.uk/~victor/data/MobileBrick/MobileBrick_Mar23.zip"
MOBILEBRICK_BYTES = 13128305497


def mobilebrick(out: Path | None = None) -> None:
    out = Path(out or data_root() / "datasets" / "mobilebrick")
    out.mkdir(parents=True, exist_ok=True)
    if (out / ".done").exists():
        return
    check_space(out, 2 * MOBILEBRICK_BYTES / 1e9, start=True)
    z = http_download(MOBILEBRICK_URL, out / "_archives" / "MobileBrick_Mar23.zip", MOBILEBRICK_BYTES)
    safe_extract_zip(z, out)
    z.unlink()
    (out / ".done").write_text(time.strftime("%Y-%m-%d %H:%M"))
    record(out, {"what": "MobileBrick_Mar23.zip", "url": MOBILEBRICK_URL, "bytes": MOBILEBRICK_BYTES,
                 "date": time.strftime("%Y-%m-%d %H:%M"), "free_gb_after": round(free_gb(out), 1)})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m flawmap.io.download")
    p.add_argument("dataset", choices=["uco3d", "omniobject3d", "mobilebrick"])
    p.add_argument("--category", default=None)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--no-metadata", action="store_true")
    p.add_argument("--attempts", type=int, default=5)
    p.add_argument("--sequences", nargs="*", default=None, help="uco3d: only these sequence names (e.g. the demo list)")
    p.add_argument("--kinds", nargs="*", default=None,
                   help="uco3d: rgb_videos mask_videos gaussian_splats segmented_point_clouds (default: all four)")
    a = p.parse_args(argv)
    try:
        if a.dataset == "uco3d":
            kw = {"kinds": tuple(a.kinds)} if a.kinds else {}
            uco3d(category=a.category or "cellular_telephone", metadata=not a.no_metadata, sequences=a.sequences, **kw)
        elif a.dataset == "omniobject3d":
            omniobject3d(a.categories, attempts=a.attempts)
        else:
            mobilebrick()
    except DiskFloor as e:
        log.error(f"stopped: {e}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
