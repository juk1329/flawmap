"""Blind rating files (``flawmap compare-blind``): a set of candidates of one recording under shuffled
letters, as videos and pictures to look at directly, for when the comparison page cannot be reached.
The letters are drawn afresh for every recording and kept apart in a key file, to be opened only after
the ratings are given.

    <out>/<pack>/README.txt
    <out>/<pack>/<rec>/1_turntable.mp4     every letter turning together (one turn at 30 degrees up)
    <out>/<pack>/<rec>/2_test_frames.jpg   held-out frames: the real frame, then every letter's render of that pose
    <out>/<pack>/<rec>/3_turn_views.jpg    a row per letter: a turn at 60 degrees, grazing, the unfilmed side
    <out>/<pack>/<rec>/4_defects.jpg       a row per letter: the middle of the face and each defect from 70 mm
    <out>/<pack>/<rec>/each/<letter>.mp4   each letter's own turntable
    <out>/<pack>.key.json                  letter -> candidate for every recording

Only letters and view names are written on the pictures: no ids, labels or scores. The turntables and
the stress views are the ones ``flawmap compare`` rendered; the test frames are rendered again over one
common grey (``rerender_test_frames``), since the saved ones differ in background from candidate to candidate."""
from __future__ import annotations

import json
import math
import random
import secrets
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

LETTERS = "ABCDEFGH"
TURN_VIEWS = [("ring60_el60_az0", "turn 60, 0 deg"), ("ring60_el60_az90", "turn 60, 90 deg"),
              ("ring60_el60_az180", "turn 60, 180 deg"), ("ring60_el60_az270", "turn 60, 270 deg"),
              ("graze_el8_az60", "grazing 8 deg"), ("graze_el3_az240", "grazing 3 deg"),
              ("under_el-40_az0", "unfilmed side 40"), ("under_el-75_az90", "unfilmed side 75")]
GREY = 235

README = """블라인드 평가 묶음 ({pack})

녹화마다 후보를 글자(A, B, ...)로 가리고 섞었습니다. 글자와 후보의 짝은 녹화마다 따로 뽑았습니다.
글자의 정체는 이 폴더 밖의 열쇠 파일({key})에 있습니다. 평가를 마치기 전에는 열지 마세요.

녹화 폴더마다:
  1_turntable.mp4    모든 글자가 함께 한 바퀴 돕니다(위에서 30도). 돌려도 깨지지 않는지, 반사가 자연스럽게 움직이는지 봅니다.
  2_test_frames.jpg  학습에서 뺀 시험 프레임입니다. 줄마다 맨 왼쪽이 실제 프레임이고, 그 옆이 글자마다 같은 자세의 렌더입니다.
                     실제와 얼마나 비슷한지 봅니다. 렌더의 배경 회색은 모든 글자가 같습니다(구별 단서가 아님).
  3_turn_views.jpg   글자마다 한 줄: 60도로 돌린 네 방향, 스치게 본 두 각도, 찍지 않은 면 두 각도.
  4_defects.jpg      글자마다 한 줄: 면 가운데와 하자마다 70 mm 앞에서 본 모습. 하자는 노란 고리 안(칸 한가운데)입니다.
                     하자가 잘 보이는지 봅니다.
  each/<글자>.mp4    글자 하나의 회전 영상(크게 보기).

평가(녹화마다, 글자마다 1-5점 + 메모):
  사실감 / 돌려도 깨지지 않음 / 하자가 잘 보임
  예) 29d7: A 4/5/3 "로고가 실제처럼 어두움", B 2/5/2 ...  또는 순위만: 29d7: C > A > E > B > D

녹화: {recs}
"""


def assign(cands: list[str], rng: random.Random) -> dict[str, str]:
    """Letters for a set of candidates in a random order: {"A": cid, ...}."""
    if len(cands) > len(LETTERS):
        raise ValueError(f"at most {len(LETTERS)} candidates")
    ids = list(cands)
    rng.shuffle(ids)
    return {LETTERS[i]: c for i, c in enumerate(ids)}


def _blank(w: int, h: int) -> np.ndarray:
    return np.full((h, w, 3), GREY, np.uint8)


def _label(img: np.ndarray, text: str, scale: float | None = None) -> np.ndarray:
    """``text`` in a white box at the top left (ASCII only, cv2)."""
    h = img.shape[0]
    s = scale or max(0.6, h / 240 * 0.7)
    th = max(1, int(round(s * 2)))
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, th)
    pad = int(6 * s)
    cv2.rectangle(img, (0, 0), (tw + 2 * pad, tht + base + 2 * pad), (255, 255, 255), -1)
    cv2.putText(img, text, (pad, tht + pad), cv2.FONT_HERSHEY_SIMPLEX, s, (20, 20, 20), th, cv2.LINE_AA)
    return img


def _header(width: int, cells: list[str], cell_w: int, first: int = 0) -> np.ndarray:
    """A strip of column names, the first column ``first`` pixels wide (the row labels)."""
    head = np.full((30, width, 3), 255, np.uint8)
    for k, t in enumerate(cells):
        cv2.putText(head, t, (first + k * cell_w + 6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)
    return head


def _row_label(h: int, text: str, w: int = 56) -> np.ndarray:
    cell = np.full((h, w, 3), 255, np.uint8)
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
    cv2.putText(cell, text, ((w - tw) // 2, (h + tht) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3, cv2.LINE_AA)
    return cell


def _read(path: Path, cell: tuple[int, int]) -> np.ndarray:
    im = cv2.imread(str(path)) if path.exists() else None
    return cv2.resize(im, cell, interpolation=cv2.INTER_AREA) if im is not None else _blank(*cell)


def _test_frames(rec_dir: Path, letters: dict[str, str], n: int) -> list[int]:
    """``n`` held-out frames rendered by every candidate of the set, spread over the recording."""
    common = None
    for cid in letters.values():
        fr = {int(p.stem) for p in (rec_dir / cid / "renders" / "heldout").glob("*.jpg")}
        common = fr if common is None else common & fr
    frames = sorted(common or [])
    if len(frames) <= n:
        return frames
    return [frames[int((k + 0.5) * len(frames) / n)] for k in range(n)]


def rerender_test_frames(rec_dir: Path, letters: dict[str, str], frames: list[int], bg: float = 0.5):
    """The test frames rendered again by every letter's candidate the way they are scored (shifted onto
    the real frame within 6 px, one RGB gain fitted on the device) but over one fixed grey for all. The
    saved held-out pictures carry that gain over their background as well, so their grey differs from
    candidate to candidate, which tells them apart. Holes stay visible as the grey. Returns
    ({frame: real BGR}, {(letter, frame): render BGR}), or None without the recording's base run."""
    import gc

    from . import registry
    from .context import CaptureContext
    from .metrics import align_shift, fit_gain

    idx = json.loads((rec_dir / "index.json").read_text())
    if not idx.get("base_run") or not Path(idx["base_run"]).exists():
        return None
    ctx = CaptureContext.from_run(idx["base_run"], out_root=rec_dir.parent)
    to_bgr = lambda x: cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    real = {fr: cv2.cvtColor(ctx.image(fr), cv2.COLOR_RGB2BGR) for fr in frames}
    ren = {}
    for L, cid in letters.items():
        cand = registry.get(cid)(ctx, rec_dir / cid)
        cand.load()
        for fr in frames:
            view = ctx.view(fr)
            im8 = ctx.image(fr)
            im = im8.astype(np.float32) / 255.0
            rgb, alpha = cand.render(view)
            alpha = np.clip(alpha, 0, 1).astype(np.float32)
            m = ctx.silhouette(view, -3.0) & ~ctx.saturated(im8)
            comp = rgb * alpha[..., None] + (1.0 - alpha[..., None]) * bg
            comp, (dx, dy) = align_shift(comp, im, m, 6.0)
            Wt = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
            a = cv2.warpAffine(alpha, Wt, (alpha.shape[1], alpha.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)[..., None]
            g = fit_gain(comp, im, m) if m.sum() >= 500 else np.ones(3, np.float32)
            # the gain on the candidate's own colour only: where it drew nothing, the common grey
            dev = np.where(a > 1e-3, (comp - (1.0 - a) * bg) / np.maximum(a, 1e-3), bg)
            ren[(L, fr)] = to_bgr(np.clip(dev * g, 0, 1) * a + (1.0 - a) * bg)
        del cand
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    return real, ren


def test_frames_sheet(rec_dir: Path, letters: dict[str, str], out: Path, n: int = 4, cell: tuple[int, int] = (480, 360),
                      rerender: bool = True) -> Path:
    W, H = cell
    rows = []
    frames = _test_frames(rec_dir, letters, n)
    fresh = rerender_test_frames(rec_dir, letters, frames) if rerender and frames else None
    for fr in frames:
        tiles = []
        if fresh is not None:
            tiles.append(_label(cv2.resize(fresh[0][fr], cell, interpolation=cv2.INTER_AREA), f"real {fr}"))
            tiles += [_label(cv2.resize(fresh[1][(L, fr)], cell, interpolation=cv2.INTER_AREA), L) for L in letters]
            rows.append(np.concatenate(tiles, 1))
            continue
        for k, (L, cid) in enumerate(letters.items()):
            pair = cv2.imread(str(rec_dir / cid / "renders" / "heldout" / f"{fr:05d}.jpg"))
            if pair is None:
                tiles.append(_label(_blank(W, H), L))
                continue
            half = pair.shape[1] // 2
            if k == 0:
                tiles.append(_label(cv2.resize(pair[:, :half], cell, interpolation=cv2.INTER_AREA), f"real {fr}"))
            tiles.append(_label(cv2.resize(pair[:, half:], cell, interpolation=cv2.INTER_AREA), L))
        rows.append(np.concatenate(tiles, 1))
    sheet = np.concatenate(rows, 0) if rows else _blank(W, H)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


def views_sheet(rec_dir: Path, letters: dict[str, str], views: list[tuple[str, str]], out: Path, cell: tuple[int, int],
                ring: set[int] | None = None) -> Path:
    """A row per letter, a column per stress view (file stem, column name); the columns in ``ring`` get a thin
    ring at the centre, where the view looks (a defect)."""
    W, H = cell
    rows = []
    for L, cid in letters.items():
        tiles = [_read(rec_dir / cid / "renders" / "stress" / f"{stem}.jpg", cell) for stem, _ in views]
        for k in ring or ():
            cv2.circle(tiles[k], (W // 2, H // 2), int(0.08 * W), (0, 215, 255), 2, cv2.LINE_AA)
        rows.append(np.concatenate([_row_label(H, L)] + tiles, 1))
    body = np.concatenate(rows, 0)
    sheet = np.concatenate([_header(body.shape[1], [t for _, t in views], W, first=56), body], 0)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


def defect_views(rec_dir: Path, letters: dict[str, str], max_defects: int = 7) -> list[tuple[str, str]]:
    """The middle of the filmed face and each defect on it, as rendered by ``flawmap compare`` (70 mm)."""
    first = rec_dir / next(iter(letters.values())) / "renders" / "stress"
    views = [("zoom_el75_az90", "middle of the face")] if (first / "zoom_el75_az90.jpg").exists() else []
    ids = sorted(int(p.stem.rsplit("_d", 1)[1]) for p in first.glob("zoom_el75_az90_d*.jpg"))
    return views + [(f"zoom_el75_az90_d{i}", f"defect {i}") for i in ids[:max_defects]]


def turntable_grid(rec_dir: Path, letters: dict[str, str], out: Path, cols: int = 3, fps: int = 30) -> Path | None:
    """Every letter's turntable in one video, a tile per letter, all turning together."""
    import imageio.v2 as imageio

    readers = []
    for L, cid in letters.items():
        p = rec_dir / cid / "turntable.mp4"
        readers.append((L, imageio.get_reader(str(p)) if p.exists() else None))
    try:
        first = next((r for _, r in readers if r is not None), None)
        if first is None:
            return None
        h, w = first.get_data(0).shape[:2]
        n = len(readers)
        cols = min(cols, n)
        rows = math.ceil(n / cols)
        iters = [(L, iter(r) if r is not None else None) for L, r in readers]
        writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=16)
        try:
            while True:
                tiles, alive = [], False
                for L, it in iters:
                    fr = next(it, None) if it is not None else None
                    alive |= fr is not None
                    tile = np.ascontiguousarray(fr[..., :3]) if fr is not None else np.full((h, w, 3), GREY, np.uint8)
                    tiles.append(_label(tile.copy(), L, scale=1.1))
                if not alive:
                    break
                tiles += [np.full((h, w, 3), 255, np.uint8)] * (rows * cols - n)
                grid = np.concatenate([np.concatenate(tiles[r * cols:(r + 1) * cols], 1) for r in range(rows)], 0)
                writer.append_data(grid)
        finally:
            writer.close()
    finally:
        for _, r in readers:
            if r is not None:
                r.close()
    return out


def make_blind_pack(rec_dir: str | Path, cands: list[str], pack_dir: str | Path, seed: int | None = None) -> dict[str, str]:
    """Write the blind files of one recording into ``pack_dir/<rec>/`` and its letters into
    ``pack_dir.key.json`` (merged with the other recordings of the pack). Returns the letters."""
    rec_dir, pack_dir = Path(rec_dir), Path(pack_dir)
    idx = json.loads((rec_dir / "index.json").read_text())
    have = {c["id"] for c in idx.get("candidates", [])}
    missing = [c for c in cands if c not in have]
    if missing:
        raise ValueError(f"{rec_dir.name}: no {', '.join(missing)} (have {', '.join(sorted(have))})")
    seed = secrets.randbits(32) if seed is None else seed
    letters = assign(cands, random.Random(seed))
    out = pack_dir / rec_dir.name
    if out.exists():
        shutil.rmtree(out)
    (out / "each").mkdir(parents=True)
    turntable_grid(rec_dir, letters, out / "1_turntable.mp4")
    test_frames_sheet(rec_dir, letters, out / "2_test_frames.jpg")
    views_sheet(rec_dir, letters, TURN_VIEWS, out / "3_turn_views.jpg", (320, 240))
    dviews = defect_views(rec_dir, letters)
    if dviews:
        views_sheet(rec_dir, letters, dviews, out / "4_defects.jpg", (480, 360),
                    ring={k for k, (stem, _) in enumerate(dviews) if "_d" in stem})
    for L, cid in letters.items():
        src = rec_dir / cid / "turntable.mp4"
        if src.exists():
            shutil.copyfile(src, out / "each" / f"{L}.mp4")
    key_path = pack_dir.with_name(pack_dir.name + ".key.json")
    key = json.loads(key_path.read_text()) if key_path.exists() else {"pack": pack_dir.name, "recs": {}}
    key["recs"][rec_dir.name] = {"letters": letters, "seed": seed, "set": list(cands), "made": time.strftime("%Y-%m-%d %H:%M:%S")}
    key_path.write_text(json.dumps(key, indent=1))
    (pack_dir / "README.txt").write_text(README.format(pack=pack_dir.name, key=key_path.name, recs=", ".join(sorted(key["recs"]))),
                                         encoding="utf-8")
    return letters
