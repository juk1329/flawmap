"""The printed ChArUco capture board: generation (printable PDF) and camera poses from it.

World frame (``geometry.transforms``): origin at the board centre = centre of the grey area, +X to
the right of the page, +Y towards the top of the page (the printed arrow), +Z out of the paper,
millimetres. OpenCV's board frame has its origin at the top-left corner of the square grid with y
going down the page and z into the paper; ``T_world_board`` converts between the two.

Corners that touch the grey area are dropped: the grey sheet replaces whole squares, so the
chessboard pattern around those corners is incomplete and their sub-pixel positions are unreliable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..geometry.camera import Camera
from ..geometry.transforms import inv_T, make_T
from ..utils.io import read_yaml, repo_root


@dataclass
class BoardSpec:
    name: str
    page_mm: tuple[float, float]
    squares: tuple[int, int]
    square_mm: float
    marker_mm: float
    dictionary: str
    grey_cols: tuple[int, int]
    grey_rows: tuple[int, int]
    grey_rgb: tuple[int, int, int]
    px_per_mm: int

    @classmethod
    def load(cls, path: str | Path | None = None, square_mm: float | None = None) -> "BoardSpec":
        d = read_yaml(path or repo_root() / "configs" / "board_a3.yaml")
        spec = cls(d["name"], tuple(d["page_mm"]), tuple(d["squares"]), float(d["square_mm"]), float(d["marker_mm"]),
                   d["dictionary"], tuple(d["grey_cols"]), tuple(d["grey_rows"]), tuple(d["grey_rgb"]), int(d["px_per_mm"]))
        if square_mm is not None:      # measured on the printout
            k = square_mm / spec.square_mm
            spec.square_mm, spec.marker_mm = square_mm, spec.marker_mm * k
        return spec

    @property
    def size_mm(self) -> tuple[float, float]:
        return self.squares[0] * self.square_mm, self.squares[1] * self.square_mm

    def cv_board(self):
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        return cv2.aruco.CharucoBoard(self.squares, self.square_mm, self.marker_mm, d)

    @property
    def T_world_board(self) -> np.ndarray:
        """OpenCV board frame -> world frame (180 degrees about x, origin moved to the centre)."""
        w, h = self.size_mm
        return make_T(np.diag([1.0, -1.0, -1.0]), [-w / 2.0, h / 2.0, 0.0])

    def grey_rect_world(self) -> tuple[float, float, float, float]:
        """(x0, x1, y0, y1) of the grey area in world mm."""
        w, h = self.size_mm
        s = self.square_mm
        x0, x1 = self.grey_cols[0] * s - w / 2.0, (self.grey_cols[1] + 1) * s - w / 2.0
        y_top, y_bot = h / 2.0 - self.grey_rows[0] * s, h / 2.0 - (self.grey_rows[1] + 1) * s
        return x0, x1, y_bot, y_top

    def usable_corner_ids(self) -> np.ndarray:
        """ChArUco corner ids whose four neighbouring squares are all printed (not grey)."""
        b = self.cv_board()
        pts = np.asarray(b.getChessboardCorners())[:, :2]
        s = self.square_mm
        keep = []
        for cid, (x, y) in enumerate(pts):
            cols = {int(np.floor((x - s / 2) / s)), int(np.floor((x + s / 2) / s))}
            rows = {int(np.floor((y - s / 2) / s)), int(np.floor((y + s / 2) / s))}
            grey = any(self.grey_cols[0] <= c <= self.grey_cols[1] and self.grey_rows[0] <= r <= self.grey_rows[1]
                       for c in cols for r in rows)
            if not grey:
                keep.append(cid)
        return np.asarray(keep, dtype=int)

    def corner_world(self) -> np.ndarray:
        """(n_corners, 3) world positions of every ChArUco corner id."""
        pts = np.asarray(self.cv_board().getChessboardCorners(), dtype=float)
        T = self.T_world_board
        return pts @ T[:3, :3].T + T[:3, 3]


# ----------------------------------------------------------------------------- printable page
def render_board_page(spec: BoardSpec, device_outline_xy: np.ndarray | None = None, device_name: str = "",
                      font_path: str | None = None) -> "Image.Image":
    """Full A3 page as an RGB image at ``spec.px_per_mm`` with the board, grey area, device outline,
    top arrow, a 100 mm scale bar and printing instructions."""
    from PIL import Image, ImageDraw, ImageFont

    k = spec.px_per_mm
    pw, ph = int(round(spec.page_mm[0] * k)), int(round(spec.page_mm[1] * k))
    bw, bh = int(round(spec.size_mm[0] * k)), int(round(spec.size_mm[1] * k))
    board_img = spec.cv_board().generateImage((bw, bh), marginSize=0, borderBits=1)
    page = Image.new("RGB", (pw, ph), (255, 255, 255))
    ox, oy = (pw - bw) // 2, (ph - bh) // 2
    page.paste(Image.fromarray(board_img).convert("RGB"), (ox, oy))
    draw = ImageDraw.Draw(page)
    s = spec.square_mm * k
    gx0, gx1 = ox + spec.grey_cols[0] * s, ox + (spec.grey_cols[1] + 1) * s
    gy0, gy1 = oy + spec.grey_rows[0] * s, oy + (spec.grey_rows[1] + 1) * s
    draw.rectangle([gx0, gy0, gx1 - 1, gy1 - 1], fill=tuple(spec.grey_rgb))
    cx, cy = ox + bw / 2.0, oy + bh / 2.0                      # world origin on the page

    def to_px(x, y):
        return cx + x * k, cy - y * k

    def font(size_mm):
        for p in [font_path, str(Path.home() / ".fonts/NanumGothic.ttf"), "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            if p and Path(p).exists():
                return ImageFont.truetype(p, int(size_mm * k))
        return ImageFont.load_default()

    if device_outline_xy is not None:
        pts = [to_px(x, y) for x, y in np.vstack([device_outline_xy, device_outline_xy[:1]])]
        draw.line(pts, fill=(70, 70, 70), width=max(int(0.4 * k), 1))
        top_y = float(device_outline_xy[:, 1].max())
        x0g, x1g, y0g, y1g = spec.grey_rect_world()
        ay0, ay1 = top_y + 8.0, min(top_y + 26.0, y1g - 4.0)      # arrow in the grey margin above the outline
        draw.line([to_px(0, ay0), to_px(0, ay1)], fill=(40, 40, 40), width=int(1.2 * k))
        tip = [to_px(0, ay1 + 5.0), to_px(-4.0, ay1 - 1.0), to_px(4.0, ay1 - 1.0)]
        draw.polygon(tip, fill=(40, 40, 40))
        draw.text(to_px(6.0, ay1 + 2.0), "TOP 위", fill=(40, 40, 40), font=font(5.0))
        if device_name:
            draw.text(to_px(x0g + 4.0, y0g + 9.0), device_name, fill=(80, 80, 80), font=font(4.0))
    # scale bar and instructions in the bottom margin
    bar_y = oy + bh + (ph - oy - bh) / 2.0
    draw.line([(ox, bar_y), (ox + 100 * k, bar_y)], fill=(0, 0, 0), width=int(0.5 * k))
    for t in (0, 50, 100):
        draw.line([(ox + t * k, bar_y - 1.5 * k), (ox + t * k, bar_y + 1.5 * k)], fill=(0, 0, 0), width=int(0.3 * k))
    msg = (f"flawmap board {spec.name}: ChArUco {spec.squares[0]}x{spec.squares[1]}, square {spec.square_mm:.2f} mm, "
           f"marker {spec.marker_mm:.2f} mm, {spec.dictionary}.  A3 실제 크기(100%)로 인쇄하고 왼쪽 막대가 100 mm인지 자로 확인")
    draw.text((ox + 105 * k, bar_y - 2.0 * k), msg, fill=(0, 0, 0), font=font(3.0))
    return page


def save_board_pdf(page, path: str | Path, spec: BoardSpec) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dpi = spec.px_per_mm * 25.4
    page.save(path, "PDF", resolution=dpi)
    page.resize((page.width // 4, page.height // 4)).save(path.with_suffix(".preview.png"))
    return path


# ----------------------------------------------------------------------------- detection and pose
@dataclass
class BoardPose:
    T_world_cam: np.ndarray
    n_corners: int
    reproj_px: float              # RMS reprojection error of the used marker and chessboard corners
    ids: np.ndarray
    image_pts: np.ndarray
    n_markers: int = 0


class BoardDetector:
    """Board pose from ArUco marker corners, refined with the chessboard corners.

    OpenCV's ChArUco corner interpolation returned no corners at all on low (20-degree) views of the
    synthetic captures although 44-53 markers were found (2026-11-15), so the pose starts from the
    marker corners (PnP), then the usable chessboard corners are projected with that pose, refined
    with ``cornerSubPix`` and added; outliers (e.g. corners hidden by the device) are dropped by
    their reprojection error.
    """

    def __init__(self, spec: BoardSpec):
        self.spec = spec
        self.board = spec.cv_board()
        params = cv2.aruco.CharucoParameters()
        dparams = cv2.aruco.DetectorParameters()
        dparams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.CharucoDetector(self.board, params, dparams)
        self.usable = np.asarray(sorted(spec.usable_corner_ids().tolist()))
        self.corner_world = spec.corner_world()
        T = spec.T_world_board
        ids = np.asarray(self.board.getIds()).ravel()
        self.marker_world = {int(i): np.asarray(p, float).reshape(4, 3) @ T[:3, :3].T + T[:3, 3]
                             for i, p in zip(ids, self.board.getObjPoints())}

    def detect_markers(self, gray: np.ndarray):
        _, _, mcorners, mids = self.detector.detectBoard(gray)
        if mids is None or len(mids) == 0:
            return np.zeros((0, 3)), np.zeros((0, 2)), 0
        obj = np.concatenate([self.marker_world[int(m)] for m in np.ravel(mids) if int(m) in self.marker_world])
        img = np.concatenate([np.asarray(c).reshape(4, 2) for c, m in zip(mcorners, np.ravel(mids)) if int(m) in self.marker_world])
        return obj, img.astype(np.float64), len(mids)

    def pose(self, gray: np.ndarray, cam: Camera, min_markers: int = 4, min_corners: int = 0) -> BoardPose | None:
        obj_m, img_m, n_m = self.detect_markers(gray)
        if n_m < min_markers:
            return None
        K = cam.K
        dist = np.asarray(cam.dist, float) if cam.dist is not None else None
        ok, rvec, tvec = cv2.solvePnP(obj_m, img_m, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(obj_m, img_m, K, dist, rvec, tvec)
        # chessboard corners predicted by the marker pose, refined to sub-pixel
        cw = self.corner_world[self.usable]
        pc, _ = cv2.projectPoints(cw, rvec, tvec, K, dist)
        pc = pc.reshape(-1, 2)
        h, w = gray.shape[:2]
        R, _ = cv2.Rodrigues(rvec)
        zc = (cw @ R.T + tvec.ravel())[:, 2]
        inside = (pc[:, 0] > 12) & (pc[:, 0] < w - 12) & (pc[:, 1] > 12) & (pc[:, 1] < h - 12) & (zc > 0)
        ids_c = self.usable[inside]
        if inside.any():
            ref = pc[inside].astype(np.float32).reshape(-1, 1, 2).copy()
            cv2.cornerSubPix(gray, ref, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
            ref = ref.reshape(-1, 2).astype(np.float64)
            keep = np.linalg.norm(ref - pc[inside], axis=1) < 1.5
            obj_c, img_c, ids_c = cw[inside][keep], ref[keep], ids_c[keep]
        else:
            obj_c, img_c = np.zeros((0, 3)), np.zeros((0, 2))
        if len(ids_c) < min_corners:
            return None
        obj = np.concatenate([obj_m, obj_c])
        img = np.concatenate([img_m, img_c])
        is_corner = np.r_[np.zeros(len(obj_m), bool), np.ones(len(obj_c), bool)]
        for _ in range(3):
            rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
            r = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
            good = r < max(3.0 * np.median(r), 1.0)
            if good.all():
                break
            obj, img, is_corner = obj[good], img[good], is_corner[good]
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        R, _ = cv2.Rodrigues(rvec)
        T_cam_world = make_T(R, tvec.ravel())
        return BoardPose(inv_T(T_cam_world), int(is_corner.sum()), err, ids_c, img_c, n_markers=int(n_m))
