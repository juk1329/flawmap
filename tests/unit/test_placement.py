"""anchor.placement: telling a device lying flat from one on a stand, on synthetic LiDAR."""
import numpy as np

from flawmap.anchor.placement import _ransac_plane, detect_placement
from flawmap.geometry.transforms import make_T


class _Cap:
    """Depth maps (mm, z along the view axis) of a scene made of planes, 256 x 192 like Stray's LiDAR."""

    def __init__(self, planes, cams, w=256, h=192, f=210.0):
        self.width, self.height = w, h
        self.K = {i: np.array([[f, 0, w / 2 - 0.5], [0, f, h / 2 - 0.5], [0, 0, 1.0]]) for i in range(len(cams))}
        self._d = {}
        vs, us = np.mgrid[0:h, 0:w]
        rays = np.stack([(us - (w / 2 - 0.5)) / f, (vs - (h / 2 - 0.5)) / f, np.ones_like(us, float)], -1)
        for i, T in enumerate(cams):
            R, c = T[:3, :3], T[:3, 3]
            rw = rays @ R.T                                     # world directions per pixel, z component 1 in camera
            best = np.full((h, w), np.inf)
            for n, d, keep in planes:
                den = rw @ n
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = -(c @ n + d) / den
                X = c + t[..., None] * rw
                ok = (t > 0) & keep(X)
                best = np.where(ok & (t < best), t, best)
            self._d[i] = np.where(np.isfinite(best), best, 0).astype(np.float32)

    def depth(self, i):
        return self._d[i]

    def confidence(self, i):
        return np.full(self._d[i].shape, 2, np.uint8)


def _look_at(eye, target, up=(0, 0, 1.0)):
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:                                # looking straight down
        x = np.cross(z, (0, 1.0, 0))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return make_T(np.stack([x, y, z], 1), np.asarray(eye, float))


def test_ransac_plane_recovers_a_tilted_plane():
    rng = np.random.default_rng(1)
    n = np.array([0.0, -np.sin(np.radians(64)), np.cos(np.radians(64))])
    a = np.cross(n, [1.0, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    X = rng.uniform(-100, 100, (3000, 1)) * a + rng.uniform(-80, 80, (3000, 1)) * b + 0.5 * rng.standard_normal((3000, 1)) * n
    X = np.concatenate([X, rng.uniform(-300, 300, (600, 3))])          # clutter
    nn, d, inl = _ransac_plane(X)
    assert abs(abs(nn @ n) - 1) < 1e-3 and inl[:3000].mean() > 0.95


def _scene(tilt_deg):
    floor = (np.array([0, 0, 1.0]), 0.0, lambda X: np.ones(X.shape[:-1], bool))
    t = np.radians(tilt_deg)
    centre = np.array([0.0, 0.0, 120.0 if tilt_deg > 5 else 3.0])
    n = np.array([0.0, -np.sin(t), np.cos(t)])
    u = np.array([1.0, 0, 0])
    v = np.cross(n, u)

    def on_device(X):
        r = X - centre
        return (np.abs(r @ u) < 110) & (np.abs(r @ v) < 140)
    dev = (n, -float(n @ centre), on_device)
    cams = []
    for k in range(10):
        a = np.radians(-35 + 7 * k)
        eye = centre + 420 * (np.cos(a) * n + np.sin(a) * u)
        cams.append(_look_at(eye, centre))
    return _Cap([floor, dev], cams), cams


def test_detect_placement_flat_and_on_a_stand():
    cap, cams = _scene(0.0)
    flat = detect_placement(cap, list(range(len(cams))), dict(enumerate(cams)))
    assert not flat["tilted"]
    cap, cams = _scene(64.0)
    stand = detect_placement(cap, list(range(len(cams))), dict(enumerate(cams)))
    assert stand["tilted"] and abs(stand["tilt_deg"] - 64.0) < 2.0 and stand["height_mm"] > 50
