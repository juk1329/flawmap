"""``texture.decompose``: direction bins, the lowest well-supported lightness, shading flattening."""
import numpy as np

from flawmap.texture.decompose import N_BINS, bin_of, flatten_shading, fuse, local_frames


def test_bins_cover_directions_once():
    n = np.tile([0.0, 0.0, 1.0], (6, 1))
    t1, t2 = local_frames(n)
    d = np.array([[0, 0, 1.0], [1, 0, 1.0], [0, 1, 1.0], [-1, 0, 1.0], [1, 0, 0.2], [0, -1, 0.3]])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    b = bin_of(d, n, t1, t2)
    assert b.min() >= 0 and b.max() < N_BINS
    assert len(set(b.tolist())) == len(b) - 0 or b[0] != b[4]        # straight above and grazing differ


def _samples(values):
    """(B, N, 3) uint8 grey samples from a (B, N) array of grey levels."""
    v = np.asarray(values, np.float32)
    return np.repeat(np.clip(v, 0, 255)[..., None], 3, -1).astype(np.uint8)


def test_surface_colour_ignores_reflections_and_a_misaligned_dark_view():
    B = 10
    # texel 0: surface 120, three bright reflections; texel 1: surface 120, one misaligned view of dark print (30)
    vals = np.full((B, 2), 120.0)
    vals[[1, 4, 7], 0] = [210, 230, 200]
    vals[3, 1] = 30
    have = np.ones((B, 2), bool)
    res = np.full((B, 2), 0.3, np.float32)
    out = fuse(_samples(vals), res, have)
    assert abs(int(out["diffuse"][0, 0]) - 120) <= 3          # reflections left out
    assert abs(int(out["diffuse"][1, 0]) - 120) <= 3          # the dark ghost has no support
    assert np.nanmax(out["excess"][:, 0].astype(float)) > 25  # the reflections are the excess


def test_black_glass_takes_the_dark_end():
    B = 8
    vals = np.array([[18], [20], [19], [90], [120], [85], [17], [100]], float)   # half the views reflect a lamp
    out = fuse(_samples(vals), np.full((B, 1), 0.3, np.float32), np.ones((B, 1), bool))
    assert int(out["diffuse"][0, 0]) < 25


def test_flatten_shading_removes_a_soft_shadow_keeps_print():
    h, w = 400, 300
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shade = 1.0 - 0.35 * np.exp(-((xx - 230) ** 2 + (yy - 300) ** 2) / (2 * 200.0 ** 2))   # a broad shadow (sigma 30 mm)
    img = np.full((h, w, 3), 170.0, np.float32) * shade[..., None]
    img[150:160, 100:200] = 60.0                                # a print stroke
    valid = np.ones((h, w), bool)
    out, gain = flatten_shading(img.astype(np.uint8), valid, 0.15)
    L = out.mean(-1)
    L0 = img.mean(-1)
    before = abs(L0[300, 230] - L0[50, 50])
    assert abs(L[300, 230] - L[50, 50]) < 0.3 * before            # most of the shadow is gone
    assert L[155, 150] < L[50, 50] - 40                        # the print stays dark
