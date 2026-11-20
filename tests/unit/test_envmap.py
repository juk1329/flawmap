"""``recon.envmap`` (the measured environment map: support, pull-push fill, room lookup) and the
other-face rule of the pipeline (``pipeline.other_face_seen``). No capture needed."""
import math

import numpy as np

from flawmap.recon.envmap import confidence, face_points, proxy_dirs, pull_push, support, texel_of


def _dirs(H, W):
    v = (np.arange(H) + 0.5) / H * math.pi
    u = ((np.arange(W) + 0.5) / W - 0.5) * 2 * math.pi
    U, V = np.meshgrid(u, v)
    return np.stack([np.sin(V) * np.cos(U), np.sin(V) * np.sin(U), np.cos(V)], -1)


def test_texel_of_inverts_the_equirectangular_layout():
    H, W = 32, 64
    d = _dirs(H, W)
    r, c = texel_of(d, H, W)
    assert (r == np.arange(H)[:, None]).all() and (c == np.arange(W)[None, :]).all()


def test_support_counts_frames_not_pixels():
    H, W = 32, 64
    w = np.tile(np.array([[0.0, 0.0, 1.0]]), (500, 1))           # all at the pole texel
    fid = np.r_[np.zeros(400, int), np.ones(100, int)]           # two frames, many pixels each
    n = support(w, fid, H, W, min_px=20)
    assert n.max() == 2 and n.sum() == 2
    n1 = support(w[:390], fid[:390], H, W, min_px=20)            # one frame only
    assert n1.max() == 1


def test_pull_push_fills_unmeasured_directions_smoothly():
    H, W = 32, 64
    d = _dirs(H, W)
    truth = (0.2 + 0.6 * np.clip(d[..., 2], 0, 1))[..., None] * np.array([1.0, 0.9, 0.8])
    conf = (d[..., 2] > 0.3).astype(np.float32)                   # only a cap measured (as a device filmed from above)
    L = np.where(conf[..., None] > 0, truth, 0.0)                  # the rest left dark, as the fit leaves it
    F = pull_push(L, conf)
    assert np.allclose(F[conf > 0], truth[conf > 0], atol=1e-5)    # measured texels kept
    meas = truth[conf > 0]
    assert F[conf == 0].min() >= meas.min() - 1e-6                 # no dark band where nothing was measured
    assert F[conf == 0].max() <= meas.max() + 1e-6
    # no jump at the rim of the measured cap (the round-3 seam): neighbouring rows change little
    jump = np.abs(np.diff(F[..., 0], axis=0)).max()
    assert jump < 0.15
    # wraps in azimuth: the first and last columns agree
    assert np.abs(F[:, 0] - F[:, -1]).max() < 0.05


def test_confidence_ramps_with_frames():
    c = confidence(np.array([0, 1, 5, 9]))
    assert c[0] == 0 and 0 < c[1] < 1 and c[2] == 1 and c[3] == 1


def test_proxy_dirs_tend_to_the_mirror_direction_for_a_far_room():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(50, 3))
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    p = rng.uniform(-140, 140, size=(50, 3)) * np.array([1, 1, 0.02])
    assert np.allclose(proxy_dirs(p, w, None), w)
    far = proxy_dirs(p, w, 1e7)
    assert np.abs(far - w).max() < 1e-4
    near = proxy_dirs(p, w, 500.0)
    assert np.allclose(np.linalg.norm(near, axis=1), 1.0)
    # the point on the sphere: p + t w with |p + t w| = R
    t = -(p * w).sum(1) + np.sqrt((p * w).sum(1) ** 2 - (p * p).sum(1) + 500.0 ** 2)
    q = p + t[:, None] * w
    assert np.allclose(np.linalg.norm(q, axis=1), 500.0) and np.allclose(near, q / 500.0)


def test_face_points_match_the_chart_layout():
    uv = np.array([[0.0, 0.0], [100.0, 50.0]])
    b = face_points(uv, "back", (200.0, 300.0), -3.0)
    s = face_points(uv, "screen", (200.0, 300.0), 3.0)
    assert np.allclose(b[0], [100.0, 150.0, -3.0]) and np.allclose(b[1], [0.0, 100.0, -3.0])
    assert np.allclose(s[0], [-100.0, 150.0, 3.0]) and np.allclose(s[1], [0.0, 100.0, 3.0])


def test_other_face_is_used_only_when_the_recording_films_it():
    from flawmap.anchor.parametric import build_device_model
    from flawmap.anchor.template import template_spec
    from flawmap.geometry.camera import Camera
    from flawmap.pipeline import other_face_seen
    from flawmap.recon.stress import look_at

    model = build_device_model(template_spec(150.0, 72.0, 8.0, 10.0))
    K = np.array([[700.0, 0, 480], [0, 700.0, 360], [0, 0, 1]])
    cam = Camera(K, 960, 720)

    def ring(z_sign, n):
        out = {}
        for k in range(n):
            a = 2 * math.pi * k / n
            eye = np.array([180 * math.cos(a), 180 * math.sin(a), z_sign * 260.0])
            out[len(out)] = look_at(eye, np.zeros(3), np.array([0, 0, z_sign * 1.0]))
        return out

    back_only = ring(-1, 40)                                     # the back is z < 0 (the back faces -z)
    cams = {i: cam for i in range(80)}
    r = other_face_seen(model, cams, back_only, "back")
    assert r["face"] == "screen" and not r["use"] and r["views"] == 0
    both = dict(back_only)
    for k, T in ring(1, 40).items():
        both[40 + k] = T
    r = other_face_seen(model, cams, both, "back")
    assert r["use"] and r["views"] == 40
    # a poorly registered other side is not used
    ious = {i: (0.95 if i < 40 else 0.6) for i in both}
    assert not other_face_seen(model, cams, both, "back", ious)["use"]


def test_pole_soften_removes_the_pinwheel_and_keeps_a_lamp():
    """Near a pole the columns of the map meet: wedges that differ there showed as a pinwheel
    straight at the face; softening on the sphere evens them out at the pole, keeps a lamp near the
    pole a spot (no ring), and leaves the map away from the poles as it was."""
    from flawmap.recon.envmap import pole_soften

    H, W = 32, 64
    L = np.zeros((H, W, 3), np.float32)
    L[:, : W // 2] = 1.0                                  # two halves meeting at both poles: wedges
    S = pole_soften(L)
    for r in (0, H - 1):                                  # the pole rows: a fifth of the contrast, no step
        assert S[r].std() < 0.25 * L[r].std()
        assert np.abs(np.diff(np.concatenate([S[r], S[r][:1]]), axis=0)).max() < 0.05
    assert np.allclose(S[12:20], L[12:20])                 # the equator band is untouched
    lamp = np.zeros((H, W, 3), np.float32)
    lamp[2, 10:13] = 5.0                                   # a lamp 14 degrees from the pole
    T = pole_soften(lamp)
    ring = T[2, 40:43].mean() / T[2, 10:13].mean()         # the far side of the same row
    assert ring < 0.05 and T[2, 10:13].mean() > 0.3
