"""``recon``: stress views, metrics, ULR helpers, splat PLY round trip (no capture needed)."""
import numpy as np

from flawmap.recon.candidates.c2_ulr import fps_directions, remap_points
from flawmap.recon.candidates.c5_free_gs import read_ply, write_ply
from flawmap.recon.context import View, plate_silhouette
from flawmap.recon.metrics import align_shift, fit_gain, psnr
from flawmap.recon.stress import look_at, stress_K


def test_look_at_points_the_optical_axis_at_the_target():
    eye, tgt = np.array([0.0, -300.0, -200.0]), np.array([10.0, 5.0, 0.0])
    T = look_at(eye, tgt, np.array([0, 0, -1.0]))
    R = T[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0)
    f = (tgt - eye) / np.linalg.norm(tgt - eye)
    assert np.allclose(R[:, 2], f)
    # the target projects to the image centre
    K = stress_K(640, 480)
    pc = R.T @ (tgt - eye)
    u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
    assert abs(u - K[0, 2]) < 1e-6 and abs(v - K[1, 2]) < 1e-6


def test_plate_silhouette_is_the_projected_hull():
    pts = np.array([[x, y, 0.0] for x in (-50, 50) for y in (-80, 80)])
    K = stress_K(640, 480)
    T = look_at(np.array([0, 0, 300.0]), np.zeros(3), np.array([0, 1.0, 0]))
    m = plate_silhouette(pts, View(K, T, 640, 480))
    assert 0.05 < m.mean() < 0.5
    assert m[240, 320] and not m[5, 5]


def test_align_shift_recovers_a_small_translation_and_gain():
    rng = np.random.default_rng(0)
    img = rng.random((120, 160, 3)).astype(np.float32)
    import cv2

    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    M = np.float32([[1, 0, 2.0], [0, 1, -1.5]])
    moved = cv2.warpAffine(img, M, (160, 120), borderMode=cv2.BORDER_REFLECT)
    m = np.zeros((120, 160), bool)
    m[20:100, 20:140] = True
    out, (dx, dy) = align_shift(moved, img, m)
    assert abs(dx - 2.0) < 0.3 and abs(dy + 1.5) < 0.3
    assert psnr(out, img, m) > psnr(moved, img, m) + 5
    g = fit_gain(img * 0.8, img, m)
    assert np.allclose(g, 1.25, atol=1e-3)


def test_fps_directions_spreads_and_remap_points_samples():
    rng = np.random.default_rng(1)
    d = rng.normal(size=(500, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    pick = fps_directions(d, 20)
    assert len(set(pick)) == 20
    sub = d[pick]
    assert (sub @ sub.T - 2 * np.eye(20)).max() < 0.95          # no two picks nearly the same
    img = np.arange(5 * 6 * 4, dtype=np.uint8).reshape(5, 6, 4)
    x = np.array([0.0, 5.0, 2.0, 100.0], np.float32)
    y = np.array([0.0, 4.0, 3.0, 1.0], np.float32)
    s = remap_points(img, x, y, width=3)
    assert (s[0] == img[0, 0]).all() and (s[1] == img[4, 5]).all() and (s[2] == img[3, 2]).all() and (s[3] == 0).all()


def test_splat_ply_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    n = 7
    d = {"means": rng.normal(size=(n, 3)).astype(np.float32), "sh0": rng.normal(size=(n, 1, 3)).astype(np.float32),
         "shN": rng.normal(size=(n, 15, 3)).astype(np.float32), "opacities": rng.normal(size=n).astype(np.float32),
         "scales": rng.normal(size=(n, 3)).astype(np.float32), "quats": rng.normal(size=(n, 4)).astype(np.float32)}
    write_ply(tmp_path / "s.ply", d)
    r = read_ply(tmp_path / "s.ply")
    for k in d:
        assert np.allclose(r[k], d[k]), k


def test_detail_correlation_rejects_uniform_renders():
    import cv2

    from flawmap.recon.metrics import hf_ncc

    rng = np.random.default_rng(3)
    real = cv2.GaussianBlur(rng.random((120, 160, 3)).astype(np.float32), (0, 0), 1.5)
    m = np.zeros((120, 160), bool)
    m[10:110, 10:150] = True
    assert hf_ncc(real, real, m) > 0.99
    flat = np.full_like(real, real[m].mean())
    assert abs(hf_ncc(flat, real, m)) < 1e-6                 # a uniform fill has no detail
    blurred = cv2.GaussianBlur(real, (0, 0), 6.0)
    assert hf_ncc(blurred, real, m) < 0.5


def test_ulr_blend_weighs_an_extra_value_like_the_colours():
    """C7 blends C1 as seen from each stored view with C2's weights (``blend_views(extra=...)``): fed
    the views' own colours as the extra value, the blend of the extra equals the blend of the colours."""
    from flawmap.recon.candidates.c2_ulr import C2ULR, cover_views, coverage_grid

    rng = np.random.default_rng(1)
    obj = C2ULR.__new__(C2ULR)
    n_views, rows, cols, res = 14, 40, 30, 1.0
    obj.views, cols_k = [], []
    for k in range(n_views):
        c = rng.uniform(40, 220, 3).astype(np.uint8)
        cols_k.append(c / 255.0)
        img = np.zeros((rows, cols, 4), np.uint8)
        img[..., :3] = c
        img[..., 3] = 255
        obj.views.append(img)
    ang = np.linspace(0, 2 * np.pi, n_views, endpoint=False)
    obj.centers = np.stack([150 * np.cos(ang), 150 * np.sin(ang), np.full(n_views, -250.0)], 1)
    obj.vdirs = obj.centers / np.linalg.norm(obj.centers, axis=1, keepdims=True)
    obj.cover = np.stack([coverage_grid(v[..., 3]) for v in obj.views])
    obj.fixed = cover_views(obj.vdirs, obj.cover)
    obj.meta = {"res_mm": res}
    obj.fade = (12.0, 25.0)
    P = np.stack([rng.uniform(-10, 10, 50), rng.uniform(-10, 10, 50), np.zeros(50)], 1)
    eye = np.array([120.0, 60.0, -260.0])
    d = eye[None, :] - P
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    uv = np.stack([rng.uniform(2, cols - 2, 50), rng.uniform(2, rows - 2, 50)], 1)
    order = {}

    def extra(k, dk):
        order[k] = True
        return np.repeat(cols_k[k][None, :], len(dk), 0).astype(np.float32)

    ulr, share, ex = obj.blend_views(P, d, uv, eye, extra)
    ulr0, share0 = obj.blend_views(P, d, uv, eye)
    assert len(order) == 12                          # the 12 bound views each gave their value
    assert np.allclose(ulr, ulr0) and np.allclose(share, share0)
    assert np.allclose(ex, ulr, atol=1e-4)           # the same weights


def _ring_views(n=36, partial=()):
    """View directions on a cone around -z and their coverage of a 12 x 9 grid: the views in
    ``partial`` see only the top third of the face (close-ups)."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c = np.stack([np.cos(ang), np.sin(ang), np.full(n, -2.0)], 1)
    vdirs = c / np.linalg.norm(c, axis=1, keepdims=True)
    cover = np.ones((n, 12, 9), bool)
    for k in partial:
        cover[k, 4:] = False
    return vdirs, cover


def test_cover_views_fill_what_the_nearest_views_miss():
    from flawmap.recon.candidates.c2_ulr import cover_views

    vdirs, cover = _ring_views(36, partial=range(0, 12))       # a third of the ring are close-ups
    fixed = cover_views(vdirs, cover, n_fixed=4, n_dir=8)
    assert 1 <= len(fixed) <= 4 and len(set(fixed)) == len(fixed)
    assert all(cover[k].all() for k in fixed)                  # whole-face views are kept for coverage
    assert cover_views(vdirs, cover, n_fixed=4, n_dir=8) == fixed   # once per recording, deterministic
    full = np.ones((36, 12, 9), bool)
    assert cover_views(vdirs, full) == []                      # nothing to fill: all slots go to direction


def test_bound_set_changes_only_by_direction_rank():
    """Round 4 chose the coverage views again for every eye direction and swapped them in steps (the
    jumps of the blind ratings, cc53da7c23 at 77-78 degrees); now they stay bound for every eye and the
    rest changes one direction rank at a time."""
    from flawmap.recon.candidates.c2_ulr import cover_views, rank_ramp, select_views

    vdirs, cover = _ring_views(36, partial=range(0, 12))
    fixed = cover_views(vdirs, cover, n_fixed=4, n_dir=8)
    prev = None
    for deg in np.arange(0, 360, 1.0):
        a = np.radians(deg)
        e = np.array([np.cos(a), np.sin(a), -2.0])
        e /= np.linalg.norm(e)
        bound = select_views(vdirs, e, fixed)
        assert len(bound) == 12 and bound[:len(fixed)] == fixed and len(set(bound)) == 12
        if prev is not None:
            assert len(set(bound) ^ set(prev)) <= 2              # at most one view in, one out per degree
        prev = bound
        w = rank_ramp(vdirs, e, bound, fixed, 0.5)
        assert np.all(w[:len(fixed)] == 1) and np.all((w >= 0) & (w <= 1))
        assert np.all(rank_ramp(vdirs, e, bound, fixed, 0.0) == 1)


def test_rank_ramp_reaches_zero_where_a_view_leaves():
    """With the ramp a view bound by direction has no weight when it leaves the set: turning in small
    steps, each leaving view's factor at the step before is near 0, and the nearest view's is 1."""
    from flawmap.recon.candidates.c2_ulr import rank_ramp, select_views

    vdirs, _ = _ring_views(36)
    prev, worst = None, 0.0
    for deg in np.arange(0, 360, 0.1):
        a = np.radians(deg)
        e = np.array([np.cos(a), np.sin(a), -2.0])
        e /= np.linalg.norm(e)
        bound = select_views(vdirs, e, [])
        w = rank_ramp(vdirs, e, bound, [], 0.5)
        assert w[0] == 1.0                                       # the nearest
        if prev is not None:
            worst = max([worst] + [pw for k, pw in zip(*prev) if k not in bound])
        prev = (bound, w)
    assert worst < 0.1


def test_turn_jumps_find_the_one_degree_spike():
    from flawmap.recon.metrics import turn_jumps

    rng = np.random.default_rng(0)
    steps = 0.015 + 0.001 * rng.standard_normal(359)
    az = np.arange(359, dtype=float)
    smooth = turn_jumps(steps, az)
    assert smooth["jump"] < 1.5 and smooth["jump_deg"] == []
    steps[77] = 0.045                                         # one 1-degree change three times the usual
    j = turn_jumps(steps, az)
    assert 2.8 < j["jump"] < 3.2 and j["jump_deg"] == [77]
    assert turn_jumps([0.01], [0.0]) == {}


def test_blend_has_no_jump_where_the_bound_set_changes():
    """Round 5: views on a ring all about as far from the eye (509e5ece1b at 139 degrees: about 14
    views within 2 degrees of each other), the eye turning so the 12 bound views change: with the
    ramps the blend changes smoothly; with a hard set (ramp 0, as in round 4) it jumps at each change."""
    from flawmap.recon.candidates.c2_ulr import C2ULR, cover_views, coverage_grid

    rng = np.random.default_rng(2)
    obj = C2ULR.__new__(C2ULR)
    n_views, rows, cols = 36, 20, 20
    obj.views = []
    for _ in range(n_views):
        img = np.zeros((rows, cols, 4), np.uint8)
        img[..., :3] = rng.uniform(30, 225, 3).astype(np.uint8)
        img[..., 3] = 255
        obj.views.append(img)
    ang = np.linspace(0, 2 * np.pi, n_views, endpoint=False)
    r = 300 * np.tan(np.radians(20))                 # a ring 20 degrees off the axis
    obj.centers = np.stack([r * np.cos(ang), r * np.sin(ang), np.full(n_views, -300.0)], 1)
    obj.vdirs = obj.centers / np.linalg.norm(obj.centers, axis=1, keepdims=True)
    obj.cover = np.stack([coverage_grid(v[..., 3]) for v in obj.views])
    obj.fixed = cover_views(obj.vdirs, obj.cover)
    assert obj.fixed == []                           # every view sees the whole face
    obj.meta = {"res_mm": 1.0}
    obj.fade = (12.0, 25.0)
    P = np.stack([rng.uniform(-8, 8, 60), rng.uniform(-8, 8, 60), np.zeros(60)], 1)
    uv = np.stack([rng.uniform(2, cols - 2, 60), rng.uniform(2, rows - 2, 60)], 1)
    re = 300 * np.tan(np.radians(6))                 # the eye 6 degrees off the axis, turning around it

    def worst_over_usual(ramp):
        obj.ramp = ramp
        outs = []
        for deg in np.arange(0, 60, 0.25):
            a = np.radians(deg)
            eye = np.array([re * np.cos(a), re * np.sin(a), -300.0])
            d = eye[None, :] - P
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            ulr, share = obj.blend_views(P, d, uv, eye)
            outs.append(ulr * share[:, None])
        steps = np.abs(np.diff(np.stack(outs), axis=0)).mean((1, 2))
        return steps.max() / np.median(steps)

    smooth, hard = worst_over_usual(0.5), worst_over_usual(0.0)
    assert smooth < 2.5
    assert hard > 2 * smooth


def test_distinct_views_keep_one_of_each_near_repeat():
    """Views nearly repeating another's direction (consecutive frames) are not bound: of each pair 0.5
    degrees apart the one that saw more of the face stays; the bound set and the ramps use those only."""
    from flawmap.recon.candidates.c2_ulr import distinct_views, rank_ramp, select_views

    az = np.repeat(np.radians(np.arange(0, 360, 12)), 2) + np.tile([0.0, np.radians(0.5) / np.sin(np.radians(30))], 30)
    c = np.stack([np.cos(az) * np.sin(np.radians(30)), np.sin(az) * np.sin(np.radians(30)), -np.full(60, np.cos(np.radians(30)))], 1)
    vdirs = c / np.linalg.norm(c, axis=1, keepdims=True)
    quality = np.tile([0.9, 1.0], 30)                     # the second of each pair saw more
    keep = distinct_views(vdirs, quality, 3.0)
    assert keep == list(range(1, 60, 2))
    assert distinct_views(vdirs, quality, 0.1) == list(range(60))
    e = vdirs[7] + vdirs[9]
    e /= np.linalg.norm(e)
    bound = select_views(vdirs, e, [], active=keep)
    assert len(bound) == 12 and set(bound) <= set(keep)
    w = rank_ramp(vdirs, e, bound, [], 0.5, active=keep)
    assert np.all((w >= 0) & (w <= 1)) and w.max() == 1.0


def test_choose_sep_keeps_the_sparser_views_unless_validation_loses():
    """Views at least 3 degrees apart turn more smoothly; they are kept unless they cost more than
    0.2 dB on the validation frames (round 5: cc53da7c23 lost 1.0 dB, the other three did not)."""
    from flawmap.recon.candidates.c2_ulr import choose_sep

    assert choose_sep({3.0: 18.10, 0.0: 19.12}) == 0.0          # cc53da7c23
    assert choose_sep({3.0: 16.2553, 0.0: 16.2575}) == 3.0      # 509e5ece1b: a tie
    assert choose_sep({3.0: 16.72, 0.0: 16.46}) == 3.0          # 29d760c369: better
    assert choose_sep({3.0: None, 0.0: 16.0}) == 0.0


def test_feather_follows_the_device_size_and_sep_without_validation():
    """The views' alpha ramp is a share of the device's width (20 mm on the 13-inch iPad, about 6.6 mm on
    a 71.5 mm phone); without validation scores the separation rule keeps the smoother set."""
    from types import SimpleNamespace

    from flawmap.recon.candidates.c2_ulr import choose_sep, feather_mm

    ipad = SimpleNamespace(spec=SimpleNamespace(width=215.0, length=280.0))
    phone = SimpleNamespace(spec=SimpleNamespace(width=71.5, length=146.7))
    assert abs(feather_mm(ipad) - 20.0) < 0.01 and abs(feather_mm(phone) - 6.65) < 0.05
    assert choose_sep({3.0: None, 0.0: None}) == 3.0


def test_model_of_a_public_video_run_is_the_scaled_fit():
    """A public video's run (pipeline.prepare_uco3d_pass) keeps the plate fit and the assumed width, not
    processing.device_estimate; the model is the fit scaled to that width."""
    from flawmap.anchor.template import model_of_run

    rep = {"device": "template", "processing": {"pose_mode": "sfm"},
           "passes": [{"template": {"size_mm": [139.49, 70.24, 7.71], "corner_radius": 12.0, "assumed_width_mm": 71.5}}]}
    m = model_of_run(rep)
    assert abs(m.spec.width - 71.5) < 1e-6 and abs(m.spec.length - 139.49 * 71.5 / 70.24) < 1e-3
