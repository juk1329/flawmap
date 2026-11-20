"""The positions of a uCO3D sequence's own splat fit (``io.uco3d.read_splat_means``): 16 bits a
coordinate in two PNGs, stored as sign(x) log(1 + |x|) over meta.json's range."""
import json
from types import SimpleNamespace

import cv2
import numpy as np

from flawmap.io.uco3d import read_splat_means


def _write(root, pts):
    d = root / "gaussian_splats"
    d.mkdir(parents=True)
    m = np.sign(pts) * np.log1p(np.abs(pts))
    mins, maxs = m.min(0) - 0.01, m.max(0) + 0.01
    q = np.round((m - mins) / (maxs - mins) * 65535).astype(np.uint32)
    side = int(np.ceil(np.sqrt(len(pts))))
    pad = np.zeros((side * side - len(pts), 3), np.uint32)
    q = np.concatenate([q, pad]).reshape(side, side, 3)
    # the files are RGB = xyz; OpenCV writes BGR
    cv2.imwrite(str(d / "means_u.png"), (q >> 8).astype(np.uint8)[..., ::-1])
    cv2.imwrite(str(d / "means_l.png"), (q & 255).astype(np.uint8)[..., ::-1])
    (d / "meta.json").write_text(json.dumps({"means": {"shape": [len(pts), 3], "mins": mins.tolist(), "maxs": maxs.tolist()}}))


def test_splat_means_decode_the_log_positions(tmp_path):
    rng = np.random.default_rng(0)
    pts = np.concatenate([rng.normal(0, 0.05, (40, 3)), rng.normal(0, 3.0, (9, 3))])    # an object and a far background
    _write(tmp_path, pts)
    got = read_splat_means(SimpleNamespace(root=tmp_path))
    assert got.shape == pts.shape
    # one 16-bit step of the log range: about 2e-5 near the object, relative far away
    np.testing.assert_allclose(got, pts, rtol=1e-3, atol=1e-4)
