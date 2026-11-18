"""Self-referential rarity (DINOv2 patch features): a scratch on a plain face stands out against the
face itself, and nothing is scored outside the face. Skipped without the cached weights."""
import numpy as np
import pytest

from flawmap.detect import rarity


pytestmark = pytest.mark.skipif(not rarity.available(), reason="torch / transformers / cached DINOv2 weights not available")


def _face(rng, h=420, w=560):
    import cv2

    base = 168 + rng.normal(0, 3.0, (h, w)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 1.0)
    return np.repeat(base[..., None], 3, 2)


def test_scratch_is_rare_on_its_own_face():
    import cv2

    rng = np.random.default_rng(0)
    img = _face(rng)
    cv2.line(img, (300, 120), (360, 190), (95, 95, 95), 3)          # a 9 mm dark scratch (0.1 mm per px)
    img = np.clip(img, 0, 255).astype(np.uint8)
    valid = np.ones(img.shape[:2], bool)
    valid[:, :20] = False                                             # off the face
    z = rarity.face_rarity(img, valid, res_mm=0.1, device="cpu")
    assert np.isnan(z[:, :20]).all()
    on = np.zeros_like(valid)
    cv2.line(on.view(np.uint8), (300, 120), (360, 190), 1, 3)
    on = on.astype(bool)
    assert np.nanmax(z[on]) > 4.0
    assert np.nanmax(z[on]) > np.nanpercentile(z[~on & valid], 99.5)
