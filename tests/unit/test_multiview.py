"""``detect.multiview``: design marks are set aside, a fixed glint is found, a moving one is not."""
from types import SimpleNamespace

import cv2
import numpy as np

from flawmap.detect.multiview import design_mask, glint_maps


def test_design_mask_finds_logo_and_print_not_a_faint_blob():
    res = 0.15
    L = np.full((700, 500), 58.0, np.float32)
    cv2.circle(L, (250, 250), 90, 30.0, -1)                    # a logo (13.5 mm)
    for k in range(8):                                         # a row of glyphs 2 mm high, 0.45 mm apart
        cv2.rectangle(L, (100 + 15 * k, 520), (111 + 15 * k, 533), 45.0, -1)
    cv2.circle(L, (420, 120), 40, 53.0, -1)                    # a faint stain (5 L)
    valid = np.ones_like(L, bool)
    m, recs = design_mask(L, valid, res)
    assert m[250, 250] and m[526, 140]
    assert not m[120, 420]
    kinds = sorted(r["kind"] for r in recs)
    assert "design" in kinds                                    # the logo (and the dense glyph row, joined into one block)


def test_glint_recurring_in_place_beats_a_moving_reflection():
    rows, cols, B = 120, 160, 6
    geom = SimpleNamespace(shape=(rows, cols), flat_index=np.arange(rows * cols))
    geom.to_image = lambda v, fill=0: np.asarray(v).reshape(rows, cols) if np.asarray(v).ndim == 1 else np.asarray(v).reshape(rows, cols, -1)
    ex = np.zeros((B, rows * cols), np.float16)
    for b in range(B):
        img = np.zeros((rows, cols), np.float32)
        if b in (1, 3, 4):
            img[60, 20:60] = 30.0                              # a scratch glint, same place, three bins
        img[10 + 15 * b, 100:150] = 20.0                       # a reflected edge that moves with the bin
        ex[b] = img.ravel()
    dec = SimpleNamespace(excess=ex)
    second, count = glint_maps(SimpleNamespace(geom=geom), dec, np.ones((rows, cols), bool), dl=6.0)
    assert second[60, 40] >= 6.0 and count[60, 40] >= 2
    assert second[25, 125] < 6.0                               # the moving one never recurs
