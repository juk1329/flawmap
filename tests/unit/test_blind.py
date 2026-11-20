"""Blind rating files (flawmap.recon.blind): letters, the files written, and nothing naming a candidate."""
import json
import random

import cv2
import numpy as np

from flawmap.recon.blind import assign, make_blind_pack


def test_assign_gives_every_candidate_one_letter():
    cands = ["c0", "c1", "c2", "c7", "c4b"]
    a = assign(cands, random.Random(3))
    assert list(a) == ["A", "B", "C", "D", "E"]
    assert sorted(a.values()) == sorted(cands)
    assert assign(cands, random.Random(3)) == a          # the seed fixes the order
    orders = {tuple(assign(cands, random.Random(s)).values()) for s in range(20)}
    assert len(orders) > 1                               # and different seeds shuffle


def _fake_recording(root, cands):
    rec = root / "take9"
    (rec).mkdir()
    (rec / "index.json").write_text(json.dumps({"rec": "take9", "candidates": [{"id": c} for c in cands]}))
    import imageio.v2 as imageio

    for k, c in enumerate(cands):
        d = rec / c / "renders"
        (d / "heldout").mkdir(parents=True)
        (d / "stress").mkdir(parents=True)
        pair = np.zeros((48, 128, 3), np.uint8)
        pair[:, :64] = 200                               # the real frame
        pair[:, 64:] = 40 * (k + 1)                      # this candidate's render
        for fr in (10, 20, 30):
            cv2.imwrite(str(d / "heldout" / f"{fr:05d}.jpg"), pair)
        for stem in ("ring60_el60_az0", "zoom_el75_az90", "zoom_el75_az90_d3"):
            cv2.imwrite(str(d / "stress" / f"{stem}.jpg"), np.full((48, 64, 3), 40 * (k + 1), np.uint8))
        w = imageio.get_writer(str(rec / c / "turntable.mp4"), fps=10, codec="libx264", pixelformat="yuv420p", macro_block_size=16)
        for _ in range(4):
            w.append_data(np.full((48, 64, 3), 40 * (k + 1), np.uint8))
        w.close()
    return rec


def test_blind_pack_writes_the_files_and_keeps_the_letters_apart(tmp_path):
    cands = ["c0", "c1", "c7"]
    rec = _fake_recording(tmp_path, cands)
    pack = tmp_path / "blind_t"
    letters = make_blind_pack(rec, cands, pack, seed=5)
    out = pack / "take9"
    for f in ("1_turntable.mp4", "2_test_frames.jpg", "3_turn_views.jpg", "4_defects.jpg", "each/A.mp4", "each/C.mp4"):
        assert (out / f).exists(), f
    key = json.loads((tmp_path / "blind_t.key.json").read_text())
    assert key["recs"]["take9"]["letters"] == letters and sorted(letters.values()) == cands
    # the package itself names no candidate
    readme = (pack / "README.txt").read_text(encoding="utf-8")
    assert not any(c in readme for c in cands)
    assert not any(c in p.name for p in out.rglob("*") for c in cands)
    # the test-frame sheet: a row per frame, the real frame then one tile per letter
    sheet = cv2.imread(str(out / "2_test_frames.jpg"))
    assert sheet.shape[0] == 3 * 360 and sheet.shape[1] == (1 + len(cands)) * 480
