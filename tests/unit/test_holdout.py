"""``eval.heldout``: time-block split with guard bands; held-out frames never become keyframes."""
import numpy as np

from flawmap.eval.heldout import Split, block_split
from flawmap.frames.select import select_keyframes


def test_blocks_guards_and_roles():
    t = np.arange(0, 20.0, 1 / 60)                     # 20 s at 60 fps
    sp = block_split(t)
    sets = [set(sp.train), set(sp.val), set(sp.test), set(sp.guard)]
    assert sum(len(s) for s in sets) == len(t)          # every frame has one role
    for a in range(4):
        for b in range(a + 1, 4):
            assert not sets[a] & sets[b]
    # test blocks are whole seconds 2, 8, 14 (period 6), validation 5, 11, 17; the first/last second stays in training
    assert set(np.floor(t[sp.test]).astype(int)) == {2, 8, 14}
    assert set(np.floor(t[sp.val]).astype(int)) == {5, 11, 17} - {19}
    # no training frame within 0.25 s of a held-out frame
    held = t[sorted(set(sp.test) | set(sp.val))]
    dist = np.min(np.abs(t[sp.train][:, None] - held[None, :]), axis=1)
    assert dist.min() > 0.25
    # round trip
    sp2 = Split.from_json(sp.to_json(), len(t))
    assert sp2.excluded == sp.excluded and sp2.train == sp.train


def test_keyframes_skip_held_out_frames():
    t = np.arange(0, 10.0, 1 / 60)
    sp = block_split(t)
    quality = {i: {"sharpness": float((i * 7919) % 101), "clipped": 0.0} for i in range(len(t))}
    keys = select_keyframes({i: q for i, q in quality.items() if i not in sp.excluded}, t, target=60)
    assert keys and not set(keys) & sp.excluded
