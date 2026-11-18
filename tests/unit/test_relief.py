"""``shape.relief``: normal map of a height field, and nothing invented without views."""
import numpy as np

from flawmap.anchor.parametric import DeviceSpec, build_device_model
from flawmap.shape.relief import normal_map, relief_of


def test_normal_map_of_a_step():
    h = np.zeros((60, 60), np.float32)
    h[:, 30:] = 1.0
    n = normal_map(h, 0.3).astype(float) / 255.0 * 2 - 1
    assert abs(n[30, 10, 2] - 1.0) < 0.02                      # flat: straight up
    assert n[30, 29, 0] < -0.3                                 # at the step the normal leans


def test_no_views_no_relief():
    model = build_device_model(DeviceSpec.by_slug("iphone-12"))
    out = relief_of(model, "back", {}, {}, {}, masks=None)
    assert not out["info"]["used"] and float(np.abs(out["height"]).max()) == 0.0
