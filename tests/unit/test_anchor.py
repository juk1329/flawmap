import numpy as np
import pytest

from flawmap.anchor.parametric import CHARTS, PARTS, DeviceSpec, build_device_model, rest_pose
from flawmap.geometry.camera import Camera
from flawmap.geometry.transforms import look_at_opencv, rot_y, rotation_angle_deg
from flawmap.surface.raycast import RaycastScene


@pytest.fixture(scope="module")
def iphone12():
    return build_device_model(DeviceSpec.by_slug("iphone-12"))


@pytest.mark.parametrize("slug", ["iphone-12", "iphone-11"])
def test_body_matches_drawing_dimensions(slug):
    spec = DeviceSpec.by_slug(slug)
    m = build_device_model(spec)
    body = np.isin(m.mesh.face_label, [PARTS.index(p) for p in ("screen", "back", "frame")])
    v = m.mesh.vertices[np.unique(m.mesh.faces[body])]
    ext = v.max(0) - v.min(0)
    assert abs(ext[0] - spec.width) < 0.02
    assert abs(ext[1] - spec.length) < 0.02
    assert abs(ext[2] - spec.thickness) < 0.02
    # the camera lens is the lowest point of the back
    assert m.mesh.vertices[:, 2].min() < m.z_back - 1.0


def test_outline_starts_at_top_centre_and_goes_clockwise(iphone12):
    xy, s = iphone12.outline_xy, iphone12.outline_s
    assert np.allclose(xy[0], [0.0, iphone12.spec.length / 2])
    assert xy[5, 0] > 0                          # moves towards +x along the top edge first
    assert np.all(np.diff(s) > 0)
    assert abs(iphone12.perimeter - (s[-1] + np.linalg.norm(xy[-1] - xy[0]))) < 1e-9


def test_mirror_coordinates(iphone12):
    m = iphone12
    i = len(m.outline_xy) // 7
    x, y = m.outline_xy[i]
    for axis, q in (("x", (-x, y)), ("y", (x, -y)), ("xy", (-x, -y))):
        s_m = m.mirror_s(m.outline_s[i], axis)
        s_q = m.frame_s_of_xy(np.array([q]))[0]
        d = abs(s_m - s_q)
        assert min(d, m.perimeter - d) < 1.0


def test_rest_pose_tilts_on_the_camera(iphone12):
    T = rest_pose(iphone12, "screen")
    tilt = rotation_angle_deg(T[:3, :3])
    assert 0.4 < tilt < 1.2
    v = iphone12.mesh.vertices @ T[:3, :3].T + T[:3, 3]
    assert abs(v[:, 2].min()) < 1e-6                  # touches the board
    Tb = rest_pose(iphone12, "back")
    assert rotation_angle_deg(Tb[:3, :3] @ rot_y(180.0).T) < 1e-6   # back up lies flat on the screen


def test_raycast_hits_screen_chart_where_expected(iphone12):
    m = iphone12
    T = rest_pose(m, "back")                      # screen down, back up
    scene = RaycastScene(m.mesh, T)
    cam = Camera(np.array([[1500.0, 0, 479.5], [0, 1500.0, 359.5], [0, 0, 1]]), 960, 720)
    T_wc = look_at_opencv(np.array([0.0, 0.0, 400.0]), np.zeros(3))
    hits = scene.cast(cam, T_wc, step=4)
    assert hits.hit.mean() > 0.05
    part = m.mesh.face_label[hits.face[hits.hit]]
    assert set(np.unique(part)) <= {PARTS.index(p) for p in ("back", "camera", "lens", "frame", "button")}
    chart, uv = m.surface_coords(hits.face[hits.hit], hits.bary[hits.hit])
    back = chart == CHARTS.index("back")
    # back chart: u = W/2 - x_cad, and with the device flipped about y, x_world = -x_cad
    p = hits.point[hits.hit][back]
    assert np.allclose(uv[back][:, 0], m.spec.width / 2 - p[:, 0], atol=1e-3)
    assert np.allclose(uv[back][:, 1], m.spec.length / 2 - p[:, 1], atol=1e-3)


def test_frame_exclusions_cover_buttons(iphone12):
    ex = {e["name"]: e for e in iphone12.frame_exclusions(margin=0.0)}
    assert {"volume_up", "volume_down", "side_button", "sim_tray"} <= set(ex)
    for e in ex.values():
        span = (e["s1"] - e["s0"]) % iphone12.perimeter
        assert 3.0 < span < 60.0
