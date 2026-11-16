import numpy as np

from flawmap.geometry.camera import Camera, project_points
from flawmap.geometry.mesh_io import LabeledMesh
from flawmap.geometry.transforms import inv_T, look_at_opencv, make_T, pose_difference, rot_z, se3_exp


def test_look_at_is_orthonormal_and_points_at_target():
    T = look_at_opencv(np.array([100.0, -50.0, 200.0]), np.zeros(3))
    R = T[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)
    fwd = R[:, 2]
    assert np.allclose(fwd, -T[:3, 3] / np.linalg.norm(T[:3, 3]))


def test_inverse_and_pose_difference():
    T = make_T(rot_z(30.0), [1.0, 2.0, 3.0])
    assert np.allclose(inv_T(T) @ T, np.eye(4))
    dr, dt = pose_difference(T, T @ se3_exp(np.array([0, 0, np.radians(2.0), 0.5, 0, 0])))
    assert abs(dr - 2.0) < 1e-6 and dt > 0


def test_camera_project_unproject_roundtrip():
    cam = Camera(np.array([[1400.0, 0, 959.5], [0, 1410.0, 719.5], [0, 0, 1]]), 1920, 1440)
    pts = np.array([[10.0, -5.0, 300.0], [-40.0, 30.0, 250.0]])
    u, v, z = cam.project(pts)
    back = cam.unproject(u, v, z)
    assert np.allclose(back, pts)


def test_scaled_camera_keeps_pixel_centres_consistent():
    cam = Camera(np.array([[1400.0, 0, 959.5], [0, 1400.0, 719.5], [0, 0, 1]]), 1920, 1440)
    q = cam.scaled(0.25)
    assert (q.width, q.height) == (480, 360)
    p = np.array([[12.0, 7.0, 300.0]])
    u, v, _ = cam.project(p)
    uq, vq, _ = q.project(p)
    assert np.allclose(uq, (u + 0.5) * 0.25 - 0.5) and np.allclose(vq, (v + 0.5) * 0.25 - 0.5)


def test_project_points_with_pose():
    cam = Camera(np.array([[1000.0, 0, 500], [0, 1000.0, 400], [0, 0, 1]]), 1000, 800)
    T_wc = look_at_opencv(np.array([0.0, 0.0, 300.0]), np.zeros(3))
    u, v, z = project_points(cam, T_wc, np.zeros((1, 3)))
    assert np.allclose([u[0], v[0], z[0]], [500, 400, 300])


def test_labeled_mesh_roundtrip(tmp_path):
    m = LabeledMesh(np.random.rand(4, 3), np.array([[0, 1, 2], [0, 2, 3]]), np.array([0, 1]), ["a", "b"], {"k": 1})
    m.save(tmp_path / "m.npz")
    m2 = LabeledMesh.load(tmp_path / "m.npz")
    assert np.allclose(m.vertices, m2.vertices) and m2.label_names == ["a", "b"] and m2.meta == {"k": 1}
    assert list(m2.faces_of("b")) == [1]
