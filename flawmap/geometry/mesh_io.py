"""Labeled triangle meshes (adapted from ISCO ``geometry/mesh_io.py``).

A ``LabeledMesh`` is a triangle mesh with one integer label per face. In flawmap the label
is the **part** of the device a face belongs to (``label_names[k]``, e.g. ``"screen"``,
``"back"``, ``"frame"``, ``"camera"``); defects are kept separately in unwrapped surface
coordinates because the reference model's triangles are much larger than a defect.

Files are ``.npz`` written with ``allow_pickle=False``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh


@dataclass
class LabeledMesh:
    vertices: np.ndarray
    faces: np.ndarray
    face_label: np.ndarray | None = None
    label_names: list[str] = field(default_factory=lambda: ["part"])
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if self.face_label is None:
            self.face_label = np.zeros(len(self.faces), dtype=np.int32)
        self.face_label = np.asarray(self.face_label, dtype=np.int32)
        assert len(self.face_label) == len(self.faces)

    @classmethod
    def from_trimesh(cls, mesh: trimesh.Trimesh, face_label=None, label_names=None, meta=None) -> "LabeledMesh":
        return cls(np.array(mesh.vertices), np.array(mesh.faces), face_label,
                   list(label_names) if label_names else ["part"], dict(meta or {}))

    def to_trimesh(self, process: bool = False) -> trimesh.Trimesh:
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=process)

    def copy(self) -> "LabeledMesh":
        return LabeledMesh(self.vertices.copy(), self.faces.copy(), self.face_label.copy(),
                           list(self.label_names), json.loads(json.dumps(self.meta)))

    def label_id(self, name: str) -> int:
        return self.label_names.index(name)

    def faces_of(self, name: str) -> np.ndarray:
        return np.flatnonzero(self.face_label == self.label_id(name))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            vertices=self.vertices.astype(np.float64),
            faces=self.faces.astype(np.int32),
            face_label=self.face_label.astype(np.int32),
            label_names=np.array(json.dumps(self.label_names)),
            meta=np.array(json.dumps(self.meta)),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "LabeledMesh":
        with np.load(path, allow_pickle=False) as z:
            return cls(z["vertices"].astype(np.float64), z["faces"].astype(np.int64), z["face_label"],
                       json.loads(str(z["label_names"])), json.loads(str(z["meta"])))


def load_mesh_any(path: str | Path) -> trimesh.Trimesh:
    """Load STL/PLY/OBJ/GLB as a single Trimesh with merged vertices."""
    mesh = trimesh.load(str(path), force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):  # pragma: no cover
        raise ValueError(f"not a triangle mesh: {path}")
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    return mesh


def bounding_radius(mesh: trimesh.Trimesh | LabeledMesh, center: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(mesh.vertices) - center, axis=1).max())
