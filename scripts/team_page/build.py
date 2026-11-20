"""The team's comparison page (a Claude artifact): every example capture with three models of it in
the device frame, turning together: the service's model before (C0), plain 3D Gaussian splatting
(C5a, what did not work) and the final method (C7 round 5).

    python scripts/team_page/build.py <out_dir>          # e.g. runs/artifact_cmp/site

The page's own files (``site/``) are copied as they are; per example (``data/<id>/``):

- ``geo.glb``, ``tex.jpg`` (+ ``rough.png``): the inspection run's fitted plate and inspection texture,
  shared by the old view (C0, the service's model before) and the final one (C7 round 5);
- ``c7_<face>/``: the realistic view of that face (manifest, ``env.f16``, ``s.f16``) with its real-frame
  views packed into one ``views.bin`` (the page reads them out by byte range);
- ``gs_<face>.splat`` + ``gs_<face>_sh.bin``: plain 3D Gaussian splatting (C5a) cut to a box around the
  device, in the 32-byte ``.splat`` layout (position, scale, RGBA from the SH base colour, rotation),
  and its view-dependent colour (SH degrees 1–2 above the base, one byte a coefficient; degree 3 is left
  out: 31–33 dB from the full colour at the test views, for 45% less data).
Binary files are stored as base64 text (``<name>.b64.txt``): the artifact host serves only web types,
and the page's fetch wrapper (``site/js/app.js``) turns them back into bytes under their own names.
``data/examples.json`` lists them with what each shows. The page's copies of the C7 renderer
(``web/recon/``) go to ``js/recon/`` with three.js imported from the CDN's module builds.
"""
import base64
import json
import re
import shutil
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flawmap.recon.candidates.c5_free_gs import read_ply  # noqa: E402

SITE = Path(__file__).resolve().parent / "site"
GS = Path("runs/artifact_cmp/gs")          # train_gs.py's outputs, one folder a face
EXAMPLES = [
    {"id": "ipad", "run": "runs/real_ipad_r5", "title": "아이패드 프로 13\" 은색, 두 면",
     "kind": "실제 촬영 (팀)", "device": "iPad Pro 13\" (실제)", "place": "나무 바닥에 눕힘, 면마다 따로 녹화",
     "gs": {"back": GS / "ipad_back", "screen": GS / "ipad_screen"},
     "look": "뒷면을 정면에서 보면 은색 알루미늄의 부드러운 반사, 화면 쪽은 검은 유리에 비친 방이 보입니다. 기존 모델은 무광 회색 판처럼 보입니다."},
    {"id": "cc53", "run": "runs/svc_check_cc53", "title": "아이패드 뒷면, 눕혀서 찍음",
     "kind": "실제 촬영 (팀)", "device": "iPad Pro 13\" (실제)", "place": "바닥에 눕힘",
     "gs": {"back": Path("runs/compare/cc53da7c23/c5a/artifact")},
     "look": "로고와 카메라 섬 주변 반사를 보세요. 4라운드에서 77°쯤 렌즈 색이 뒤집히던 자리입니다."},
    {"id": "509e", "run": "runs/svc_check_509e", "title": "아이패드 뒷면, 거치대에 세워 찍음",
     "kind": "실제 촬영 (팀)", "device": "iPad Pro 13\" (실제)", "place": "거치대에 약 64° 기울여 세움 (아래 가장자리 일부 가림)",
     "gs": {"back": GS / "509e_back"},          # the older comparison run's frame is flipped: trained again on this run
     "look": "거치대에 가려졌던 아래 가장자리와, 돌릴 때 반사 경계가 건너뛰지 않는지 보세요. 이 녹화만 3° 간격 규칙이 걸렸습니다."},
    {"id": "6c41", "run": "runs/svc_check_6c41", "title": "아이패드 뒷면, 장에 기대 세워 찍음",
     "kind": "실제 촬영 (팀)", "device": "iPad Pro 13\" (실제)", "place": "장에 약 76° 기대 세움, 앞쪽 90°만 찍음",
     "gs": {"back": Path("runs/compare/6c417d24a0/c5a/artifact")},
     "look": "찍지 않은 방향(뒤로 돌렸을 때)에서 반사가 어떻게 채워지는지 보세요."},
    {"id": "synth_dent", "run": "runs/synth_dent", "title": "아이폰 12 블랙, 모서리 찍힘 (가상)",
     "kind": "가상 촬영", "device": "iPhone 12 (가상 렌더)", "place": "가상 바닥, 1.6초",
     "gs": {"back": GS / "synth_dent_back"},
     "look": "아주 짧은 녹화(1.6초, 25프레임)입니다. 검증 구간이 없어 실물처럼 보기는 기본 설정으로 만들었습니다."},
    {"id": "synth_demo", "run": "runs/synth_demo", "title": "아이폰 12 블랙, 두 면 (가상)",
     "kind": "가상 촬영", "device": "iPhone 12 (가상 렌더)", "place": "가상 바닥, 면마다 5.5초",
     "gs": {"back": GS / "synth_demo_back", "screen": GS / "synth_demo_screen"},
     "look": "두 면을 각각 다른 녹화로 만들어 한 모델에 얹었습니다."},
    {"id": "uco3d_59015", "run": "runs/uco3d_59015-27660-21826", "title": "흰색 아이폰, 액정 금 (공개 영상)",
     "kind": "공개 영상 (uCO3D, CC BY 4.0)", "device": "아이폰 (공개 영상, LiDAR 없음)", "place": "검은 천 위, 8.4초",
     "gs": {"screen": GS / "uco3d_59015_screen"},
     "look": "LiDAR와 ARKit 없이 SfM 카메라로 만든 경우입니다. 검은 화면 유리와 금이 간 자리를 보세요. 크기는 폭 71.5 mm로 가정했습니다."},
    {"id": "uco3d_2712", "run": "runs/uco3d_2712-37354-8763", "title": "청록 테두리 휴대폰, 액정 금 (공개 영상)",
     "kind": "공개 영상 (uCO3D, CC BY 4.0)", "device": "휴대폰 (공개 영상, LiDAR 없음)", "place": "나무 판자 위, 18.5초",
     "gs": {"screen": GS / "uco3d_2712_screen"},
     "look": "나무 판자 틈 사이로 찍은 공개 영상입니다. 가우시안 스플래팅은 판자와 휴대폰이 한데 엉깁니다."},
]
C0 = 0.28209479177387814
MAX_SPLATS = 160_000
MARGIN_FRAC = 0.25        # the box kept around the device, as a share of its length
SH_DEGREE = 2             # view-dependent colour kept (the splats were trained with degree 3)


def write_b64(dst: Path, data: bytes) -> None:
    """``dst`` as base64 text, ``<dst>.b64.txt`` (the page asks for ``dst`` and decodes)."""
    Path(str(dst) + ".b64.txt").write_text(base64.b64encode(data).decode("ascii"))
THREE = "https://cdn.jsdelivr.net/npm/three@0.180.0"
MODULES = ["common.js", "plate_vd.js", "mesh_ulr.js", "mesh_hybrid.js"]


def strip_glb(src: Path, dst: Path) -> None:
    """The plate's geometry only: the GLB's placeholder image, texture and material go (both
    renderers bring their own material, and an embedded image would be decoded for nothing)."""
    b = src.read_bytes()
    jlen = struct.unpack("<I", b[12:16])[0]
    j = json.loads(b[20:20 + jlen])
    off = 20 + jlen
    blen = struct.unpack("<I", b[off:off + 4])[0]
    blob = b[off + 8:off + 8 + blen]
    for k in ("images", "textures", "samplers", "materials"):
        j.pop(k, None)
    used = []
    for m in j["meshes"]:
        for pr in m["primitives"]:
            pr.pop("material", None)
            used += list(pr["attributes"].values()) + ([pr["indices"]] if "indices" in pr else [])
    views, out = {}, bytearray()
    for a in sorted(set(used)):
        acc = j["accessors"][a]
        v = acc["bufferView"]
        if v not in views:
            bv = j["bufferViews"][v]
            while len(out) % 4:
                out += b"\0"
            views[v] = {**bv, "byteOffset": len(out), "buffer": 0}
            out += blob[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]
    order = sorted(views)
    remap = {v: n for n, v in enumerate(order)}
    j["bufferViews"] = [views[v] for v in order]
    for acc in j["accessors"]:
        acc["bufferView"] = remap.get(acc["bufferView"], acc["bufferView"])
    while len(out) % 4:
        out += b"\0"
    j["buffers"] = [{"byteLength": len(out)}]
    js = json.dumps(j, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(out), 0x004E4942) + bytes(out)
    write_b64(dst, struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)


def copy_modules(site: Path) -> None:
    """web/recon's C7 renderer for the page, importing three.js from the CDN (the page has no import map)."""
    d = site / "js" / "recon"
    d.mkdir(parents=True, exist_ok=True)
    for name in MODULES:
        s = Path("web/recon", name).read_text()
        s = s.replace("from 'three';", f"from '{THREE}/+esm';")
        s = re.sub(r"from 'three/addons/([^']+)';", lambda m: f"from '{THREE}/examples/jsm/{m.group(1)}/+esm';", s)
        assert "'three" not in s, name
        (d / name).write_text(s)


def splat_bytes(ply: Path, half: np.ndarray, margin: float) -> tuple[bytes, bytes, dict]:
    """C5a's PLY cut to the device box grown by ``margin`` (metres): the 32-byte .splat records and the
    view-dependent colour (spherical harmonics above the base colour) as one byte a coefficient,
    ``(v / scale) * 127 + 128``. Over ``MAX_SPLATS``, splats on the device (its box + 5 mm) are kept
    first, then the rest by opacity × volume."""
    d = read_ply(ply)
    m = d["means"]
    op = 1.0 / (1.0 + np.exp(-d["opacities"]))
    keep = np.all(np.abs(m) <= half + margin, axis=1) & (op > 0.02)
    idx = np.flatnonzero(keep)
    if len(idx) > MAX_SPLATS:
        near = np.all(np.abs(m[idx]) <= half + 0.005, axis=1)
        rank = op[idx] * np.exp(d["scales"][idx].sum(1)) + near * 1e9
        idx = idx[np.argsort(-rank)[:MAX_SPLATS]]
    sc = np.exp(d["scales"][idx]).astype(np.float32)
    rgb = np.clip(0.5 + C0 * d["sh0"][idx, 0, :], 0, 1)
    rgba = np.concatenate([rgb, op[idx, None]], 1)
    q = d["quats"][idx]
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(idx), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    rec["p"] = m[idx]
    rec["s"] = sc
    rec["c"] = np.clip(np.round(rgba * 255), 0, 255).astype(np.uint8)
    rec["r"] = np.clip(np.round(q * 128 + 128), 0, 255).astype(np.uint8)
    sh = d["shN"][idx][:, : (SH_DEGREE + 1) ** 2 - 1]    # (n, coefficients, rgb)
    scale = float(np.quantile(np.abs(sh), 0.999)) if sh.size else 1.0
    shq = np.clip(np.round(sh / max(scale, 1e-6) * 127 + 128), 0, 255).astype(np.uint8)
    meta = {"splats": len(idx), "splats_trained": len(m), "sh_rest": int(sh.shape[1]), "sh_scale": round(scale, 6)}
    return rec.tobytes(), shq.reshape(len(idx), -1).tobytes(), meta


def pack_views(src: Path, dst: Path, man: dict) -> None:
    """The face's WebP views in one file; the manifest gets each one's byte range."""
    blob, ranges = bytearray(), []
    for v in man["ulr"]["views"]:
        b = (src / v["file"]).read_bytes()
        ranges.append([len(blob), len(b)])
        blob += b
    write_b64(dst / "views.bin", bytes(blob))
    man["ulr"]["pack"] = "views.bin"
    for v, r in zip(man["ulr"]["views"], ranges):
        v["range"] = r


def main(out: Path) -> None:
    shutil.rmtree(out / "data", ignore_errors=True)        # no files left over from an earlier build
    shutil.copytree(SITE, out, dirs_exist_ok=True)
    copy_modules(out)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    index = []
    for ex in EXAMPLES:
        run = Path(ex["run"])
        b = run / "bundle"
        dd = data / ex["id"]
        dd.mkdir(exist_ok=True)
        strip_glb(b / "model.glb", dd / "geo.glb")
        shutil.copy2(b / "texture.jpg", dd / "tex.jpg")
        has_rough = (b / "roughness.png").exists()
        if has_rough:
            r = cv2.imread(str(b / "roughness.png"), cv2.IMREAD_UNCHANGED)
            r = cv2.resize(r, (r.shape[1] // 2, r.shape[0] // 2), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dd / "rough.png"), r, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        defects = json.loads((b / "defects.json").read_text())
        rep = json.loads((run / "run.json").read_text())
        dev = defects["device"]
        half = np.array([dev["width_mm"], dev["length_mm"], dev["thickness_mm"]]) * 0.5e-3
        real = json.loads((b / "real" / "index.json").read_text())
        faces = []
        for f in real["faces"]:
            face = f["face"]
            src = b / "real" / face
            man = json.loads((src / "manifest.json").read_text())
            same_tex = (src / "texture.jpg").read_bytes() == (b / "texture.jpg").read_bytes()
            fd = dd / f"c7_{face}"
            fd.mkdir(exist_ok=True)
            for k in ("s", "env"):
                write_b64(fd / man["srm"][k], (src / man["srm"][k]).read_bytes())
            man["files"] = {"model": "../geo.glb", "texture": "../tex.jpg" if same_tex else "texture.jpg"}
            if not same_tex:
                shutil.copy2(src / "texture.jpg", fd / "texture.jpg")
            pack_views(src, fd, man)
            (fd / "manifest.json").write_text(json.dumps(man))
            gs = None
            gdir = ex["gs"].get(face)
            ply = (gdir / "artifact" / "splat.ply") if gdir and (gdir / "artifact" / "splat.ply").exists() else (gdir / "splat.ply" if gdir else None)
            if ply and ply.exists():
                bts, shb, meta = splat_bytes(ply, half, margin=MARGIN_FRAC * float(half[:2].max() * 2))
                write_b64(dd / f"gs_{face}.splat", bts)
                write_b64(dd / f"gs_{face}_sh.bin", shb)
                gs = {"file": f"gs_{face}.splat", "sh": f"gs_{face}_sh.bin", **meta}
            p = next((q for q in rep["passes"] if q["face_up"] == face), {})
            faces.append({"face": face, "c7": f"c7_{face}/manifest.json", "gs": gs, "choice": f["choice"], "frames": f["frames"],
                          "seconds": f["seconds"], "views": len(man["ulr"]["views"]), "active": len(man["ulr"].get("active_views", [])),
                          "recording": f.get("recording"), "pass_frames": p.get("n_frames")})
        index.append({**{k: ex[k] for k in ("id", "title", "kind", "device", "place", "look")},
                      "size_mm": [round(dev["length_mm"], 1), round(dev["width_mm"], 1), round(dev["thickness_mm"], 1)],
                      "files": {"model": "geo.glb", "texture": "tex.jpg", **({"roughness": "rough.png"} if has_rough else {})},
                      "faces": faces, "defects": len(defects.get("defects", []))})
        print(ex["id"], [(f["face"], f["views"], bool(f["gs"]), f["gs"]["splats"] if f["gs"] else 0) for f in faces], flush=True)
    (data / "examples.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
