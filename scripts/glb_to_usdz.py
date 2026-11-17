"""GLB (one textured mesh) -> USDZ for macOS Quick Look / Preview.

The bundle's model.glb carries only a placeholder texture (the web viewer loads texture.jpg), so pass
the real texture:

    python scripts/glb_to_usdz.py runs/<run>/bundle/model.glb out.usdz runs/<run>/bundle/texture.jpg
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import trimesh

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
s = trimesh.load(src)
g = list(s.dump()) if hasattr(s, "dump") else [s]
m = g[0] if len(g) == 1 else trimesh.util.concatenate(g)
V = np.asarray(m.vertices, float)
F = np.asarray(m.faces, int)
UV = np.asarray(m.visual.uv, float)          # trimesh: origin bottom-left (same as USD st)
N = np.asarray(m.vertex_normals, float)
from PIL import Image
img = Image.open(sys.argv[3]) if len(sys.argv) > 3 else m.visual.material.baseColorTexture
buf = io.BytesIO()
img.convert("RGB").save(buf, "JPEG", quality=95)
tex = buf.getvalue()


def arr3(a):
    return ", ".join(f"({x:.6g}, {y:.6g}, {z:.6g})" for x, y, z in a)


lo, hi = V.min(0), V.max(0)
usda = f"""#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "Root"
{{
    def Mesh "Device"
    {{
        float3[] extent = [({lo[0]:.6g}, {lo[1]:.6g}, {lo[2]:.6g}), ({hi[0]:.6g}, {hi[1]:.6g}, {hi[2]:.6g})]
        int[] faceVertexCounts = [{", ".join(["3"] * len(F))}]
        int[] faceVertexIndices = [{", ".join(map(str, F.ravel()))}]
        point3f[] points = [{arr3(V)}]
        normal3f[] normals = [{arr3(N)}] (
            interpolation = "vertex"
        )
        texCoord2f[] primvars:st = [{", ".join(f"({u:.6g}, {v:.6g})" for u, v in UV)}] (
            interpolation = "vertex"
        )
        uniform token subdivisionScheme = "none"
        uniform bool doubleSided = 1
        rel material:binding = </Root/Mat>
    }}

    def Material "Mat"
    {{
        token outputs:surface.connect = </Root/Mat/PBR.outputs:surface>

        def Shader "PBR"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = </Root/Mat/Tex.outputs:rgb>
            float inputs:roughness = 0.6
            float inputs:metallic = 0
            token outputs:surface
        }}

        def Shader "Reader"
        {{
            uniform token info:id = "UsdPrimvarReader_float2"
            string inputs:varname = "st"
            float2 outputs:result
        }}

        def Shader "Tex"
        {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @texture.jpg@
            float2 inputs:st.connect = </Root/Mat/Reader.outputs:result>
            token inputs:wrapS = "clamp"
            token inputs:wrapT = "clamp"
            token inputs:sourceColorSpace = "sRGB"
            float3 outputs:rgb
        }}
    }}
}}
"""


def add(zf, name, data):
    # USDZ: stored (no compression), each file's data 64-byte aligned (pad the local header extra field)
    zi = zipfile.ZipInfo(name, date_time=(2026, 11, 17, 0, 0, 0))
    zi.compress_type = zipfile.ZIP_STORED
    off = zf.fp.tell() + 30 + len(name.encode())
    pad = (-(off + 4)) % 64
    zi.extra = b"\x34\x12" + pad.to_bytes(2, "little") + b"\0" * pad
    zf.writestr(zi, data)


with zipfile.ZipFile(dst, "w") as zf:
    add(zf, "model.usda", usda.encode())
    add(zf, "texture.jpg", tex)
with zipfile.ZipFile(dst) as zf:
    for zi in zf.infolist():
        data_off = zi.header_offset + 30 + len(zi.filename.encode()) + len(zi.extra)
        print(zi.filename, zi.file_size, "aligned" if data_off % 64 == 0 else f"NOT aligned {data_off % 64}")
print(len(V), "verts", len(F), "faces", img.size, "texture", dst.stat().st_size, "bytes")
