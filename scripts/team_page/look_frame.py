"""The page's three panels through one real frame's camera, beside that frame: checks that a model
sits where the recording saw the device (any example, including public videos).

    python scripts/team_page/look_frame.py <run_dir> <pass_index> <frame no> <out_json>    # env_isaaclab: camera + frame
then gs_check.py's browser part reads <out_json> (``--from-json``).
"""
import json
import sys
from pathlib import Path

import cv2

from flawmap.recon.context import context_for_pass

run, k, i, out = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
ctx = context_for_pass(run, k, out.parent / "ctx_tmp")
view = ctx.view(i)
img = ctx.image(i)
cv2.imwrite(str(out.with_suffix(".jpg")), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
out.write_text(json.dumps({"K": view.K.tolist(), "T_cad_cam": view.T_cad_cam.tolist(), "W": view.W, "H": view.H, "frame": i}))
print(view.W, view.H, len(ctx.train_frames))
