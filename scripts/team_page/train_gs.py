"""C5a (plain 3D Gaussian splatting, the whole frame: what a splatting app does) for one pass of a run,
for the team's comparison page: the method that looked right from the filmed views and broke when
turned.

    python scripts/team_page/train_gs.py <run_dir> <pass_index> <out_dir>
"""
import json
import sys
from pathlib import Path

from flawmap.recon.candidates.c5_free_gs import C5aVanilla
from flawmap.recon.context import context_for_pass

run, k, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
ctx = context_for_pass(run, k, out / "work")
ctx.ensure_frames()
cand = C5aVanilla(ctx, out=out)
fit = cand.run_fit()
print(run.name, k, json.dumps({kk: fit.get(kk) for kk in ("splats", "frames", "seconds", "artifact_mb")}), flush=True)
