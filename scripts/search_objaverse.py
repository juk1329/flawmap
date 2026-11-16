"""Search the Objaverse 1.0 metadata (Sketchfab objects, per-object CC licences) for reference
models of the target devices. Downloads the 160 metadata shards (577 MB, no login) into
data/datasets/objaverse/metadata and writes runs/objaverse_search.json."""
import gzip
import json
import re
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

root = Path(__file__).resolve().parents[1]
out_dir = root / "data" / "datasets" / "objaverse"
pat = re.compile(r"iphone\s?(11|12)\b|ipad\s?(air|mini|9|10|pro)|ipad\b", re.I)
hits = []
for k in range(160):
    name = f"metadata/000-{k:03d}.json.gz"
    p = hf_hub_download("allenai/objaverse", name, repo_type="dataset", local_dir=out_dir)
    with gzip.open(p, "rt") as f:
        meta = json.load(f)
    for uid, m in meta.items():
        text = " ".join([m.get("name") or "", " ".join(t.get("name", "") for t in m.get("tags", [])), (m.get("description") or "")[:300]])
        if pat.search(text):
            hits.append({"uid": uid, "name": m.get("name"), "license": m.get("license"), "faces": m.get("faceCount"),
                         "url": m.get("viewerUrl"), "tags": [t.get("name") for t in m.get("tags", [])][:12]})
    if k % 20 == 0:
        print(f"{k + 1}/160 shards, {len(hits)} hits", flush=True)
(root / "runs").mkdir(exist_ok=True)
json.dump(hits, open(root / "runs" / "objaverse_search.json", "w"), ensure_ascii=False, indent=1)
print(f"done: {len(hits)} hits")
