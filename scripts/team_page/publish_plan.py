"""The built page's files in groups small enough for one artifact publish each (64 MB, 255 files),
with the media type each needs: ``python scripts/team_page/publish_plan.py runs/artifact_cmp/site``.
Prints one JSON ``files`` map per group (sources relative to the site folder); the page itself
(``index.html``) is the publish's ``file_path``. ``_preview.html`` (shots.py) is left out."""
import json
import sys
from pathlib import Path

LIMIT = 60e6
TYPES = {".txt": "text/plain", ".js": "text/javascript", ".json": "application/json", ".jpg": "image/jpeg", ".png": "image/png"}

site = Path(sys.argv[1])
files = sorted(p for p in site.rglob("*") if p.is_file() and p.name not in ("index.html", "_preview.html"))
# the page's code and index first, then one example at a time
key = lambda p: (0 if p.parts[len(site.parts)] == "js" or p.name == "examples.json" else 1, str(p))
groups, cur, size = [], {}, 0.0
for p in sorted(files, key=key):
    rel = str(p.relative_to(site))
    b = p.stat().st_size
    if cur and (size + b > LIMIT or len(cur) >= 250):
        groups.append(cur)
        cur, size = {}, 0.0
    cur[rel] = {"from": rel, "contentType": TYPES[p.suffix]}
    size += b
if cur:
    groups.append(cur)
for i, g in enumerate(groups):
    mb = sum((site / k).stat().st_size for k in g) / 1e6
    print(f"# group {i + 1}: {len(g)} files, {mb:.1f} MB", file=sys.stderr)
print(json.dumps(groups))
