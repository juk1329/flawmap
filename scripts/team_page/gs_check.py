"""The page's three panels through one test camera of a recording (for gs_ref.py's gsplat renders).

    python scripts/team_page/gs_check.py <site_dir> <example id> <compare index.json> <test view no> <out_dir>
    python scripts/team_page/gs_check.py <site_dir> <example id> <camera.json from look_frame.py> 0 <out_dir>
"""
import asyncio
import json
import sys
from pathlib import Path

from playwright.async_api import async_playwright


async def main(site: Path, ex: str, index: Path, k: int, out: Path) -> None:
    d = json.loads(index.read_text())
    tv = d["test_views"][k] if "test_views" in d else d
    url = "http://127.0.0.1:8000/" + str(site / "_preview.html")
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--use-gl=angle", "--use-angle=gl-egl", "--ignore-gpu-blocklist"])
        pg = await b.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=2)
        logs = []
        pg.on("pageerror", lambda e: logs.append(str(e)))
        await pg.goto(url, wait_until="domcontentloaded")
        await pg.wait_for_selector(".pick")
        ids = await pg.evaluate("fetch('./data/examples.json').then(r => r.json()).then(l => l.map(x => x.id))")
        await (await pg.query_selector_all(".pick"))[ids.index(ex)].click()
        for _ in range(240):
            st = await pg.eval_on_selector_all(".mat .state", "els => els.map(e => e.hidden ? 'ok' : e.textContent)")
            if all(s == "ok" or "못했" in s for s in st):
                break
            await pg.wait_for_timeout(500)
        await pg.add_style_tag(content=".mat { background: #000 !important; } .mat-head { display: none; }")
        # the principal point at the centre, as the page's camera has it
        K = [row[:] for row in tv["K"]]
        await pg.evaluate("([T, K]) => window.__flawmapLook(T, K, 2 * K[1][2])", [tv["T_cad_cam"], K])
        await pg.wait_for_timeout(4000)
        for key in ("old", "fail", "final"):
            await (await pg.query_selector(f"#panel-{key} canvas")).screenshot(path=str(out / f"page_{key}_{k}.png"))
        print(st, logs[:5])
        await b.close()


asyncio.run(main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])))
