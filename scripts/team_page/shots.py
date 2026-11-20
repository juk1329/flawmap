"""Look at the team's comparison page before it is published: served by the demo service's local
/runs route, wrapped the way the artifact host wraps a page, under a content policy as strict as the
host's (no WebAssembly, workers only from the page's own files, fetches to the page's own files).

    python scripts/team_page/shots.py <site_dir> <out_dir> [example ids...]
"""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net/npm/ https://cdnjs.cloudflare.com https://unpkg.com; "
       "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; img-src 'self' data: blob:; "
       "connect-src 'self'; worker-src 'self' blob:")


async def main(site: Path, out: Path, only: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    page_html = (site / "index.html").read_text()
    (site / "_preview.html").write_text(
        f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<meta http-equiv="Content-Security-Policy" content="{CSP}"></head><body>{page_html}</body></html>')
    url = "http://127.0.0.1:8000/" + str(site / "_preview.html")
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--use-gl=angle", "--use-angle=gl-egl", "--ignore-gpu-blocklist"])
        for scheme, vw in (("light", 1440), ("dark", 1440), ("light", 390)):
            pg = await b.new_page(viewport={"width": vw, "height": 900}, color_scheme=scheme)
            logs = []
            pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
            pg.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
            await pg.goto(url, wait_until="domcontentloaded")
            await pg.wait_for_selector(".pick", timeout=60000)
            picks = await pg.query_selector_all(".pick")
            ids = await pg.evaluate("fetch('./data/examples.json').then(r => r.json()).then(l => l.map(x => x.id))")
            for i, ex in enumerate(ids):
                if only and ex not in only:
                    continue
                if vw < 800 and i > 0:
                    break
                await picks[i].click()
                # polled from here: a string predicate in the page would be blocked by the content policy
                for _ in range(240):
                    st = await pg.eval_on_selector_all(".mat .state", "els => els.map(e => e.hidden || /못했/.test(e.textContent))")
                    if all(st):
                        break
                    await pg.wait_for_timeout(500)
                else:
                    logs.append(f"{ex}: still loading after 120 s")
                await pg.wait_for_timeout(2500)
                states = await pg.eval_on_selector_all(".mat .state", "els => els.map(e => e.hidden ? 'ok' : e.textContent)")
                tag = f"{scheme}_{vw}_{ex}"
                bench = await pg.query_selector("section[aria-label='세 모델 비교']")
                await bench.screenshot(path=str(out / f"{tag}_oblique.png"))
                if vw >= 800 and scheme == "light":
                    for view in ("top", "side", "flip"):
                        await pg.click(f"[data-view='{view}']")
                        await pg.wait_for_timeout(2500)
                        await bench.screenshot(path=str(out / f"{tag}_{view}.png"))
                    await pg.click("[data-view='oblique']")
                    # a drag on the middle panel: all three turn
                    box = await (await pg.query_selector("#panel-fail canvas")).bounding_box()
                    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
                    await pg.mouse.move(cx, cy)
                    await pg.mouse.down()
                    await pg.mouse.move(cx + 160, cy + 40, steps=12)
                    await pg.mouse.up()
                    await pg.wait_for_timeout(2500)
                    await bench.screenshot(path=str(out / f"{tag}_dragged.png"))
                print(tag, states, flush=True)
            if scheme == "light" and vw >= 800:
                await pg.screenshot(path=str(out / "full_page.png"), full_page=True)
            if vw < 800:
                await pg.screenshot(path=str(out / "mobile_page.png"), full_page=True)
                sw = await pg.evaluate("[document.documentElement.scrollWidth, document.documentElement.clientWidth]")
                print("mobile scroll width", sw)
            for line in logs[:30]:
                print("  ", line[:300])
            await pg.close()
        await b.close()


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]))
