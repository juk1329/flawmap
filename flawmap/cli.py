"""Command line: ``flawmap <command>`` (also ``python -m flawmap.cli``)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_board(args: argparse.Namespace) -> int:
    from .anchor.parametric import DeviceSpec, build_device_model
    from .poses.board import BoardSpec, render_board_page, save_board_pdf

    spec = BoardSpec.load(args.board)
    outline, name = None, ""
    if args.device:
        dev = DeviceSpec.by_slug(args.device)
        outline, name = build_device_model(dev).outline_xy, dev.name
    elif args.size:
        # no drawing: a rounded rectangle of the measured length, width and corner radius as the guide
        from .anchor.template import template_spec

        L, W = args.size[0], args.size[1]
        r = args.size[2] if len(args.size) > 2 else 0.06 * W
        outline, name = build_device_model(template_spec(L, W, 6.0, r)).outline_xy, f"{L:.1f} x {W:.1f} mm"
    page = render_board_page(spec, outline, name)
    tag = args.device or ("x".join(f"{v:g}" for v in args.size[:2]) if args.size else "blank")
    out = Path(args.out or f"runs/board/board_{spec.name}_{tag}.pdf")
    save_board_pdf(page, out, spec)
    print(f"wrote {out} ({spec.page_mm[0]:.0f} x {spec.page_mm[1]:.0f} mm) and {out.with_suffix('.preview.png')}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import PassInput, run

    passes = []
    for spec in args.passes:
        face, _, path = spec.partition("=")
        if face not in ("screen", "back") or not path:
            raise SystemExit(f"--pass expects screen=PATH or back=PATH, got {spec!r}")
        passes.append(PassInput(Path(path), face, placement=args.placement))
    out = Path(args.out or f"runs/{args.device}_{Path(passes[0].root).name}")
    meta = {}
    if args.holdout == "blocks":
        # time blocks held out for judging models against real frames (eval.heldout); nothing learns from them
        import json

        from .eval.heldout import block_split
        from .io.stray import read_stray

        splits = {}
        for p in passes:
            sp = block_split(read_stray(p.root).timestamps)
            p.exclude = sp.excluded
            splits[p.name] = sp.to_json()
        out.mkdir(parents=True, exist_ok=True)
        (out / "holdout.json").write_text(json.dumps(splits))
        meta["holdout"] = {k: {"test": len(v["test"]), "val": len(v["val"]), "guard": len(v["guard"]), "train": v["train"]}
                           for k, v in splits.items()}
    best = args.quality == "max"
    rep = run(passes, args.device, out, keyframes=args.keyframes, board_square_mm=args.square_mm, poses=args.poses,
              board_path=args.board, refine="colmap" if best else None, sharp_views=best, segment=best, meta=meta)
    print(f"{len(rep['defects'])} defects, coverage {rep['coverage']['overall']:.1%}, {rep['timing_s']['total']:.0f}s -> {out / 'bundle'}")
    if args.realistic:
        _realistic(out)
    return 0


def _realistic(run_dir: Path, keep_work: bool = False) -> None:
    from .recon.service import build_realistic

    idx = build_realistic(run_dir, keep_work=keep_work)
    for f in idx["faces"]:
        print(f"realistic view of the {f['face']}: {f['seconds']:.0f} s, {f['web_mb']} MB -> {run_dir / 'bundle' / f['manifest']}")
    for f in idx["failed"]:
        print(f"realistic view of the {f['face']} failed: {f['error']}")


def cmd_realistic(args: argparse.Namespace) -> int:
    _realistic(Path(args.run), keep_work=args.keep_work)
    return 0


def cmd_realistic_backfill(args: argparse.Namespace) -> int:
    """The realistic view for every finished result of the service that has none yet (results made
    before it existed), one after another; the requester pages' summaries are refreshed."""
    from .recon.service import build_realistic
    from .service.job import summarize
    from .service.store import Store

    store = Store(Path(args.data))
    rows = [d for d in store.list(status="done", limit=100000)
            if (args.again or not (Path(d["run_dir"]) / "bundle" / "real" / "index.json").exists())
            and (args.superseded or not d.get("superseded_by"))]
    print(f"{len(rows)} results without the realistic view")
    for d in rows:
        rd = Path(d["run_dir"])
        if not (rd / "run.json").exists():
            continue
        idx = build_realistic(rd)
        store.update(d["id"], summary=summarize(rd))
        print(f"{d['id']} {rd}: faces {[f['face'] for f in idx['faces']]}, failed {[f['face'] for f in idx['failed']]}, {idx['seconds']:.0f} s",
              flush=True)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .recon.runner import run_compare

    idx = run_compare(args.base_run, [c.strip() for c in args.candidates.split(",") if c.strip()], args.out, refit=args.refit,
                      stress=not args.no_stress, round_name=args.round, turn_only=args.turn_only)
    for c in idx.get("candidates", []):
        s = c["summary"]
        print(f"{c['id']:4s} {c['label']}: held-out PSNR {s.get('psnr_g')} (gain fitted), LPIPS {s.get('lpips_g')}, "
              f"floaters {s.get('turn30_float')}, holes under {s.get('under_hole')}, flicker {s.get('flicker')}, "
              f"turn jump {s.get('turn_jump')}")
    print(f"-> {Path(args.out) / idx['rec']} (open /compare.html?rec={idx['rec']} on the flawmap server)")
    return 0


def cmd_compare_sheet(args: argparse.Namespace) -> int:
    from .recon.sheet import write_sheet

    print(write_sheet(Path(args.out) / args.rec, frame=args.frame))
    return 0


def cmd_compare_blind(args: argparse.Namespace) -> int:
    from .recon.blind import make_blind_pack

    base = [c.strip() for c in args.cands.split(",") if c.strip()]
    extra = {}
    for e in args.extra:
        rec, _, cs = e.partition("=")
        extra[rec.strip()] = [c.strip() for c in cs.split(",") if c.strip()]
    pack = Path(args.out) / args.pack
    for k, rec in enumerate(args.recs):
        cands = base + [c for c in extra.get(rec, []) if c not in base]
        # every recording gets its own order, also with a fixed seed
        make_blind_pack(Path(args.out) / rec, cands, pack, seed=None if args.seed is None else args.seed + k)
        print(f"{rec}: {len(cands)} candidates under letters -> {pack / rec}")
    print(f"which letter is which: {pack.with_name(pack.name + '.key.json')} (open only after rating)")
    return 0


def cmd_uco3d(args: argparse.Namespace) -> int:
    from .pipeline import run_uco3d

    out = Path(args.out or f"runs/uco3d_{args.name}")
    rep = run_uco3d(args.name, out, face_up=args.face_up, data_root=args.data, keyframes=args.keyframes)
    t = rep["passes"][0]["template"]
    print(f"{len(rep['defects'])} defects, coverage {rep['coverage']['overall']:.1%}, plate {t['size_mm'][0]:.1f} x "
          f"{t['size_mm'][1]:.1f} x {t['size_mm'][2]:.1f} (width assumed), {rep['timing_s']['total']:.0f}s -> {out / 'bundle'}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from .check import check_capture

    out = Path(args.out or f"runs/check/{Path(args.capture).name}")
    rep = check_capture(args.capture, out, device=args.device, face_up=args.face_up, square_mm=args.square_mm)
    print(f"report: {out / 'report.json'}")
    for b in rep["board_by_height"]:
        print(f"  height {b['elev_deg'][0]:>2}-{b['elev_deg'][1]:<2} deg: board in {b['board_found']}/{b['frames']} frames, "
              f"reproj {b['median_reproj_px'] if b['median_reproj_px'] is None else round(b['median_reproj_px'], 2)} px")
    if "registration" in rep:
        print(f"  model outline vs image edges: median {rep['registration']['median_px']:.2f} px = {rep['registration']['median_mm']:.3f} mm")
    if "outline" in rep:
        print(f"  outline profile: {rep['outline']['n_samples']} samples, spread {rep['outline']['sigma_mm']:.3f} mm, dents {rep['outline']['dents']}")
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    from .batch import main_summary, run_batch
    from .utils.logging import add_file_handler

    add_file_handler(Path(args.out) / "batch.log")
    summary = run_batch(Path(args.root), Path(args.out), Path(args.gates) if args.gates else None, force=args.force)
    print(f"summary: {Path(args.out) / 'summary.md'}")
    return main_summary(summary)


def cmd_serve(args: argparse.Namespace) -> int:
    from .server.app import serve

    data = None if args.no_service else Path(args.data)
    serve(Path(args.runs), host=args.host, port=args.port, data_dir=data, seed=not args.no_examples)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="flawmap")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("board", help="write the printable A3 capture board (PDF)")
    b.add_argument("--device", help="device slug in configs/devices (prints its outline on the grey area)")
    b.add_argument("--board", help="board YAML (default configs/board_a3.yaml; tablets: configs/board_a2.yaml)")
    b.add_argument("--size", type=float, nargs="+", metavar="MM",
                   help="no drawing: print a rounded-rectangle guide of LENGTH WIDTH [CORNER_RADIUS] mm")
    b.add_argument("--out", help="output PDF path")
    b.set_defaults(func=cmd_board)

    r = sub.add_parser("run", help="process one device: capture pass(es) -> defects -> viewer bundle")
    r.add_argument("--device", required=True, help="device slug in configs/devices")
    r.add_argument("--pass", dest="passes", action="append", required=True,
                   help="screen=PATH or back=PATH (Stray Scanner folder); give one or both")
    r.add_argument("--out", help="output folder (default runs/<device>_<capture>)")
    r.add_argument("--keyframes", type=int, default=140)
    r.add_argument("--square-mm", type=float, help="measured board square size of the printout (default 25.0)")
    r.add_argument("--poses", default="auto", choices=["auto", "board", "boardless"],
                   help="camera poses from the printed board, without it (ARKit + model outline), or whichever fits (default)")
    r.add_argument("--board", help="board YAML (default configs/board_a3.yaml; tablets: configs/board_a2.yaml)")
    r.add_argument("--quality", default="standard", choices=["standard", "max"],
                   help="max: SAM 2 device masks to check the registration and keep floor out of the texture, COLMAP "
                        "refinement of boardless poses, textures and detections from the sharpest views "
                        "(slower; needs torch, sam2 and pycolmap)")
    r.add_argument("--placement", default="auto", choices=["auto", "flat"],
                   help="auto: also handle a device standing on a stand or leaning (anchor.placement); flat: the floor-only path")
    r.add_argument("--holdout", default="none", choices=["none", "blocks"],
                   help="blocks: hold out 1-second time blocks (with guard bands) from everything that learns from frames, "
                        "for judging models against real frames later (flawmap compare); written to <out>/holdout.json")
    r.add_argument("--realistic", action="store_true",
                   help="also build the realistic view (C7: measured reflection + real frames, flawmap.recon.service) into the bundle")
    r.set_defaults(func=cmd_run)

    rl = sub.add_parser("realistic", help="add the realistic view (C7, flawmap.recon.service) to a finished run's bundle")
    rl.add_argument("run", help="a finished `flawmap run` folder (runs/<name>)")
    rl.add_argument("--keep-work", action="store_true", help="keep <run>/realistic (the fitted C1/C7 artifacts) after the export")
    rl.set_defaults(func=cmd_realistic)

    rb = sub.add_parser("realistic-backfill", help="add the realistic view to every finished service result that has none")
    rb.add_argument("--data", default="data/service", help="the service's data folder")
    rb.add_argument("--again", action="store_true", help="build it again where it exists")
    rb.add_argument("--superseded", action="store_true", help="also results replaced by a newer one")
    rb.set_defaults(func=cmd_realistic_backfill)

    cp = sub.add_parser("compare", help="fit reconstruction candidates on one recording and score them side by side (recon)")
    cp.add_argument("--base-run", required=True, help="a finished `flawmap run ... --holdout blocks` of one recording")
    cp.add_argument("--candidates", default="c0,c2", help="comma-separated ids (recon.registry), e.g. c0,c2,c5a")
    cp.add_argument("--out", default="runs/compare")
    cp.add_argument("--refit", action="store_true", help="fit again even if the candidate's artifact exists")
    cp.add_argument("--no-stress", action="store_true", help="skip the stress views (held-out frames only)")
    cp.add_argument("--round", default="", help="label of this comparison round (stored with the scores)")
    cp.add_argument("--turn-only", action="store_true",
                    help="render and score only the turntable again (flicker, jumps), keeping the other scores")
    cp.set_defaults(func=cmd_compare)

    cs = sub.add_parser("compare-sheet", help="one picture per recording: every candidate's held-out render and stress views")
    cs.add_argument("rec", help="recording id under runs/compare")
    cs.add_argument("--out", default="runs/compare")
    cs.add_argument("--frame", type=int, help="held-out frame to show (default: the median one)")
    cs.set_defaults(func=cmd_compare_sheet)

    cb = sub.add_parser("compare-blind", help="blind rating files: a set of candidates under shuffled letters (videos and pictures), "
                                              "the letters kept in a separate key file")
    cb.add_argument("recs", nargs="+", help="recording ids under runs/compare")
    cb.add_argument("--cands", default="c0,c1,c2,c7,c4b", help="comma-separated candidate ids rated together")
    cb.add_argument("--extra", action="append", default=[], metavar="REC=c5d,...", help="more candidates for one recording")
    cb.add_argument("--pack", default="blind", help="folder under --out (and <pack>.key.json beside it)")
    cb.add_argument("--out", default="runs/compare")
    cb.add_argument("--seed", type=int, help="fixed shuffle (default: random; stored in the key file)")
    cb.set_defaults(func=cmd_compare_blind)

    u = sub.add_parser("uco3d", help="process a public uCO3D phone video without a reference drawing (fitted plate model)")
    u.add_argument("name", help="sequence name, e.g. 59015-27660-21826")
    u.add_argument("--face-up", default="screen", choices=["screen", "back"], help="which face the video shows")
    u.add_argument("--data", default="data/datasets/uco3d")
    u.add_argument("--keyframes", type=int, default=100)
    u.add_argument("--out")
    u.set_defaults(func=cmd_uco3d)

    c = sub.add_parser("check", help="first look at a capture: decoding, board per height, ARKit drift, model fit")
    c.add_argument("capture", help="Stray Scanner folder")
    c.add_argument("--device", help="device slug: also fit the reference model and write overlays")
    c.add_argument("--face-up", default="back", choices=["screen", "back"])
    c.add_argument("--square-mm", type=float)
    c.add_argument("--out")
    c.set_defaults(func=cmd_check)

    bt = sub.add_parser("batch", help="process every device folder with a manifest.yaml (capture spec v2) with fixed gates")
    bt.add_argument("root", help="a device folder with manifest.yaml, or a folder of them (e.g. data/captures)")
    bt.add_argument("--out", default="runs/batch", help="results and summary.json / summary.md (default runs/batch)")
    bt.add_argument("--gates", help="gate config (default configs/batch.yaml)")
    bt.add_argument("--force", action="store_true", help="process again even when a result exists")
    bt.set_defaults(func=cmd_batch)

    s = sub.add_parser("serve", help="inspection service (upload, queue, seller and buyer pages) and the viewer")
    s.add_argument("--runs", default="runs", help="development runs shown at /?run=<name>")
    s.add_argument("--data", default="data/service", help="service storage: database, uploads, results")
    s.add_argument("--no-service", action="store_true", help="viewer only, no uploads")
    s.add_argument("--no-examples", action="store_true", help="do not list runs/synth_demo and synth_dent as example listings")
    s.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept other machines on the network")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
