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
    page = render_board_page(spec, outline, name)
    out = Path(args.out or f"runs/board/board_{spec.name}_{args.device or 'blank'}.pdf")
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
        passes.append(PassInput(Path(path), face))
    out = Path(args.out or f"runs/{args.device}_{Path(passes[0].root).name}")
    rep = run(passes, args.device, out, keyframes=args.keyframes, board_square_mm=args.square_mm, poses=args.poses)
    print(f"{len(rep['defects'])} defects, coverage {rep['coverage']['overall']:.1%}, {rep['timing_s']['total']:.0f}s -> {out / 'bundle'}")
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
    b.add_argument("--board", help="board YAML (default configs/board_a3.yaml)")
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
    r.set_defaults(func=cmd_run)

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
