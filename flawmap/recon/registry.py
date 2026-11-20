"""The candidates, by id. Add a candidate by writing ``candidates/cN_*.py`` and listing it here."""
from __future__ import annotations

import importlib

CANDIDATES = {
    "c0": ("flawmap.recon.candidates.c0_current", "C0Current"),
    "c1": ("flawmap.recon.candidates.plate_vd", "C1SRM"),
    "c2": ("flawmap.recon.candidates.c2_ulr", "C2ULR"),
    "c3": ("flawmap.recon.candidates.plate_vd", "C3SHTexture"),
    "c4": ("flawmap.recon.candidates.c4_plate_gs", "C4PlateGS"),
    "c4b": ("flawmap.recon.candidates.c4b_plate_gs_reflect", "C4bPlateGSReflect"),
    "c5a": ("flawmap.recon.candidates.c5_free_gs", "C5aVanilla"),
    "c5b": ("flawmap.recon.candidates.c5_free_gs", "C5bMasked"),
    "c5c": ("flawmap.recon.candidates.c5_free_gs", "C5cDepth"),
    "c5d": ("flawmap.recon.candidates.c5_free_gs", "C5d2DGS"),
    "c5e": ("flawmap.recon.candidates.c5_free_gs", "C5ePlateInit"),
    "c6": ("flawmap.recon.candidates.c6_tsdf", "C6TSDF"),
    "c7": ("flawmap.recon.candidates.c7_hybrid", "C7Hybrid"),
}


def get(cid: str):
    if cid not in CANDIDATES:
        raise KeyError(f"unknown candidate {cid!r}; known: {', '.join(CANDIDATES)}")
    mod, name = CANDIDATES[cid]
    return getattr(importlib.import_module(mod), name)
