#!/usr/bin/env python3
"""
Create conditioning HDF5 for generation (num_points_per_layer from G4 showers).

Reads a merged shower file, computes num_points_per_layer directly from the
shower hit data (real G4 counts per layer), and saves the first N showers
as a conditioning file.

Usage:
    python scripts/lemurs/create_cond.py [--n 2000] [--output allegro_2k_cond.h5]
    python scripts/lemurs/create_cond.py --detector odd --src output_dataset/ODD/odd_150k.h5 --n 10000
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

BASE   = Path(__file__).resolve().parents[2]   # repo root (this file lives in scripts/lemurs/)
SRC    = BASE / "output_dataset/ALLEGRO/h5/allegro_100k.h5"
OUTDIR = BASE.parent / "AllShowers-AllGeometries/data"

DETECTOR_CONFIGS = {
    "allegro":     {"num_layers": 45, "default_src": "output_dataset/ALLEGRO/h5/allegro_100k.h5"},
    "odd":         {"num_layers": 48, "default_src": "output_dataset/ODD/odd_150k.h5"},
    "cld":         {"num_layers": 45, "default_src": "output_dataset/FCCee_CLD/cld_150k.h5"},
    "par04_siw":   {"num_layers": 90, "default_src": "output_dataset/Par04_SiW/par04_siw_150k.h5"},
    "par04_scipb": {"num_layers": 45, "default_src": "output_dataset/Par04_SciPb/par04_scipb_150k.h5"},
}

NUM_LAYERS = 45   # default, overridden by --detector


def compute_num_points_per_layer(src_path, n):
    """Count hits per layer for the first N showers directly from the HDF5."""
    npl = np.zeros((n, NUM_LAYERS), dtype=np.int32)
    with h5py.File(src_path, "r", locking=False) as f:
        showers_ds = f["showers"]
        shape_meta = f["shape"][:]
        n_cols = int(shape_meta[2])   # 5 for ALLEGRO
        for i in tqdm(range(n), desc="Computing num_points_per_layer"):
            raw = showers_ds[i]
            if len(raw) == 0:
                continue
            nhits = len(raw) // n_cols
            pts   = raw.reshape(nhits, n_cols)
            layers = np.clip(pts[:, 2].astype(int), 0, NUM_LAYERS - 1)
            for l in layers:
                npl[i, l] += 1
    return npl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="allegro", choices=list(DETECTOR_CONFIGS.keys()),
                    help="Detector name (default: allegro)")
    ap.add_argument("--n",      type=int,   default=2000,
                    help="Number of conditioning showers (default: 2000)")
    ap.add_argument("--src",    default=None,
                    help="Source H5 path (default: from detector config)")
    ap.add_argument("--output", default=None,
                    help="Output filename (default: <det>_Nk_cond.h5)")
    args = ap.parse_args()

    global NUM_LAYERS
    det_cfg = DETECTOR_CONFIGS[args.detector]
    NUM_LAYERS = det_cfg["num_layers"]

    src = Path(args.src) if args.src else BASE / det_cfg["default_src"]
    n   = args.n
    out_name = args.output or f"{args.detector}_{n//1000}k_cond.h5"
    out_path = OUTDIR / out_name

    print(f"Source : {src}")
    print(f"Output : {out_path}")
    print(f"Showers: {n}")

    with h5py.File(src, "r", locking=False) as f:
        total = len(f["num_points"])
        if n > total:
            raise ValueError(f"Requested {n} showers but source only has {total}")
        print(f"  Source total showers: {total:,}")

        energies   = f["energies"][:n].astype(np.float32)          # (n,1)
        directions = f["directions"][:n].astype(np.float32)        # (n,3)
        sf         = f["sampling_fraction"][:n].astype(np.float32) # (n,1)
        num_layers = f["num_layers"][:n].astype(np.int32)          # (n,1)
        pdg        = f["pdg"][:n].astype(np.int32)                 # (n,)
        layer_z_pos = f["layer_z_pos"][:n].astype(np.float32)      # (n,45)

        # Use precomputed observables if available, else compute on-the-fly
        if "observables" in f and "num_points_per_layer" in f["observables"]:
            print("  Reading num_points_per_layer from observables group …")
            npl = f["observables/num_points_per_layer"][:n].astype(np.int32)
        else:
            print("  No observables group — computing num_points_per_layer from showers …")
            npl = compute_num_points_per_layer(src, n)

    sf_vals = sf.flatten()
    print(f"\n  SF: min={sf_vals.min():.4f}  max={sf_vals.max():.4f}  mean={sf_vals.mean():.4f}")
    print(f"  num_layers unique: {np.unique(num_layers.flatten()).tolist()}")
    print(f"  num_points_per_layer: total/shower min={npl.sum(1).min()}  max={npl.sum(1).max()}  mean={npl.sum(1).mean():.0f}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("energies",             data=energies)
        f.create_dataset("directions",           data=directions)
        f.create_dataset("sampling_fraction",    data=sf)
        f.create_dataset("num_layers",           data=num_layers)
        f.create_dataset("pdg",                  data=pdg)
        f.create_dataset("layer_z_pos",          data=layer_z_pos)
        f.create_dataset("num_points_per_layer", data=npl)
        f.attrs["n_showers"]  = n
        f.attrs["source"]     = str(src)
        f.attrs["description"] = f"{args.detector} generation conditioning: G4 num_points_per_layer"

    print(f"\nSaved: {out_path}")
    with h5py.File(out_path, "r") as f:
        for k in f.keys():
            d = f[k]
            print(f"  {k}: {d.shape}")


if __name__ == "__main__":
    main()
