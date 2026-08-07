#!/usr/bin/env python3
"""
Merge LEMURS H5 files: per-detector 10k files → single per-detector H5 → combined lemurs_4M.h5

Step 1: merge 10k individual files per detector into <det>_1M.h5
Step 2: combine 4 per-detector files into lemurs_4M.h5 (pad layer_z_pos to max_layers=90)

Usage:
    python merge_h5_lemurs.py --step merge-detector --detector par04_siw --h5-subdirs h5_150k_100gev h5_850k_100gev
    python merge_h5_lemurs.py --step merge-detector --detector all --h5-subdirs h5_150k_100gev h5_850k_100gev
    python merge_h5_lemurs.py --step combine-all
    python merge_h5_lemurs.py --step all --h5-subdirs h5_150k_100gev h5_850k_100gev
"""
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

_REPO = Path(os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parent.parent.parent)))
BASE = _REPO / "output_dataset"

DETECTORS = {
    "par04_siw":   {"dir": "Par04_SiW",   "max_layers": 90},
    "par04_scipb": {"dir": "Par04_SciPb",  "max_layers": 45},
    "odd":         {"dir": "ODD",          "max_layers": 48},
    "fccee_cld":   {"dir": "FCCee_CLD",    "max_layers": 45},
}

MAX_LAYERS = 90  # pad all to this for combined file

# Datasets to merge and their per-event shapes (None = scalar/1D)
FIXED_DATASETS = {
    "energies":          (1,),
    "directions":        (3,),
    "num_points":        None,       # shape (N,)
    "num_layers":        (1,),
    "sampling_fraction": (1,),
    "pdg":               None,       # shape (N,)
    "shower_ids":        None,       # shape (N,)
    "gun_position":      (3,),
}
# layer_z_pos handled separately (variable width per detector)
# showers = vlen, handled separately


def count_events(h5_files):
    """Pre-scan all files to get total event count."""
    total = 0
    for f in h5_files:
        with h5py.File(f, "r") as hf:
            total += hf["energies"].shape[0]
    return total


def merge_detector(det_name, max_files=None, h5_subdirs=None, output_name=None):
    """Merge individual H5 files for one detector into a single H5."""
    cfg = DETECTORS[det_name]
    subdirs = h5_subdirs or ["h5_1M"]
    h5_files = []
    for subdir in subdirs:
        h5_dir = BASE / cfg["dir"] / subdir
        h5_files.extend(sorted(h5_dir.glob("*.h5")))
    if max_files:
        h5_files = h5_files[:max_files]

    if not h5_files:
        print(f"  No H5 files found in {[str(BASE / cfg['dir'] / s) for s in subdirs]}")
        return None

    out_name = output_name or f"{det_name}_1M.h5"
    out_path = BASE / cfg["dir"] / out_name
    n_total = count_events(h5_files)
    print(f"  {det_name}: {len(h5_files)} files, {n_total} events → {out_path.name}")

    # Read first file for dtypes and layer_z_pos width
    with h5py.File(h5_files[0], "r") as ref:
        lzp_width = ref["layer_z_pos"].shape[1]
        vlen_dt = h5py.special_dtype(vlen=np.float32)

    with h5py.File(out_path, "w") as out:
        # Create datasets
        ds = {}
        for name, per_evt in FIXED_DATASETS.items():
            with h5py.File(h5_files[0], "r") as ref:
                dtype = ref[name].dtype
            if per_evt is None:
                ds[name] = out.create_dataset(name, shape=(n_total,), dtype=dtype)
            else:
                ds[name] = out.create_dataset(name, shape=(n_total, *per_evt), dtype=dtype)

        ds["layer_z_pos"] = out.create_dataset(
            "layer_z_pos", shape=(n_total, lzp_width), dtype=np.float32
        )
        ds["showers"] = out.create_dataset(
            "showers", shape=(n_total,), dtype=vlen_dt
        )

        # Fill
        idx = 0
        for fpath in tqdm(h5_files, desc=f"  {det_name}", unit="file"):
            with h5py.File(fpath, "r") as hf:
                n = hf["energies"].shape[0]
                for name in FIXED_DATASETS:
                    ds[name][idx : idx + n] = hf[name][:]
                ds["layer_z_pos"][idx : idx + n] = hf["layer_z_pos"][:]
                # vlen: read all at once, write slice
                showers_batch = [hf["showers"][i] for i in range(n)]
                ds["showers"][idx : idx + n] = showers_batch
                idx += n

        # Update shower_ids to be sequential
        ds["shower_ids"][:] = np.arange(n_total, dtype=np.int32)

        # Copy attributes from first file + update count
        with h5py.File(h5_files[0], "r") as ref:
            for attr_name, attr_val in ref.attrs.items():
                out.attrs[attr_name] = attr_val
        out.attrs["n_showers"] = n_total
        out.attrs["n_files"] = len(h5_files)

        # shape dataset
        max_hits = int(np.max(ds["num_points"][:]))
        out.create_dataset("shape", data=np.array([n_total, max_hits, 5], dtype=np.int64))

    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")
    return out_path


def combine_all(det_files=None):
    """Combine per-detector merged files into lemurs_4M.h5, padding layer_z_pos to MAX_LAYERS."""
    if det_files is None:
        det_files = {}
        for det_name, cfg in DETECTORS.items():
            p = BASE / cfg["dir"] / f"{det_name}_1M.h5"
            if p.exists():
                det_files[det_name] = p
            else:
                print(f"  WARNING: {p} not found, skipping {det_name}")

    if not det_files:
        print("  No per-detector files found!")
        return None

    out_path = BASE / "lemurs_4M.h5"
    vlen_dt = h5py.special_dtype(vlen=np.float32)

    # Count total events
    n_total = 0
    for det_name, fpath in det_files.items():
        with h5py.File(fpath, "r") as hf:
            n = hf["energies"].shape[0]
            n_total += n
            print(f"  {det_name}: {n} events")
    print(f"  Total: {n_total} events → {out_path.name}")

    with h5py.File(out_path, "w") as out:
        # Create datasets — layer_z_pos padded to MAX_LAYERS
        ds = {}
        for name, per_evt in FIXED_DATASETS.items():
            # Use float32 for most, int32 for integer fields
            dtype = np.int32 if name in ("num_points", "pdg", "shower_ids") else np.float32
            if per_evt is None:
                ds[name] = out.create_dataset(name, shape=(n_total,), dtype=dtype)
            else:
                ds[name] = out.create_dataset(name, shape=(n_total, *per_evt), dtype=dtype)

        ds["layer_z_pos"] = out.create_dataset(
            "layer_z_pos", shape=(n_total, MAX_LAYERS), dtype=np.float32
        )
        ds["showers"] = out.create_dataset(
            "showers", shape=(n_total,), dtype=vlen_dt
        )

        idx = 0
        for det_name, fpath in det_files.items():
            with h5py.File(fpath, "r") as hf:
                n = hf["energies"].shape[0]
                for name in FIXED_DATASETS:
                    ds[name][idx : idx + n] = hf[name][:]

                # Pad layer_z_pos
                lzp = hf["layer_z_pos"][:]
                lzp_width = lzp.shape[1]
                if lzp_width < MAX_LAYERS:
                    padded = np.zeros((n, MAX_LAYERS), dtype=np.float32)
                    padded[:, :lzp_width] = lzp
                    ds["layer_z_pos"][idx : idx + n] = padded
                else:
                    ds["layer_z_pos"][idx : idx + n] = lzp

                # vlen showers — batch read + write
                CHUNK = 10000
                for start in tqdm(range(0, n, CHUNK), desc=f"  {det_name} showers", unit="chunk"):
                    end = min(start + CHUNK, n)
                    batch = [hf["showers"][j] for j in range(start, end)]
                    ds["showers"][idx + start : idx + end] = batch

                idx += n

        # Sequential shower_ids
        ds["shower_ids"][:] = np.arange(n_total, dtype=np.int32)

        # shape metadata
        max_hits = int(np.max(ds["num_points"][:]))
        out.create_dataset("shape", data=np.array([n_total, max_hits, 5], dtype=np.int64))

        out.attrs["detector"] = "lemurs_combined"
        out.attrs["n_showers"] = n_total
        out.attrs["max_layers"] = MAX_LAYERS
        out.attrs["detectors"] = list(det_files.keys())

    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["merge-detector", "combine-all", "all"], required=True)
    parser.add_argument("--detector", default="all", help="Detector name or 'all'")
    parser.add_argument("--max-files", type=int, default=None, help="Limit files per detector (for testing)")
    parser.add_argument("--h5-subdirs", nargs="+", default=None,
                        help="H5 subdirectory names (default: h5_1M). Multiple dirs are merged together.")
    parser.add_argument("--output-name", default=None, help="Output filename (default: <det>_1M.h5)")
    args = parser.parse_args()

    if args.step in ("merge-detector", "all"):
        dets = list(DETECTORS.keys()) if args.detector == "all" else [args.detector]
        for det in dets:
            print(f"\n--- Merging {det} ---")
            merge_detector(det, max_files=args.max_files,
                           h5_subdirs=args.h5_subdirs, output_name=args.output_name)

    if args.step in ("combine-all", "all"):
        print("\n--- Combining all detectors ---")
        combine_all()


if __name__ == "__main__":
    main()
