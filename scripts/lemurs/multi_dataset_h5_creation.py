#!/usr/bin/env python3
"""
Merge individual HDF5 files (vlen showerdata format) into one dataset.

Works for any detector/version — input and output dirs passed via CLI.

Usage:
    python multi_dataset_h5_creation.py --input-dir output_dataset/ALLEGRO/h5/final --output output_dataset/ALLEGRO/h5/allegro_1M.h5
"""

import argparse
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm


VLEN_F32 = h5py.special_dtype(vlen=np.dtype("float32"))

# Fixed-shape datasets and their per-event trailing dimensions (None = scalar 1D)
FIXED_DATASETS = {
    "energies":          (1,),
    "directions":        (3,),
    "num_points":        None,
    "num_layers":        (1,),
    "sampling_fraction": (1,),
    "pdg":               None,
    "shower_ids":        None,
}


def merge(input_dir, output_path):
    input_dir   = Path(input_dir)
    output_path = Path(output_path)

    h5_files = sorted(input_dir.glob("*.h5"))
    assert h5_files, f"No HDF5 files in {input_dir}"

    # --- pre-scan ---
    meta = []
    for p in tqdm(h5_files, desc="Scanning"):
        try:
            with h5py.File(p, "r") as f:
                meta.append({"path": p, "n": len(f["num_points"])})
        except Exception as e:
            print(f"  skip {p.name}: {e}")

    total = sum(m["n"] for m in meta)
    print(f"{len(meta)} files | {total:,} showers total")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- discover optional datasets from first file ---
    with h5py.File(meta[0]["path"], "r") as f0:
        lzp_width = f0["layer_z_pos"].shape[1]
        has_gun   = "gun_position" in f0
        gun_cols  = f0["gun_position"].shape[1] if has_gun else None
        ex_attrs  = dict(f0.attrs)

    with h5py.File(output_path, "w") as fout:
        # create fixed-shape datasets
        ds = {}
        for name, per_evt in FIXED_DATASETS.items():
            with h5py.File(meta[0]["path"], "r") as ref:
                dtype = ref[name].dtype
            if per_evt is None:
                ds[name] = fout.create_dataset(name, shape=(total,), dtype=dtype)
            else:
                ds[name] = fout.create_dataset(name, shape=(total, *per_evt), dtype=dtype)

        ds["showers"]     = fout.create_dataset("showers",     (total,),            dtype=VLEN_F32)
        ds["layer_z_pos"] = fout.create_dataset("layer_z_pos", (total, lzp_width),  dtype=np.float32)
        if has_gun:
            ds["gun_position"] = fout.create_dataset("gun_position", (total, gun_cols), dtype=np.float32)

        # fill
        idx = 0
        for m in tqdm(meta, desc="Merging"):
            with h5py.File(m["path"], "r") as f:
                n = m["n"]
                for name in FIXED_DATASETS:
                    if name == "shower_ids":
                        ds[name][idx:idx+n] = np.arange(idx, idx+n, dtype=np.int32)
                    elif name == "pdg":
                        ds[name][idx:idx+n] = f[name][:] if name in f else np.full(n, 22, np.int32)
                    else:
                        ds[name][idx:idx+n] = f[name][:]

                ds["showers"][idx:idx+n]     = f["showers"][:]
                ds["layer_z_pos"][idx:idx+n] = f["layer_z_pos"][:]
                if has_gun:
                    ds["gun_position"][idx:idx+n] = f["gun_position"][:]
            idx += n

        fout.attrs.update(ex_attrs)
        fout.attrs.update({"n_showers": total, "n_files": len(meta)})
        fout.create_dataset("shape", data=np.array(
            [total, int(ds["num_points"][:].max()), 5], dtype=np.int64))

    size_gb = output_path.stat().st_size / 1e9
    print(f"Saved: {output_path}  ({size_gb:.2f} GB)")

    with h5py.File(output_path, "r") as f:
        for k in sorted(f.keys()):
            print(f"  {k}: {f[k].shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge individual H5 files into one dataset.")
    ap.add_argument("--input-dir", required=True, help="Directory with individual H5 files")
    ap.add_argument("--output",    required=True, help="Output merged H5 file path")
    args = ap.parse_args()
    merge(args.input_dir, args.output)
