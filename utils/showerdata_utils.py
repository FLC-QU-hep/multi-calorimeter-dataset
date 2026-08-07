"""
Convert H5 datasets to showerdata training-ready format.

showerdata: https://github.com/FLC-QU-hep/ShowerData
(pip install git+https://github.com/FLC-QU-hep/ShowerData)

Requires the AllShowers-AllGeometries venv (has showerdata):
    source "$WORK_DIR/AllShowers-AllGeometries/.venv/bin/activate"
    # $WORK_DIR is the workspace parent of the repo root (dirname of REPO_ROOT)
"""

import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

try:
    import showerdata as _showerdata
    from showerdata.shift_showers import shift_layers as _shift_layers
    _HAS_SHOWERDATA = True
except ImportError:
    _HAS_SHOWERDATA = False


def convert_to_showerdata(src_h5: str, dst_h5: str, pdg: int = 22) -> None:
    """
    Convert an intermediate unified H5 file to showerdata training-ready format.

    Handles both SimpleBox (incident_energies/momentum/sampling_fraction/num_layers/
    layer_z_pos) and ALLEGRO (incident_energy/incident_momentum) field conventions.
    Applies column reorder [x,y,z,time,E]->[x,y,z,E,time], momentum normalisation,
    and shift_layers(inverse=False) when layer_z_pos is present.
    """
    if not _HAS_SHOWERDATA:
        print("WARNING: showerdata not found — skipping convert_to_showerdata. Activate AllShowers venv.")
        return

    print(f"\nCONVERT TO SHOWERDATA FORMAT\n  src: {src_h5}\n  dst: {dst_h5}\n")

    with h5py.File(src_h5, "r") as f:
        # Column reorder: [x, y, z, time, E] -> [x, y, z, E, time]
        events = f["events"][:][:, :, [0, 1, 2, 4, 3]]

        # Support both naming conventions (SimpleBox vs ALLEGRO)
        if "incident_energies" in f:
            energies = f["incident_energies"][:].flatten().astype(np.float32)
        else:
            energies = f["incident_energy"][:].flatten().astype(np.float32)

        if "momentum" in f:
            momentum = f["momentum"][:].astype(np.float32)
        else:
            momentum = f["incident_momentum"][:].astype(np.float32)

        sampling_fraction = f["sampling_fraction"][:] if "sampling_fraction" in f else None
        num_layers        = f["num_layers"][:]        if "num_layers"        in f else None
        layer_z_pos       = f["layer_z_pos"][:]       if "layer_z_pos"       in f else None

    # Normalise momentum -> unit direction vectors
    norm = np.linalg.norm(momentum, axis=1, keepdims=True)
    directions = np.divide(momentum, norm, out=np.zeros_like(momentum), where=norm != 0)

    # Apply shift_layers (straighten showers to z-axis for training)
    if layer_z_pos is not None:
        print(f"  Applying shift_layers(inverse=False) on {len(events):,} showers...")
        for i in tqdm(range(len(events)), desc="  shift_layers"):
            events[i] = _shift_layers(
                shower=events[i],
                direction=directions[i],
                layer_bottom_pos=layer_z_pos[i],
                calo_surface=0.0,
                inverse=False,
            )
    else:
        print("  WARNING: layer_z_pos not found — skipping shift_layers.")

    # Save as showerdata format
    showers = _showerdata.Showers(
        points=events,
        pdg=np.full(len(energies), pdg, dtype=np.int32),
        energies=energies,
        directions=directions,
    )
    dst_path = Path(dst_h5)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    _showerdata.save(showers, str(dst_path), overwrite=True)

    # Append conditioning fields that showerdata.save() doesn't handle
    with h5py.File(str(dst_path), "a") as f:
        if sampling_fraction is not None:
            f.create_dataset("sampling_fraction", data=sampling_fraction)
        if num_layers is not None:
            f.create_dataset("num_layers", data=num_layers)

    size_gb = dst_path.stat().st_size / 1e9
    print(f"\n  Saved {len(energies):,} showers -> {dst_path}  ({size_gb:.2f} GB)")


def combine_temp_to_showerdata(
    h5_temp_dir: str,
    dst_h5: str,
    max_num_layers: int = 45,
    pdg: int = 22,
) -> None:
    """
    Combine per-config temp H5 files directly into showerdata training-ready format.

    Reads config_*.h5 files from h5_temp_dir (output of process_individual_configs).
    Writes in streaming fashion (one config at a time) to avoid OOM — no large
    upfront array allocations. Uses variable-length HDF5 datasets for showers.
    """
    if not _HAS_SHOWERDATA:
        print("WARNING: showerdata not found — skipping combine_temp_to_showerdata. Activate AllShowers venv.")
        return

    h5_temp_dir = Path(h5_temp_dir)
    dst_path = Path(dst_h5)

    h5_files = sorted(h5_temp_dir.glob("config_*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No config_*.h5 files found in {h5_temp_dir}")

    print(f"\nCOMBINE TEMP H5 -> SHOWERDATA FORMAT\n  src: {h5_temp_dir}  ({len(h5_files)} configs)\n  dst: {dst_h5}\n")

    # Pre-scan: count total showers and find global max_points
    total_showers, max_points_global, valid_files = 0, 0, []
    for f_path in tqdm(h5_files, desc="  scanning"):
        try:
            with h5py.File(f_path, "r") as f:
                n, pts, _ = f["events"].shape
            total_showers += n
            max_points_global = max(max_points_global, pts)
            valid_files.append(f_path)
        except Exception as e:
            print(f"  WARNING: skipping {f_path.name}: {e}")

    print(f"  {total_showers:,} showers | max_points={max_points_global} | {len(valid_files)} valid configs")

    # Create output HDF5 with pre-allocated fixed datasets + variable-length showers.
    # No large upfront numpy arrays — write one config at a time.
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dst_path, "w") as fout:
        ds_energies  = fout.create_dataset("energies",          shape=(total_showers, 1), dtype=np.float32)
        ds_dirs      = fout.create_dataset("directions",        shape=(total_showers, 3), dtype=np.float32)
        ds_pdg       = fout.create_dataset("pdg",               data=np.full(total_showers, pdg, dtype=np.int32))
        ds_ids       = fout.create_dataset("shower_ids",        data=np.arange(total_showers, dtype=np.int32))
        ds_npts      = fout.create_dataset("num_points",        shape=(total_showers,),   dtype=np.int32)
        ds_sf        = fout.create_dataset("sampling_fraction", shape=(total_showers, 1), dtype=np.float32)
        ds_nl        = fout.create_dataset("num_layers",        shape=(total_showers, 1), dtype=np.int32)
        fout.create_dataset("shape", data=np.array([total_showers, max_points_global, 5], dtype=np.int64))
        ds_showers   = fout.create_dataset("showers", shape=(total_showers,), dtype=h5py.vlen_dtype(np.float32))

        idx = 0
        for f_path in tqdm(valid_files, desc="  processing"):
            with h5py.File(f_path, "r") as f:
                ev  = f["events"][:].astype(np.float32)       # [x,y,z,time,E]
                en  = f["incident_energy"][:].flatten().astype(np.float32)
                mom = f["incident_momentum"][:].astype(np.float32)
                sf  = f["sampling_fraction"][:].flatten().astype(np.float32)
                nl  = int(f.attrs["geometry_num_layers"])
                lt  = float(f.attrs["geometry_layer_thickness"])

            n = len(en)

            # Column reorder: [x,y,z,time,E] -> [x,y,z,E,time]
            ev = ev[:, :, [0, 1, 2, 4, 3]]

            # Normalize momentum -> unit directions
            norm = np.linalg.norm(mom, axis=1, keepdims=True)
            dirs = np.divide(mom, norm, out=np.zeros_like(mom), where=norm != 0)

            # Build layer_z_pos for this config
            layer_z_row = np.zeros(max_num_layers, dtype=np.float32)
            layer_z_row[:nl] = np.arange(nl, dtype=np.float32) * lt

            # Apply shift_layers(inverse=False) per shower, count non-zero hits
            num_pts = np.zeros(n, dtype=np.int32)
            vlen_batch = np.empty(n, dtype=object)
            for i in range(n):
                ev[i] = _shift_layers(
                    shower=ev[i],
                    direction=dirs[i],
                    layer_bottom_pos=layer_z_row,
                    calo_surface=0.0,
                    inverse=False,
                )
                k = int(np.any(ev[i] != 0, axis=-1).sum())
                num_pts[i] = k
                vlen_batch[i] = ev[i, :k, :].flatten()

            # Write batch to HDF5 (no large RAM allocation)
            ds_energies[idx:idx+n, 0] = en
            ds_dirs[idx:idx+n]        = dirs
            ds_sf[idx:idx+n, 0]       = sf
            ds_nl[idx:idx+n, 0]       = nl
            ds_npts[idx:idx+n]        = num_pts
            ds_showers[idx:idx+n]     = vlen_batch
            idx += n

    print(f"  Saved {total_showers:,} showers -> {dst_path}  ({dst_path.stat().st_size / 1e9:.2f} GB)")


def convert_allegro_to_showerdata(src_h5: str, dst_h5: str, pdg: int = 22) -> None:
    """
    Apply shift_layers to an ALLEGRO vlen HDF5 and write a training-ready copy.

    ALLEGRO uses a cylindrical local frame: x = tangential, y = beam-axis.
    The angular drift (cot θ · Δr per layer) shows only in y, so shift_layers
    must use the LOCAL direction (0, cos θ, sin θ) with radial depths as layer_z_pos:

        local_dir = (0, dz, sqrt(dx²+dy²)) = (0, cos θ, sin θ)
        y -= local_dir[1]/local_dir[2] * layer_z_pos[layer]
          = (cos θ / sin θ) * Δr_l   ← removes the beam-axis angular drift

    This is the exact analogue of SimpleBox's shift_layers(inverse=False).
    At inference, apply shift_layers(inverse=True) with the same local direction
    to restore the physical drift.
    """
    if not _HAS_SHOWERDATA:
        print("WARNING: showerdata not found — skipping. Activate AllShowers venv.")
        return

    print(f"\nCONVERT ALLEGRO -> SHOWERDATA FORMAT\n  src: {src_h5}\n  dst: {dst_h5}\n")

    with h5py.File(src_h5, "r") as f:
        raw_showers = f["showers"][:]
        nhits       = f["num_points"][:]
        e_inc       = f["energies"][:].ravel().astype(np.float32)
        dirs_global = f["directions"][:].astype(np.float32)        # (N,3) global
        sf          = f["sampling_fraction"][:].ravel() if "sampling_fraction" in f else None
        nl          = f["num_layers"][:].ravel().astype(np.int32)  if "num_layers"  in f else None
        layer_z_pos = f["layer_z_pos"][:].astype(np.float32)       if "layer_z_pos" in f else None
        attrs       = dict(f.attrs)
        n_feat      = int(attrs.get("shape", [0, 0, 5])[2]) if "shape" in attrs else 5

    n = len(nhits)

    # Local direction for ALLEGRO cylindrical geometry:
    #   depth axis   = radial  = sin θ = sqrt(dx²+dy²) = dirs_global[:,0:2] norm
    #   lateral axis = beam-z  = cos θ = dz             = dirs_global[:,2]
    #   tangential   = ê_t    = 0      (gun always shoots radially)
    sin_theta  = np.sqrt(dirs_global[:, 0]**2 + dirs_global[:, 1]**2).astype(np.float32)
    cos_theta  = dirs_global[:, 2].astype(np.float32)
    local_dirs = np.stack([np.zeros(n, np.float32), cos_theta, sin_theta], axis=1)

    # Apply shift_layers per shower (vlen — no need to reconstruct dense array)
    print(f"  Applying shift_layers(inverse=False) on {n:,} showers...")
    shifted = np.empty(n, dtype=object)
    for i in tqdm(range(n), desc="  shift_layers"):
        k   = nhits[i]
        pts = raw_showers[i].reshape(k, n_feat)            # (k, n_feat)
        if layer_z_pos is not None:
            pts = _shift_layers(
                shower=pts,
                direction=local_dirs[i],
                layer_bottom_pos=layer_z_pos[i],
                calo_surface=0.0,
                inverse=False,
            )
        shifted[i] = pts.ravel()

    dst_path = Path(dst_h5)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Writing {dst_path} ...")
    with h5py.File(dst_path, "w") as hf:
        ds = hf.create_dataset("showers", shape=(n,), dtype=h5py.vlen_dtype(np.float32))
        for j, arr in enumerate(shifted):
            ds[j] = arr

        hf.create_dataset("energies",          data=e_inc.reshape(-1, 1))
        hf.create_dataset("directions",        data=dirs_global)   # global — used for conditioning
        hf.create_dataset("num_points",        data=nhits)
        hf.create_dataset("shower_ids",        data=np.arange(n, dtype=np.int32))
        hf.create_dataset("shape",             data=np.array([n, int(nhits.max()), n_feat], np.int64))
        if sf is not None:
            hf.create_dataset("sampling_fraction", data=sf.reshape(-1, 1))
        if nl is not None:
            hf.create_dataset("num_layers",        data=nl.reshape(-1, 1))
        if layer_z_pos is not None:
            # Store the single geometry row for inverse shift at inference
            hf.create_dataset("layer_z_pos", data=layer_z_pos[0:1])

        hf.attrs.update(attrs)
        hf.attrs["shift_layers_applied"] = True
        hf.attrs["local_direction"]      = "(0, cos_theta, sin_theta)"

    size_gb = dst_path.stat().st_size / 1e9
    print(f"\n  Saved {n:,} showers -> {dst_path}  ({size_gb:.2f} GB)")
