#!/usr/bin/env python3
"""
Process ALLEGRO ECal step-level ROOT → showerdata-compatible HDF5.

Geometry : ALLEGRO_o1_v03  (LAr/Pb barrel accordion, 11 radial layers)
Data     : ECalBarrelModuleThetaMergedContributions (HitCreationMode=2)

Local frame:  Δh_x = hit_x − gun_x,  Δh_y = hit_y − gun_y
Projection:   1 mm×1 mm grid per layer (--cell-size, default 1.0)

HDF5 keys (showerdata-compatible, matches SimpleBox training format):
    showers           vlen float32  flat (n_hits×5): [Δh_x, Δh_y, layer, E_GeV, 0]
    energies          (N,1) float32   E_incident [GeV]
    directions        (N,3) float32   unit vector (sinθcosφ, sinθsinφ, cosθ)
    num_layers        (N,1) int32     distinct active layers hit
    num_points        (N,)  int32     hits/shower after projection
    pdg               (N,)  int32     22
    sampling_fraction (N,1) float32   E_dep/E_incident
    shape             (3,)  int64     [N, max_hits, 5]
    shower_ids        (N,)  int32     0…N-1
    gun_position      (N,2) float32   (gun_x, gun_y) [mm]

Usage:
    python process_root_to_h5_allegro.py \\
        --root-file  output_dataset/ALLEGRO/root/final/allegro_final_0000.root \\
        --output-dir output_dataset/ALLEGRO/h5/final/
"""

import argparse
import sys
from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import uproot
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils import calo_geometry
from utils.preprocessing_data import cluster_to_grid, get_incident_momentum, process_calo_hits

CONTRIB   = "ECalBarrelModuleThetaMergedContributions"
CP        = f"{CONTRIB}/{CONTRIB}"
E_THR_MEV  = 0.01
N_FEAT     = 5
PDG        = 22
CELL_MM    = 1.0
SF_NOMINAL = 0.162   # Fixed nominal sampling fraction for ALLEGRO_o1_v03 (calorimeter property)
DH_T_MAX   = 500.0   # mm — local frame acceptance window (tangential)
DH_Z_MAX   = 1000.0  # mm — local frame acceptance window (beam-axis)

# Cylindrical local frame:
#   dh_t = r_hit * wrap(φ_hit − φ_gun)   [mm]  tangential (≈ r·Δφ)
#   dh_z = z_hit − z_gun                  [mm]  beam-axis offset


def _wrap_angle(dphi):
    """Wrap angle difference to [-π, π]."""
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def process_allegro_file(root_file, output_dir, n_events=None, cell_size=CELL_MM):
    root_file  = Path(root_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_file = output_dir / root_file.with_suffix(".h5").name

    if h5_file.exists():
        print(f"Already exists: {h5_file}  (delete to reprocess)")
        return str(h5_file)
    if not root_file.exists():
        sys.exit(f"ROOT file not found: {root_file}")

    print(f"\n{'='*60}\n{root_file.name}\n{'='*60}")
    tree = uproot.open(root_file)["events"]
    geo  = calo_geometry.ALLEGROGeometry(version="v03")

    # ── MC truth ──────────────────────────────────────────────────────────────
    mc_raw = tree.arrays([
        "MCParticles/MCParticles.momentum.x",
        "MCParticles/MCParticles.momentum.y",
        "MCParticles/MCParticles.momentum.z",
        "MCParticles/MCParticles.generatorStatus",
        "MCParticles/MCParticles.PDG",
        "MCParticles/MCParticles.vertex.x",
        "MCParticles/MCParticles.vertex.y",
        "MCParticles/MCParticles.vertex.z",
    ], library="ak")

    energies_all, momenta_all = get_incident_momentum(mc_raw, verbose=True)
    dirs_all = (momenta_all / np.linalg.norm(momenta_all, axis=1, keepdims=True)).astype(np.float32)

    sel = mc_raw["MCParticles/MCParticles.generatorStatus"] == 1
    gx_all  = np.array(ak.flatten(mc_raw["MCParticles/MCParticles.vertex.x"][sel]), np.float32)
    gy_all  = np.array(ak.flatten(mc_raw["MCParticles/MCParticles.vertex.y"][sel]), np.float32)
    gz_all  = np.array(ak.flatten(mc_raw["MCParticles/MCParticles.vertex.z"][sel]), np.float32)
    gphi_all = np.arctan2(gy_all, gx_all)   # azimuthal angle of gun position

    # ── step-level contributions (all events, loaded once) ────────────────────
    contribs = tree.arrays([
        f"{CP}.energy",
        f"{CP}.stepPosition.x", f"{CP}.stepPosition.y",
        f"{CP}.stepPosition.z", f"{CP}.time",
    ], library="ak")

    if np.count_nonzero(ak.to_numpy(contribs[f"{CP}.stepPosition.x"][0])) == 0:
        sys.exit("ERROR: stepPosition.x all-zero — HitCreationMode=2 not set!")

    # ── per-event loop ────────────────────────────────────────────────────────
    n_total = min(len(energies_all), n_events) if n_events else len(energies_all)
    showers, valid = [], []

    for i in tqdm(range(n_total)):
        # 1. load all steps (no threshold — preserve full E_dep for correct SF)
        data = process_calo_hits(contribs, event_index=i,
                                 contrib_prefix=CP,
                                 energy_threshold_mev=0.0)
        if len(data["energy"]) == 0:
            continue

        # 2. barrel filter: keep only hits inside r_min ≤ rho ≤ r_max
        x_g, y_g, z_g = data["x"], data["y"], data["z"]
        rho = np.sqrt(x_g**2 + y_g**2)
        barrel = (rho >= geo.r_min) & (rho <= geo.r_max)
        n_before = len(rho)
        x_g, y_g, z_g = x_g[barrel], y_g[barrel], z_g[barrel]
        rho, t_g = rho[barrel], data["time"][barrel]
        e_g = data["energy"][barrel]
        if len(e_g) == 0:
            continue

        # 3. layer assignment from rho
        layer = geo.rho_to_layer(rho).astype(np.int32)

        # 4. cylindrical local frame (BEFORE clustering)
        dphi = _wrap_angle(np.arctan2(y_g, x_g) - gphi_all[i])
        dh_t = (rho * dphi).astype(np.float32)       # tangential [mm]
        dh_z = (z_g - gz_all[i]).astype(np.float32)  # beam-axis  [mm]

        # 4b. local frame acceptance window — removes scattered secondaries
        win = (np.abs(dh_t) < DH_T_MAX) & (np.abs(dh_z) < DH_Z_MAX)
        dh_t, dh_z = dh_t[win], dh_z[win]
        rho, e_g, layer, t_g = rho[win], e_g[win], layer[win], t_g[win]
        if len(e_g) == 0:
            continue

        # 5. 1 mm × 1 mm grid clustering in local frame (dh_t, dh_z per layer)
        if cell_size > 0:
            local_data = {"x": dh_t, "y": dh_z, "z": rho.astype(np.float32),
                          "energy": e_g.astype(np.float32),
                          "layer": layer, "time": t_g.astype(np.float32)}
            clustered = cluster_to_grid(local_data, cell_size_mm=cell_size,
                                        geometry=geo, verbose=False)
            dh_t, dh_z = clustered["x"], clustered["y"]
            e_g, layer  = clustered["energy"], clustered["layer"]

        # 6. energy threshold on clustered cells [GeV]
        keep = e_g > (E_THR_MEV * 1e-3)
        dh_t, dh_z = dh_t[keep], dh_z[keep]
        e_g, layer  = e_g[keep], layer[keep]

        if len(e_g) == 0:
            continue

        if i < 3:  # verbose for first 3 events
            print(f"\n  evt {i}: {n_before} steps → {len(rho)} in barrel+window"
                  f" → {len(e_g)} cells after {cell_size}mm grid + thr"
                  f"  E_dep={e_g.sum()*1e3:.1f} MeV")

        pts = np.stack([dh_t, dh_z, layer.astype(np.float32),
                        e_g, np.zeros_like(e_g)], axis=1)  # (n, 5)
        showers.append(pts.ravel())
        valid.append(i)

    if not showers:
        raise RuntimeError("No valid events after processing!")

    # ── aggregate statistics ──────────────────────────────────────────────────
    n     = len(showers)
    nhits = np.array([len(s) // N_FEAT for s in showers], dtype=np.int32)
    e_inc = energies_all[valid].astype(np.float32)
    e_dep = np.array([s.reshape(-1, N_FEAT)[:, 3].sum() for s in showers], np.float32)
    sf    = np.full(n, SF_NOMINAL, dtype=np.float32)
    nl    = np.array([len(np.unique(s.reshape(-1, N_FEAT)[:, 2].astype(int)))
                      for s in showers], np.int32)

    print(f"\nValid: {n}/{n_total}  hits [{nhits.min()}, {nhits.max()}] mean={nhits.mean():.0f}")
    print(f"SF: {sf.mean():.4f} ± {sf.std():.4f}   n_layers: {nl.mean():.1f}")

    # ── layer_z_pos: radial depth of each layer from calorimeter surface ───────
    # layer_z_pos[l] = geo.layer_boundaries[l] - geo.r_min  (= 0 for l=0)
    # Padded to 45 for compatibility with SimpleBox shift_layers infrastructure.
    # Use LOCAL direction (0, cos θ, sin θ) when calling shift_layers on ALLEGRO data.
    _lzp = np.zeros(45, dtype=np.float32)
    _lzp[:geo.num_layers] = (geo.layer_boundaries[:geo.num_layers] - geo.r_min).astype(np.float32)
    layer_z_pos = np.tile(_lzp, (n, 1))  # (N, 45) — same row for all (fixed geometry)

    # ── save HDF5 ─────────────────────────────────────────────────────────────
    print(f"\nSaving → {h5_file}")
    with h5py.File(h5_file, "w") as hf:
        ds = hf.create_dataset("showers", shape=(n,), dtype=h5py.vlen_dtype(np.float32))
        for j, arr in enumerate(showers):
            ds[j] = arr

        hf.create_dataset("energies",          data=e_inc.reshape(-1, 1))
        hf.create_dataset("directions",        data=dirs_all[valid])
        hf.create_dataset("num_layers",        data=nl.reshape(-1, 1))
        hf.create_dataset("num_points",        data=nhits)
        hf.create_dataset("pdg",               data=np.full(n, PDG, dtype=np.int32))
        hf.create_dataset("sampling_fraction", data=sf.reshape(-1, 1))
        hf.create_dataset("shower_ids",        data=np.arange(n, dtype=np.int32))
        hf.create_dataset("gun_position",      data=np.stack(
                              [gx_all[valid], gy_all[valid], gz_all[valid]], axis=1))
        hf.create_dataset("shape",             data=np.array([n, int(nhits.max()), N_FEAT], np.int64))
        hf.create_dataset("layer_z_pos",       data=layer_z_pos)

        hf.attrs.update({
            "detector":             "ALLEGRO_o1_v03",
            "features":             "x, y, layer, energy_GeV, 0",
            "local_frame":          "cylindrical: x=r*wrap(phi_hit-phi_gun), y=z_hit-z_gun",
            "shift_layers_note":    "use local_dir=(0, dz, sqrt(dx2+dy2)) with layer_z_pos",
            "r_min_mm":             geo.r_min,
            "r_max_mm":             geo.r_max,
            "num_layers":           geo.num_layers,
            "energy_threshold_mev": E_THR_MEV,
            "cell_size_mm":         cell_size,
        })

    print(f"Done: {h5_file}")
    return str(h5_file)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-file",   required=True)
    ap.add_argument("--output-dir",  required=True)
    ap.add_argument("--n-events",    type=int,   default=None)
    ap.add_argument("--cell-size",   type=float, default=CELL_MM,
                    help="Grid cell size [mm] for x-y projection (0 = no clustering)")
    args = ap.parse_args()
    process_allegro_file(args.root_file, args.output_dir, args.n_events, args.cell_size)
