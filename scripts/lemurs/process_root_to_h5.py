#!/usr/bin/env python3
"""
Process LEMURS ECal step-level ROOT → showerdata-compatible HDF5.

Generalized version of process_root_to_h5_allegro.py for all 5 LEMURS
barrel calorimeters. Uses LEMURSBarrelGeometry for layer assignment.

Local frame (cylindrical):
    dh_t = r_hit * wrap(phi_hit - phi_gun)   [mm]  tangential
    dh_z = z_hit - z_gun                     [mm]  beam-axis offset

HDF5 keys (showerdata-compatible):
    showers           vlen float32  flat (n_hits×5): [dh_t, dh_z, layer, E_GeV, 0]
    energies          (N,1) float32   E_incident [GeV]
    directions        (N,3) float32   unit vector (sinθcosφ, sinθsinφ, cosθ)
    num_layers        (N,1) int32     distinct active layers hit
    num_points        (N,)  int32     hits/shower after projection
    pdg               (N,)  int32     22
    sampling_fraction (N,1) float32   E_dep/E_incident
    shape             (3,)  int64     [N, max_hits, 5]
    shower_ids        (N,)  int32     0…N-1
    gun_position      (N,3) float32   (gun_x, gun_y, gun_z) [mm]

Usage:
    python process_root_to_h5.py \\
        --detector par04_siw \\
        --root-file output_dataset/Par04_SiW/root/par04_siw_0000.root \\
        --output-dir output_dataset/Par04_SiW/h5/
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
from utils.calo_geometry import LEMURSBarrelGeometry

E_THR_MEV  = 0.01
N_FEAT     = 5
PDG        = 22
CELL_MM    = 1.0
DH_T_MAX   = 500.0   # mm — local frame acceptance window (tangential)
DH_Z_MAX   = 1000.0  # mm — local frame acceptance window (beam-axis)


def _wrap_angle(dphi):
    """Wrap angle difference to [-π, π]."""
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def process_lemurs_file(detector, root_file, output_dir, n_events=None, cell_size=CELL_MM):
    geo = LEMURSBarrelGeometry(detector)
    cp = geo.contrib_collection + "/" + geo.contrib_collection

    root_file  = Path(root_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_file = output_dir / root_file.with_suffix(".h5").name

    if h5_file.exists():
        print(f"Already exists: {h5_file}  (delete to reprocess)")
        return str(h5_file)
    if not root_file.exists():
        sys.exit(f"ROOT file not found: {root_file}")

    print(f"\n{'='*60}\n{detector}: {root_file.name}\n{'='*60}")
    print(repr(geo))
    tree = uproot.open(root_file)["events"]

    # ── MC truth ──────────────────────────────────────────────────────────────
    mc_raw = tree.arrays([
        "MCParticles/MCParticles.momentum.x",
        "MCParticles/MCParticles.momentum.y",
        "MCParticles/MCParticles.momentum.z",
        "MCParticles/MCParticles.generatorStatus",
        "MCParticles/MCParticles.vertex.x",
        "MCParticles/MCParticles.vertex.y",
        "MCParticles/MCParticles.vertex.z",
    ], library="ak")

    sel = mc_raw["MCParticles/MCParticles.generatorStatus"] == 1
    px_all = ak.firsts(mc_raw["MCParticles/MCParticles.momentum.x"][sel]).to_numpy().astype(np.float32)
    py_all = ak.firsts(mc_raw["MCParticles/MCParticles.momentum.y"][sel]).to_numpy().astype(np.float32)
    pz_all = ak.firsts(mc_raw["MCParticles/MCParticles.momentum.z"][sel]).to_numpy().astype(np.float32)
    energies_all = np.sqrt(px_all**2 + py_all**2 + pz_all**2).astype(np.float32)
    momenta_all = np.stack([px_all, py_all, pz_all], axis=1)
    dirs_all = (momenta_all / np.linalg.norm(momenta_all, axis=1, keepdims=True)).astype(np.float32)

    gx_all = ak.firsts(mc_raw["MCParticles/MCParticles.vertex.x"][sel]).to_numpy().astype(np.float32)
    gy_all = ak.firsts(mc_raw["MCParticles/MCParticles.vertex.y"][sel]).to_numpy().astype(np.float32)
    gz_all = ak.firsts(mc_raw["MCParticles/MCParticles.vertex.z"][sel]).to_numpy().astype(np.float32)
    gphi_all = np.arctan2(gy_all, gx_all)

    print(f"Events in file: {len(energies_all)}")
    print(f"E_inc range: [{energies_all.min():.2f}, {energies_all.max():.2f}] GeV")

    # ── step-level contributions ─────────────────────────────────────────────
    contribs = tree.arrays([
        f"{cp}.energy",
        f"{cp}.stepPosition.x", f"{cp}.stepPosition.y",
        f"{cp}.stepPosition.z",
    ], library="ak")

    # Verify step positions are non-zero
    first_x = ak.to_numpy(contribs[f"{cp}.stepPosition.x"][0])
    if np.count_nonzero(first_x) == 0:
        sys.exit("ERROR: stepPosition.x all-zero — HitCreationMode=2 not set!")

    # ── per-event loop ───────────────────────────────────────────────────────
    n_total = min(len(energies_all), n_events) if n_events else len(energies_all)
    showers, valid = [], []

    for i in tqdm(range(n_total), desc=detector):
        e_g = ak.to_numpy(contribs[f"{cp}.energy"][i]).astype(np.float32)
        x_g = ak.to_numpy(contribs[f"{cp}.stepPosition.x"][i]).astype(np.float32)
        y_g = ak.to_numpy(contribs[f"{cp}.stepPosition.y"][i]).astype(np.float32)
        z_g = ak.to_numpy(contribs[f"{cp}.stepPosition.z"][i]).astype(np.float32)

        if len(e_g) == 0:
            continue

        # barrel filter
        rho = np.sqrt(x_g**2 + y_g**2)
        barrel = (rho >= geo.r_min) & (rho <= geo.r_max)
        x_g, y_g, z_g = x_g[barrel], y_g[barrel], z_g[barrel]
        rho, e_g = rho[barrel], e_g[barrel]
        if len(e_g) == 0:
            continue

        # layer assignment
        layer = geo.rho_to_layer(rho).astype(np.int32)

        # cylindrical local frame
        dphi = _wrap_angle(np.arctan2(y_g, x_g) - gphi_all[i])
        dh_t = (rho * dphi).astype(np.float32)
        dh_z = (z_g - gz_all[i]).astype(np.float32)

        # acceptance window
        win = (np.abs(dh_t) < DH_T_MAX) & (np.abs(dh_z) < DH_Z_MAX)
        dh_t, dh_z = dh_t[win], dh_z[win]
        e_g, layer = e_g[win], layer[win]
        if len(e_g) == 0:
            continue

        # grid clustering (optional)
        if cell_size > 0:
            keys = {}
            for li in np.unique(layer):
                m = layer == li
                ix = (dh_t[m] / cell_size).astype(np.int32)
                iy = (dh_z[m] / cell_size).astype(np.int32)
                for jj in range(m.sum()):
                    k = (int(li), int(ix[jj]), int(iy[jj]))
                    keys[k] = keys.get(k, 0.0) + float(e_g[m][jj])
            if not keys:
                continue
            arr = np.array([(ix * cell_size + cell_size / 2, iy * cell_size + cell_size / 2, li, en, 0.0)
                            for (li, ix, iy), en in keys.items()], dtype=np.float32)
            dh_t = arr[:, 0]
            dh_z = arr[:, 1]
            layer = arr[:, 2].astype(np.int32)
            e_g = arr[:, 3]

        # energy threshold
        keep = e_g > (E_THR_MEV * 1e-3)
        dh_t, dh_z = dh_t[keep], dh_z[keep]
        e_g, layer = e_g[keep], layer[keep]
        if len(e_g) == 0:
            continue

        if i < 3:
            print(f"\n  evt {i}: {len(e_g)} cells, E_dep={e_g.sum()*1e3:.1f} MeV, "
                  f"layers=[{layer.min()},{layer.max()}]")

        pts = np.stack([dh_t, dh_z, layer.astype(np.float32),
                        e_g, np.zeros_like(e_g)], axis=1)
        showers.append(pts.ravel())
        valid.append(i)

    if not showers:
        raise RuntimeError("No valid events after processing!")

    # ── aggregate ────────────────────────────────────────────────────────────
    n = len(showers)
    nhits = np.array([len(s) // N_FEAT for s in showers], dtype=np.int32)
    e_inc = energies_all[valid].astype(np.float32)
    e_dep = np.array([s.reshape(-1, N_FEAT)[:, 3].sum() for s in showers], np.float32)
    # Fixed nominal SF per calorimeter geometry (not per-event E_dep/E_inc)
    sf = np.full(n, geo.nominal_sf, dtype=np.float32)
    # Fixed NL = total layers for this detector geometry
    nl = np.full(n, geo.num_layers, dtype=np.int32)

    sf_per_event = (e_dep / e_inc).astype(np.float32)
    nl_per_event = np.array([len(np.unique(s.reshape(-1, N_FEAT)[:, 2].astype(int)))
                              for s in showers], np.int32)
    print(f"\nValid: {n}/{n_total}  hits [{nhits.min()}, {nhits.max()}] mean={nhits.mean():.0f}")
    print(f"SF nominal: {geo.nominal_sf:.4f}  (per-event mean: {sf_per_event.mean():.4f} +/- {sf_per_event.std():.4f})")
    print(f"NL nominal: {geo.num_layers}  (per-event mean: {nl_per_event.mean():.1f})")

    # ── layer_z_pos: radial depth from calorimeter surface ───────────────────
    max_layers = max(geo.num_layers, 45)  # pad to at least 45 for compatibility
    _lzp = np.zeros(max_layers, dtype=np.float32)
    _lzp[:geo.num_layers] = (geo.layer_boundaries[:geo.num_layers] - geo.r_min).astype(np.float32)
    layer_z_pos = np.tile(_lzp, (n, 1))

    # ── save HDF5 ────────────────────────────────────────────────────────────
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
            "detector":             detector,
            "features":             "dh_t, dh_z, layer, energy_GeV, 0",
            "local_frame":          "cylindrical: dh_t=r*wrap(phi_hit-phi_gun), dh_z=z_hit-z_gun",
            "r_min_mm":             float(geo.r_min),
            "r_max_mm":             float(geo.r_max),
            "num_layers":           int(geo.num_layers),
            "layer_thickness_mm":   float(geo.layer_thickness),
            "energy_threshold_mev": E_THR_MEV,
            "cell_size_mm":         cell_size,
        })

    print(f"Done: {h5_file}")
    return str(h5_file)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector",    required=True,
                    choices=list(LEMURSBarrelGeometry.DETECTORS.keys()))
    ap.add_argument("--root-file",   required=True)
    ap.add_argument("--output-dir",  required=True)
    ap.add_argument("--n-events",    type=int,   default=None)
    ap.add_argument("--cell-size",   type=float, default=CELL_MM,
                    help="Grid cell size [mm] (0 = no clustering)")
    args = ap.parse_args()
    process_lemurs_file(args.detector, args.root_file, args.output_dir,
                        args.n_events, args.cell_size)