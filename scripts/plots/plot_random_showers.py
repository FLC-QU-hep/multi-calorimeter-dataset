#!/usr/bin/env python3
"""
Plot showers in a 3x4 grid with different sampling fractions and layer counts.

Layout:
  Column 1: Fixed SF ≈ 0.030, varying layers (15, 25, 35, 45)
  Column 2: Fixed layers ≈ 35, varying SF (0.015, 0.025, 0.035, 0.045)
  Column 3: Random selection

Each plot shows y vs layer_number, with point size proportional to energy.
Z-axis is converted to layer number and ranges from 0 to 45 for all plots.

Usage:
    python scripts/plot_random_showers.py --ref-dir sf_nlayers_angles
"""

import argparse
import os
import numpy as np
import h5py
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[2])))


def find_shower_by_criteria(sf_array, nl_array, energy_array,
                              target_sf=None, target_nl=None,
                              target_energy=None, energy_tol=0.1,
                              sf_tol=0.001, nl_tol=1, rng=None):
    """Find a shower index matching the criteria."""
    mask = np.ones(len(sf_array), dtype=bool)

    if target_sf is not None:
        mask &= np.abs(sf_array - target_sf) < sf_tol
    if target_nl is not None:
        mask &= np.abs(nl_array - target_nl) < nl_tol
    if target_energy is not None:
        mask &= np.abs(energy_array - target_energy) < energy_tol

    candidates = np.where(mask)[0]
    if len(candidates) == 0:
        return None

    if rng is not None:
        return rng.choice(candidates)
    return candidates[0]


def plot_grid_preloaded(events_cache, idx_to_pos,
                        incident_energies, momentum,
                        sampling_fractions, num_layers,
                        view_selections, view='xy'):
    """
    Create a 4x3 grid of shower plots using preloaded data.

    Args:
        events_cache: Preloaded events array (only needed events)
        idx_to_pos: Dict mapping original event index to position in events_cache
        incident_energies: Full incident energies array
        momentum: Full momentum array
        sampling_fractions: Full sampling fractions array
        num_layers: Full num_layers array
        view_selections: List of (row_idx, col_idx, event_idx, title_suffix)
        view: 'xy', 'xz', or 'yz'
    """
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))

    for row_idx, col_idx, idx, title_suffix in view_selections:
        ax = axes[row_idx, col_idx]

        if idx is None:
            ax.text(0.5, 0.5, "No match found", ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(title_suffix)
            continue

        # Look up event in preloaded cache
        pos = idx_to_pos[idx]
        shower = events_cache[pos]
        mask = shower[:, 4] > 0
        x = shower[mask, 0]
        y = shower[mask, 1]
        z_layer = shower[mask, 2]  # Already in layer numbers
        e = shower[mask, 4]

        energy = incident_energies[idx]
        px, py, pz = momentum[idx]
        sf = sampling_fractions[idx]
        nl = int(num_layers[idx])

        # Compute spherical angles from momentum
        p_mag = np.sqrt(px**2 + py**2 + pz**2)
        theta_deg = np.degrees(np.arccos(np.clip(pz / p_mag, -1, 1)))
        phi_deg = np.degrees(np.arctan2(py, px))

        # Scatter with size based on energy
        e_normalized = (e - e.min()) / (e.max() - e.min() + 1e-6)
        sizes = 1 + e_normalized * 500

        # Select coordinates based on view
        if view == 'xy':
            x_coord, y_coord = x, y
            xlabel, ylabel = "x [mm]", "y [mm]"
            xlim, ylim = (-150, 150), (-150, 150)
            vline_pos = None
        elif view == 'xz':
            x_coord, y_coord = z_layer, x
            xlabel, ylabel = "Layer number", "x [mm]"
            xlim, ylim = (0, 45), (-150, 150)
            vline_pos = nl
        else:  # yz view
            x_coord, y_coord = z_layer, y
            xlabel, ylabel = "Layer number", "y [mm]"
            xlim, ylim = (0, 45), (-150, 150)
            vline_pos = nl

        ax.scatter(x_coord, y_coord, s=sizes, c='darkred', alpha=0.65, edgecolors='none')

        if vline_pos is not None:
            ax.axvline(vline_pos, color='dodgerblue', linestyle='--', linewidth=2,
                      label=f'Max layer = {nl}')
            ax.legend(loc="upper right")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(
            f"{title_suffix}\n"
            f"E = {energy:.1f} GeV, "
            rf"$\theta$ = {theta_deg:.1f}°, $\phi$ = {phi_deg:.1f}°"
        )

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot showers in 3x4 grid")
    parser.add_argument("--ref-dir", type=str, default="sf_nlayers_angles",
                        help="Reference directory")
    parser.add_argument("--sub-dir", type=str, default="SiW_final",
                        help="Sub-directory (default: SiW_final)")
    parser.add_argument("--h5-name", type=str, default="1000simplebox_1-100GeV_sf-nlayers.h5",
                        help="HDF5 filename")
    parser.add_argument("--metadata-file", type=str, default="final_metadata.json",
                        help="Metadata JSON filename")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    h5_file = BASE_DIR / f"output_dataset/SimpleBox/h5/{args.ref_dir}/{args.sub_dir}/{args.h5_name}"
    metadata_json = BASE_DIR / f"calo_configs/par04/SimpleBox/{args.ref_dir}/{args.sub_dir}/{args.metadata_file}"

    # =========================================================================
    # LOAD LIGHTWEIGHT ARRAYS (once — a few MB each)
    # =========================================================================
    print("Loading metadata arrays from HDF5...")
    with h5py.File(h5_file, "r") as f:
        incident_energies = f["incident_energies"][:, 0]
        momentum = f["momentum"][:]
        sampling_fractions = f["sampling_fraction"][:, 0]
        num_layers = f["num_layers"][:, 0]

    total_events = len(incident_energies)

    # =========================================================================
    # ANGLE STATISTICS
    # =========================================================================
    px, py, pz = momentum[:, 0], momentum[:, 1], momentum[:, 2]
    p_mag = np.sqrt(px**2 + py**2 + pz**2)
    cos_theta = np.clip(pz / p_mag, -1.0, 1.0)
    theta_vals = np.degrees(np.arccos(cos_theta))
    phi_vals = np.degrees(np.arctan2(py, px))

    print(f"\n{'='*50}")
    print(f"DATASET STATISTICS (Total events: {total_events})")
    print(f"{'='*50}")
    print(f"Theta Range: [{np.min(theta_vals):8.4f}, {np.max(theta_vals):8.4f}] deg")
    print(f"Phi Range:   [{np.min(phi_vals):8.4f}, {np.max(phi_vals):8.4f}] deg")
    print(f"{'='*50}\n")

    # Load metadata
    with open(metadata_json, "r") as f:
        metadata = json.load(f)

    rng = np.random.default_rng(args.seed)

    # =========================================================================
    # SELECTION CRITERIA (auto-detect from metadata)
    # =========================================================================
    with open(metadata_json, "r") as mf:
        meta = json.load(mf)

    fixed_sf_configs = [m for m in meta if m.get('branch') == 'fixed_sf']
    fixed_layers_configs = [m for m in meta if m.get('branch') == 'fixed_layers']

    if fixed_sf_configs:
        col1_sf = fixed_sf_configs[0]['target_sf']
        col1_layers = sorted([m['num_layers'] for m in fixed_sf_configs])
    else:
        col1_sf, col1_layers = 0.03, [15, 25, 35, 45]

    if fixed_layers_configs:
        col2_layers = fixed_layers_configs[0]['num_layers']
        col2_sfs = sorted([m['target_sf'] for m in fixed_layers_configs])
    else:
        col2_layers, col2_sfs = 35, [0.015, 0.025, 0.035, 0.045]

    # Pick a config for column 3 (varying energy)
    col3_sf = col1_sf
    col3_layers = col1_layers[len(col1_layers) // 2] if col1_layers else 35

    COL1_ENERGY = 75.0  # GeV
    COL2_ENERGY = 85.0  # GeV
    COL3_ENERGIES = [10.0, 45.0, 75.0, 100.0]  # GeV

    selections = [
        # Column 1: Fixed SF, varying layers
        [(col1_sf, l, COL1_ENERGY, f"SF={col1_sf:.3f}, {l} layers")
         for l in col1_layers[:4]],

        # Column 2: Fixed layers, varying SF
        [(sf, col2_layers, COL2_ENERGY, f"SF={sf:.3f}, {col2_layers} layers")
         for sf in col2_sfs[:4]],

        # Column 3: Fixed calorimeter, varying energy
        [(col3_sf, col3_layers, e, f"SF={col3_sf:.3f}, {col3_layers} layers")
         for e in COL3_ENERGIES],
    ]

    # =========================================================================
    # PRE-FIND ALL SHOWER INDICES (for all 3 views, same rng order as original)
    # =========================================================================
    print("Finding shower candidates for all views...")
    view_names = ['xy', 'xz', 'yz']
    all_view_indices = {}
    needed_event_indices = set()

    for view in view_names:
        view_list = []
        for col_idx, col_selections in enumerate(selections):
            for row_idx, (target_sf, target_nl, target_energy, title_suffix) in enumerate(col_selections):
                idx = find_shower_by_criteria(
                    sampling_fractions, num_layers, incident_energies,
                    target_sf=target_sf, target_nl=target_nl,
                    target_energy=target_energy, rng=rng
                )
                view_list.append((row_idx, col_idx, idx, title_suffix))
                if idx is not None:
                    needed_event_indices.add(idx)
        all_view_indices[view] = view_list

    n_found = len(needed_event_indices)
    print(f"Found {n_found} unique events to load (out of {total_events:,} total)")

    # =========================================================================
    # LOAD ONLY NEEDED EVENTS (a few MB instead of ~192 GB)
    # =========================================================================
    if n_found > 0:
        unique_indices = np.array(sorted(needed_event_indices))
        print(f"Loading {n_found} selected events from HDF5...")
        with h5py.File(h5_file, "r") as f:
            events_cache = f["events"][unique_indices.tolist()]
        idx_to_pos = {int(idx): pos for pos, idx in enumerate(unique_indices)}
        print("Events loaded.\n")
    else:
        events_cache = np.array([])
        idx_to_pos = {}
        print("Warning: No matching events found.\n")

    # =========================================================================
    # PLOT STYLE
    # =========================================================================
    plt.rcParams.update({
        'font.size': 22,
        'axes.labelsize': 24,
        'axes.titlesize': 23,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 21
    })

    # =========================================================================
    # GENERATE PLOTS
    # =========================================================================
    for view in tqdm(view_names, desc="Generating views"):
        fig = plot_grid_preloaded(
            events_cache, idx_to_pos,
            incident_energies, momentum, sampling_fractions, num_layers,
            all_view_indices[view], view=view
        )
        out_path = h5_file.parent / f"random_showers_{view}_grid.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        tqdm.write(f"  Saved {view} view to {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
