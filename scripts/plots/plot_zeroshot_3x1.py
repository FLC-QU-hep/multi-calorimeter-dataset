#!/usr/bin/env python3
"""
3x1 physics plot for the SiW zeroshot dataset (sf=0.035, nl=35).
Panels: longitudinal layer energy | radial energy | cell energy spectrum.
Style matches analysis_3x2_comparison.png.
"""
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

BASE = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[2]))
H5_PATH = (
    f"{BASE}/output_dataset/SimpleBox/showerdata/sf_nlayers_angles/"
    "SiW_zeroshot/1simplebox_100k_1-100GeV_sf0035_nl35_zeroshot.h5"
)
OUT_PNG = (
    f"{BASE}/output_dataset/SimpleBox/showerdata/sf_nlayers_angles/"
    "SiW_zeroshot/analysis_3x1_zeroshot_sf0035_nl35.png"
)

if len(sys.argv) > 1:
    H5_PATH = sys.argv[1]
if len(sys.argv) > 2:
    OUT_PNG = sys.argv[2]

# ── reference-style rcParams ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.fontsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": True,
    "axes.spines.right": True,
})

COLOR = "#3A76AF"   # blue matching viridis-low used in reference


def load_showerdata(path):
    with h5py.File(path, "r") as f:
        showers_vlen = f["showers"][:]
        num_points   = f["num_points"][:]
        energies     = f["energies"][:].flatten()
        shape        = f["shape"][:]
        sf           = float(f["sampling_fraction"][0, 0])
        nl           = int(f["num_layers"][0, 0])
    n_feat = int(shape[2])
    return showers_vlen, num_points, energies, n_feat, sf, nl


def main():
    print(f"Loading {H5_PATH} ...")
    showers_vlen, num_points, energies, n_feat, sf, nl = load_showerdata(H5_PATH)
    n_showers = len(num_points)
    print(f"  {n_showers:,} showers  sf={sf:.4f}  nl={nl}  n_feat={n_feat}")

    # ── accumulate per-layer and per-hit data ─────────────────────────────────
    # feature order after combine_temp_to_showerdata: [x, y, layer_idx, E_GeV, time]
    layer_energy_sum = np.zeros(nl, dtype=np.float64)   # total E per layer (GeV)
    layer_nhits      = np.zeros(nl, dtype=np.int64)

    all_r   = []
    all_e   = []   # GeV

    for i, shower in enumerate(showers_vlen):
        n = int(num_points[i])
        if n == 0:
            continue
        pts = shower[:n * n_feat].reshape(n, n_feat)
        x_s     = pts[:, 0]
        y_s     = pts[:, 1]
        layer_s = pts[:, 2].astype(int)
        e_s     = pts[:, 3]   # GeV

        # clamp layer index to valid range
        valid = (layer_s >= 0) & (layer_s < nl)
        np.add.at(layer_energy_sum, layer_s[valid], e_s[valid])
        np.add.at(layer_nhits,      layer_s[valid], 1)

        all_r.append(np.sqrt(x_s**2 + y_s**2))
        all_e.append(e_s)

    all_r = np.concatenate(all_r)
    all_e = np.concatenate(all_e)

    # mean energy per layer in MeV (averaged over all showers)
    mean_e_layer_mev = layer_energy_sum / n_showers * 1e3   # GeV → MeV

    # radial profile: mean E per bin in MeV
    r_max   = np.percentile(all_r, 99.5)
    r_bins  = np.linspace(0, r_max, 51)
    r_cents = 0.5 * (r_bins[:-1] + r_bins[1:])
    e_radial = np.array([
        all_e[(all_r >= r_bins[j]) & (all_r < r_bins[j + 1])].sum()
        for j in range(len(r_bins) - 1)
    ]) / n_showers * 1e3   # → MeV

    # cell energy in MeV
    e_mev = all_e * 1e3

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"SiW Zeroshot  —  SF $\\approx$ {sf:.3f},  L = {nl}",
        fontsize=15, fontweight="bold", y=1.01
    )

    label = f"G4  sf={sf:.3f},  nl={nl}"

    # --- Panel 1: Longitudinal layer energy ---
    ax = axes[0]
    layers = np.arange(nl)
    ax.step(layers, mean_e_layer_mev, where="mid",
            color=COLOR, linewidth=2.0, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Layer Number")
    ax.set_ylabel("Mean Energy [MeV]")
    ax.set_title("Longitudinal Layer Energy")
    ax.set_xlim(-0.5, nl - 0.5)
    ax.legend()

    # --- Panel 2: Radial energy distribution ---
    ax = axes[1]
    # mask zero bins for clean log plot
    mask = e_radial > 0
    ax.step(r_cents[mask], e_radial[mask], where="mid",
            color=COLOR, linewidth=2.0, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Radius [mm]")
    ax.set_ylabel("Mean Energy [MeV]")
    ax.set_title("Radial Energy Distribution")
    ax.legend()

    # --- Panel 3: Cell energy spectrum ---
    ax = axes[2]
    bins = np.logspace(np.log10(max(e_mev.min(), 1e-2)),
                       np.log10(e_mev.max()), 80)
    ax.hist(e_mev, bins=bins, histtype="step", linewidth=2.0,
            color=COLOR, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cell Energy [MeV]")
    ax.set_ylabel("Counts")
    ax.set_title("Cell Energy Spectrum")
    ax.legend()

    plt.tight_layout()
    import os; os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved -> {OUT_PNG}")
    return OUT_PNG


if __name__ == "__main__":
    main()
