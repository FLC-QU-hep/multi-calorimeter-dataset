#!/usr/bin/env python3
"""
Generate the LEMURS H5 validation 2x3 comparison plot.

Reads H5 files for par04_siw, par04_scipb, odd, fccee_cld and produces:
  - Top row: Incident Energy, ECal Deposit vs E + SF fit, Number of Cells
  - Bottom row: Longitudinal profile, Lateral profile (radius), Cell energy spectrum
"""
import sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt

import os
_REPO = Path(os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parent.parent.parent)))
sys.path.append(str(_REPO))
from utils.calo_geometry import LEMURSBarrelGeometry

BASE = _REPO / "output_dataset"

DETECTORS = {
    "par04_siw":  {"dir": "Par04_SiW",  "color": "#4B0082"},
    "par04_scipb": {"dir": "Par04_SciPb", "color": "#006D6F"},
    "odd":        {"dir": "ODD",         "color": "#CC5500"},
    "fccee_cld":  {"dir": "FCCee_CLD",   "color": "#8B9A1B"},
}

N_FEAT = 5


MAX_SHOWERS = int(os.environ.get("MAX_SHOWERS", 20000))


def load_h5(det_name, det_cfg):
    h5_dir = BASE / det_cfg["dir"] / "h5_1M"
    h5_files = sorted(h5_dir.glob("*.h5"))
    if not h5_files:
        print(f"WARNING: no H5 files for {det_name} in {h5_dir}")
        return None

    all_showers, all_energies, all_nhits = [], [], []
    total = 0
    for f in h5_files:
        if total >= MAX_SHOWERS:
            break
        with h5py.File(f, "r") as hf:
            n = hf["energies"].shape[0]
            take = min(n, MAX_SHOWERS - total)
            for i in range(take):
                s = hf["showers"][i].reshape(-1, N_FEAT)
                all_showers.append(s)
            all_energies.append(hf["energies"][:take].ravel())
            all_nhits.append(hf["num_points"][:take].ravel())
            total += take
    print(f"{det_name}: loaded {total} showers from {h5_dir}")

    return {
        "showers": all_showers,
        "energies": np.concatenate(all_energies),
        "nhits": np.concatenate(all_nhits),
    }


def main():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    emax_markers = {}  # store max energy per detector for markers

    for det_name, det_cfg in DETECTORS.items():
        data = load_h5(det_name, det_cfg)
        if data is None:
            continue

        geo = LEMURSBarrelGeometry(det_name)
        color = det_cfg["color"]
        energies = data["energies"]
        nhits = data["nhits"]
        showers = data["showers"]
        n = len(energies)

        # Sampling fraction
        e_dep = np.array([s[:, 3].sum() for s in showers])
        sf = np.mean(e_dep / energies)

        label_base = det_name

        # --- Top left: Incident Energy ---
        ax = axes[0, 0]
        e_bins = np.logspace(np.log10(max(energies.min(), 0.5)), np.log10(energies.max()), 20)
        ax.hist(energies, bins=e_bins, histtype="step", color=color, linewidth=1.5,
                label=f"{label_base} [{energies.min():.3g}-{energies.max():.3g} GeV]")
        emax_markers[det_name] = (energies.max(), color)

        # --- Top center: ECal Deposit vs E ---
        ax = axes[0, 1]
        ax.scatter(energies, e_dep, s=8, alpha=0.6, color=color, zorder=2)
        # SF fit line
        e_sort = np.sort(energies)
        ax.plot(e_sort, sf * e_sort, "--", color=color, linewidth=1.5,
                label=f"{label_base} (SF={sf:.4f})")

        # --- Top right: Number of Cells ---
        ax = axes[0, 2]
        c_bins = np.logspace(np.log10(max(nhits.min(), 1)), np.log10(nhits.max()), 30)
        ax.hist(nhits, bins=c_bins, histtype="step", color=color, linewidth=1.5,
                label=f"{label_base} (mean={nhits.mean():.0f}, max={nhits.max()})")

        # --- Bottom left: Longitudinal profile ---
        ax = axes[1, 0]
        all_layers = np.concatenate([s[:, 2] for s in showers])
        all_layer_e = np.concatenate([s[:, 3] for s in showers]) * 1e3  # MeV
        l_bins = np.arange(-0.5, geo.num_layers + 0.5, 1)
        layer_energy, _ = np.histogram(all_layers, bins=l_bins, weights=all_layer_e)
        layer_energy /= n
        ax.hist(np.arange(geo.num_layers), bins=l_bins, weights=layer_energy,
                histtype="step", color=color, linewidth=1.5, label=label_base)

        # --- Bottom center: Lateral profile (radius = |dh_t|) ---
        ax = axes[1, 1]
        all_dh_t = np.concatenate([s[:, 0] for s in showers])
        all_e = np.concatenate([s[:, 3] for s in showers]) * 1e3  # MeV
        radius = np.abs(all_dh_t)

        r_bins = np.linspace(0, 120, 25)
        ax.hist(radius, bins=r_bins, weights=all_e / n,
                histtype="step", color=color, linewidth=1.5, label=label_base)

        # --- Bottom right: Cell energy spectrum ---
        ax = axes[1, 2]
        all_cell_e = np.concatenate([s[:, 3] for s in showers]) * 1e3  # MeV
        ax.hist(all_cell_e, bins=np.logspace(np.log10(max(all_cell_e.min(), 1e-4)),
                np.log10(all_cell_e.max()), 60),
                histtype="step", color=color, linewidth=1.5, label=label_base)

    # --- Formatting ---
    FS_LABEL = 16

    # Top left: Incident Energy + max energy markers
    axes[0, 0].set_xlabel("Incident Energy [GeV]", fontsize=FS_LABEL)
    axes[0, 0].set_ylabel("Events", fontsize=FS_LABEL)
    axes[0, 0].set_xscale("log")
    # for dname, (emax, col) in emax_markers.items():
    #     axes[0, 0].axvline(emax, color=col, linestyle=":", alpha=0.6, linewidth=1.0)
    #     axes[0, 0].text(emax, 1.02, f"{emax:.0f}", color=col,
    #                     fontsize=7, ha="center", va="bottom", rotation=90,
    #                     transform=axes[0, 0].get_xaxis_transform(), clip_on=False)
    # global_emax = max(v[0] for v in emax_markers.values())
    axes[0, 0].set_xlim(1,1e3)
    axes[0, 0].legend(fontsize=12)

    axes[0, 1].set_xlabel("Incident Energy [GeV]", fontsize=FS_LABEL)
    axes[0, 1].set_ylabel("ECal Deposit [GeV]", fontsize=FS_LABEL)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlim(1,1e3)
    axes[0, 1].set_ylim(1e-2, None)
    axes[0, 1].legend(fontsize=12)

    axes[0, 2].set_xlabel("Number of Cells", fontsize=FS_LABEL)
    axes[0, 2].set_ylabel("Events", fontsize=FS_LABEL)
    axes[0, 2].set_xscale("log")
    axes[0, 2].set_xlim(5e1, None)
    axes[0, 2].legend(fontsize=12)

    axes[1, 0].set_xlabel("Layer Number", fontsize=FS_LABEL)
    axes[1, 0].set_ylabel("Mean Energy [MeV]", fontsize=FS_LABEL)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylim(3e-1, None)
    axes[1, 0].set_xlim(0, 90)
    # Add dashed vertical lines at each detector's last layer + rotated labels at top
    for det_name, det_cfg in DETECTORS.items():
        geo = LEMURSBarrelGeometry(det_name)
        x_pos = geo.num_layers - 0.5
        axes[1, 0].axvline(x_pos, color=det_cfg["color"], linestyle="--",
                            alpha=0.5, linewidth=0.8)
        axes[1, 0].text(x_pos-1.5, 0.9, str(geo.num_layers), color=det_cfg["color"],
                        fontsize=16, ha="center", va="bottom", rotation=90,
                        transform=axes[1, 0].get_xaxis_transform(), clip_on=False)

    axes[1, 1].set_xlabel("Radius [mm]", fontsize=FS_LABEL)
    axes[1, 1].set_ylabel("Mean Energy [MeV]", fontsize=FS_LABEL)
    axes[1, 1].set_xlim(0, 120)
    axes[1, 1].set_yscale("log")

    axes[1, 2].set_xlabel("Cell Energy [MeV]", fontsize=FS_LABEL)
    axes[1, 2].set_ylabel("Counts", fontsize=FS_LABEL)
    axes[1, 2].set_xscale("log")
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_xlim(1e-2, None)

    # Global legend at bottom
    handles, labels = [], []
    for det_name, det_cfg in DETECTORS.items():
        handles.append(plt.Line2D([0], [0], color=det_cfg["color"], linewidth=2))
        labels.append(det_name)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=26,
                frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = BASE / "lemurs_h5_validation.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()