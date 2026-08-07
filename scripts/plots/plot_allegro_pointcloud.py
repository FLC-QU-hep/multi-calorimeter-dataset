#!/usr/bin/env python3
"""
Point cloud diagnostic plots for ALLEGRO ECal HDF5 dataset.
Reads the output of process_root_to_h5_allegro.py (showerdata-compatible schema).

Features in HDF5 showers: [x, y, layer, energy_GeV, 0]
  x = r_hit * wrap(phi_hit - phi_gun)   [mm]  tangential local frame  (≡ SimpleBox x)
  y = z_hit - z_gun                     [mm]  beam-axis local frame   (≡ SimpleBox y)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.calo_geometry import ALLEGROGeometry

plt.rcParams.update({
    'font.size': 14, 
    'axes.labelsize': 15, 
    'axes.titlesize': 15,
    'xtick.labelsize': 12, 
    'ytick.labelsize': 12, 
    'legend.fontsize': 12,
    'savefig.dpi': 200, 
    'savefig.bbox': 'tight',
})


def load_h5(path):
    """Load showerdata-compatible HDF5 and return flat-per-event arrays."""
    with h5py.File(path, 'r') as f:
        raw_showers = f['showers'][:]          # (N,) object — each element flat float32
        nhits       = f['num_points'][:]       # (N,)
        e_inc       = f['energies'][:].ravel() # (N,)
        sf          = f['sampling_fraction'][:].ravel() if 'sampling_fraction' in f else None
        nl          = f['num_layers'][:].ravel().astype(int) if 'num_layers' in f else None
        dirs        = f['directions'][:] if 'directions' in f else None   # (N,3)
        attrs       = dict(f.attrs)

    # Reconstruct (N, max_hits, 5) — variable length, pad with zeros
    n_feat   = int(attrs.get('shape', [0, 0, 5])[2]) if 'shape' in attrs else 5
    max_hits = int(nhits.max())
    n        = len(nhits)
    events   = np.zeros((n, max_hits, n_feat), dtype=np.float32)
    for i, arr in enumerate(raw_showers):
        k = nhits[i]
        events[i, :k] = arr.reshape(k, n_feat)

    return events, nhits, e_inc, sf, nl, dirs, attrs


def flatten_hits(events, nhits):
    """Stack all hits from all showers into one array."""
    return np.vstack([events[i, :nhits[i]] for i in range(len(nhits))])


def plot_pointclouds(h5_file, output_dir):
    h5_file    = Path(h5_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events, nhits, e_inc, sf, nl, dirs, attrs = load_h5(h5_file)
    geo = ALLEGROGeometry()

    n = len(nhits)
    all_hits = flatten_hits(events, nhits)
    dht_all = all_hits[:, 0]       # tangential local [mm]
    dhz_all = all_hits[:, 1]       # z local [mm]
    lay_all = all_hits[:, 2].astype(int)
    e_all   = all_hits[:, 3]       # hit energy [GeV]

    e_dep = np.array([events[i, :nhits[i], 3].sum() for i in range(n)])
    sf_arr = (e_dep / e_inc) if sf is None else sf
    cell_size = attrs.get('cell_size_mm', 1.0)

    print(f"Loaded {n} showers,  {len(e_all):,} total hits")
    print(f"Hits/event:        mean={nhits.mean():.0f}, max={nhits.max()}")
    print(f"x range:           [{dht_all.min():.1f}, {dht_all.max():.1f}] mm")
    print(f"y range:           [{dhz_all.min():.1f}, {dhz_all.max():.1f}] mm")
    print(f"Sampling fraction: {sf_arr.mean():.4f} ± {sf_arr.std():.4f}")
    print(f"\nLayer distribution (hits | E_frac):")
    for l in range(geo.num_layers):
        mask_l = lay_all == l
        h = mask_l.sum()
        ef = e_all[mask_l].sum() / e_all.sum() * 100
        print(f"  Layer {l:2d}: {h:8,} hits  {ef:5.1f}% energy")

    log_e = np.log10(np.clip(e_all, 1e-9, None))
    norm  = mcolors.Normalize(vmin=log_e.min(), vmax=log_e.max())
    kw    = dict(s=1, cmap='hot', alpha=0.5, edgecolors='none', norm=norm)

    # sub-sample for scatter plots
    max_pts = 80_000
    if len(e_all) > max_pts:
        idx = np.random.choice(len(e_all), max_pts, replace=False)
        dht_p, dhz_p, lay_p, e_p = dht_all[idx], dhz_all[idx], lay_all[idx], e_all[idx]
    else:
        dht_p, dhz_p, lay_p, e_p = dht_all, dhz_all, lay_all, e_all
    log_ep = np.log10(np.clip(e_p, 1e-9, None))

    # ── Figure 1: local frame scatter (all showers) ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(f"ALLEGRO Local Frame — All Showers  ({n} events, "
                 f"{len(e_all):,} hits, cell={cell_size} mm)")

    sc = axes[0].scatter(dht_p, dhz_p, c=log_ep, **kw)
    axes[0].set(xlabel='x [mm]', ylabel='y [mm]',
                title='Transverse local frame (all showers)')
    plt.colorbar(sc, ax=axes[0], label='log$_{10}$(E [GeV])')

    sc2 = axes[1].scatter(lay_p, dht_p, c=log_ep, **kw)
    axes[1].set(xlabel='Layer', ylabel='x [mm]', title='Layer vs x')
    plt.colorbar(sc2, ax=axes[1], label='log$_{10}$(E [GeV])')

    sc3 = axes[2].scatter(lay_p, dhz_p, c=log_ep, **kw)
    axes[2].set(xlabel='Layer', ylabel='y [mm]', title='Layer vs y')
    plt.colorbar(sc3, ax=axes[2], label='log$_{10}$(E [GeV])')

    plt.tight_layout()
    fig.savefig(output_dir / 'allegro_pc_local_frame.png')
    plt.close()
    print("Saved allegro_pc_local_frame.png")

    # ── Figure 2: multi-shower grid (plot_shower_comparison.py style) ─────────
    # Pick 3 representative showers: low / medium / high energy
    e_order = np.argsort(e_inc)
    n10 = max(1, len(e_order) // 10)
    pick_indices = [e_order[n10], e_order[len(e_order) // 2], e_order[-n10]]
    colors_row   = ['firebrick', 'steelblue', 'darkgreen']

    def _shower_on_ax(ax, dht, dhz, lay, e, view, color):
        mask = e > 1e-9
        if not mask.any():
            return
        dt, dz, lv, ev = dht[mask], dhz[mask], lay[mask], e[mask]
        e_norm = (ev - ev.min()) / (ev.max() - ev.min() + 1e-10)
        sz = 2 + e_norm * 250
        if view == 'xy':
            ax.scatter(dt, dz, s=sz, c=color, alpha=0.7, edgecolors='none')
            ax.set(xlabel='x [mm]', ylabel='y [mm]',
                   xlim=(-300, 300), ylim=(-300, 300))
        elif view == 'xz':
            ax.scatter(lv, dt, s=sz, c=color, alpha=0.7, edgecolors='none')
            ax.axvline(geo.num_layers, color='k', ls='--', lw=1.5, alpha=0.5)
            ax.set(xlabel='Layer', ylabel='x [mm]',
                   xlim=(-1, 45), ylim=(-300, 300))
        else:  # yz
            ax.scatter(lv, dz, s=sz, c=color, alpha=0.7, edgecolors='none')
            ax.axvline(geo.num_layers, color='k', ls='--', lw=1.5, alpha=0.5)
            ax.set(xlabel='Layer', ylabel='y [mm]',
                   xlim=(-1, 45), ylim=(-300, 300))
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
            spine.set_color(color)

    with plt.rc_context({'font.family': 'serif', 'font.size': 22,
                         'axes.labelsize': 28, 'axes.titlesize': 26,
                         'xtick.labelsize': 22, 'ytick.labelsize': 22,
                         'axes.linewidth': 1.5,
                         'xtick.major.width': 1.5, 'ytick.major.width': 1.5}):
        fig, axes = plt.subplots(3, 3, figsize=(30, 26))

        for row, (idx, col) in enumerate(zip(pick_indices, colors_row)):
            hits_r = events[idx, :nhits[idx]]
            dht_r, dhz_r = hits_r[:, 0], hits_r[:, 1]
            lay_r, e_r   = hits_r[:, 2].astype(int), hits_r[:, 3]

            # title info
            ei = e_inc[idx]
            sfi = sf_arr[idx] if hasattr(sf_arr, '__len__') else sf_arr
            if dirs is not None:
                d = dirs[idx]
                pmag = np.linalg.norm(d)
                theta_deg = np.degrees(np.arccos(np.clip(d[2] / pmag, -1, 1))) if pmag > 0 else 0
                phi_deg   = np.degrees(np.arctan2(d[1], d[0]))
                ang_str   = rf"$\theta$={theta_deg:.0f}°, $\phi$={phi_deg:.0f}°"
            else:
                ang_str = ''

            title = (rf"$\bf{{E}}$={ei:.1f} GeV  |  SF={sfi:.3f}"
                     + (f"\n{ang_str}" if ang_str else '')
                     + f"  |  {nhits[idx]:,} cells")

            for col_idx, view in enumerate(['xy', 'xz', 'yz']):
                ax = axes[row, col_idx]
                _shower_on_ax(ax, dht_r, dhz_r, lay_r, e_r, view, col)
                if col_idx == 0:
                    ax.set_title(title, fontsize=24, pad=10)

        col_headers = ['x vs y  (top view)',
                       'Layer vs x',
                       'Layer vs y']
        for col_idx, hdr in enumerate(col_headers):
            axes[0, col_idx].set_title(
                hdr + '\n' + axes[0, col_idx].get_title(), fontsize=24, pad=10)

        plt.tight_layout()
        fig.savefig(output_dir / 'allegro_pc_showers.png', dpi=200, bbox_inches='tight')
        plt.close()
    print("Saved allegro_pc_showers.png")

    # ── Figure 3: dataset overview (2×3) ──────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(21, 11))

    # (0,0) Incident energy — linear bins 1–100 GeV
    axes[0, 0].hist(e_inc, bins=25, range=(1, 100), histtype='step', lw=1.5, color='darkviolet')
    axes[0, 0].set(xlabel='$E_{inc}$ [GeV]', ylabel='Events',
                   title='Incident energy', xlim=(1, 100))

    # (0,1) Hit multiplicity
    axes[0, 1].hist(nhits, bins=20, histtype='stepfilled', color='coral', alpha=0.7)
    axes[0, 1].axvline(nhits.mean(), color='red', ls='--', lw=1.5,
                       label=f'mean={nhits.mean():.0f}')
    axes[0, 1].set(xlabel='Hits/event', ylabel='Events', title='Hit multiplicity')
    axes[0, 1].legend()

    # (0,2) Sampling fraction
    axes[0, 2].hist(sf_arr, bins=15, histtype='stepfilled', color='seagreen', alpha=0.7)
    axes[0, 2].axvline(sf_arr.mean(), color='darkgreen', ls='--', lw=1.5,
                       label=f'mean={sf_arr.mean():.3f}')
    axes[0, 2].set(xlabel='Sampling fraction', ylabel='Events', title='Sampling fraction')
    axes[0, 2].legend()

    # (1,0) Hit energy spectrum
    e_mev = e_all[e_all > 1e-9] * 1e3
    bins_mev = np.logspace(np.log10(e_mev.min()), np.log10(e_mev.max()), 80)
    axes[1, 0].hist(e_mev, bins=bins_mev, histtype='step', lw=1.5, color='orange')
    axes[1, 0].set(xlabel='Hit energy [MeV]', ylabel='Count',
                   title='Hit energy spectrum', xscale='log', yscale='log')

    # (1,1) Longitudinal profile (mean E_dep per layer) — step histogram like plot_allegro_root.py
    lbins = np.arange(geo.num_layers + 1)
    mean_e, _ = np.histogram(lay_all, bins=lbins, weights=e_all)
    mean_e /= n
    axes[1, 1].hist(np.arange(geo.num_layers), bins=lbins, weights=mean_e,
                    histtype='step', lw=1.5, color='steelblue')
    axes[1, 1].set(xlabel='Layer (depth)', ylabel='Mean $E_{dep}$ [GeV]',
                   title='Longitudinal profile',
                   xlim=(0, geo.num_layers), xticks=np.arange(geo.num_layers))
    axes[1, 1].set_ylim(bottom=0)

    # (1,2) Radial (lateral) profile in local frame: mean E vs r_local = sqrt(dh_t²+dh_z²)
    r_local = np.sqrt(dht_all**2 + dhz_all**2)
    r_bins = np.linspace(0, np.percentile(r_local, 99), 50)
    mean_r, _ = np.histogram(r_local, bins=r_bins, weights=e_all)
    mean_r /= n
    axes[1, 2].bar(0.5 * (r_bins[:-1] + r_bins[1:]), mean_r,
                   width=np.diff(r_bins), color='slateblue', alpha=0.8, edgecolor='none')
    axes[1, 2].set(xlabel=r'$r_{local}$ [mm]',
                   ylabel='Mean $E_{dep}$ [GeV]', title='Radial (lateral) profile')
    axes[1, 2].set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(output_dir / 'allegro_pc_overview.png')
    plt.close()
    print("Saved allegro_pc_overview.png")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5-file',    required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    plot_pointclouds(args.h5_file, args.output_dir)
