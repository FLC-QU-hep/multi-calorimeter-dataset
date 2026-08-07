#!/usr/bin/env python3
"""
Diagnostic plots for ALLEGRO ECal step-level simulation (v03).
module load maxwell mamba && . mamba-init && conda activate calo-transfer
python multi-calorimeter-dataset/plot_allegro_root.py \
    --root-file multi-calorimeter-dataset/output_dataset/ALLEGRO/root/angles/allegro_angles_0000.root \
    --output-dir multi-calorimeter-dataset/output_dataset/ALLEGRO/root/plots/
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uproot
import awkward as ak

REPO_ROOT = Path(os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO_ROOT))
from utils.calo_geometry import ALLEGROGeometry

ECAL_COLLECTION = "ECalBarrelModuleThetaMerged"
CONTRIB_COLLECTION = "ECalBarrelModuleThetaMergedContributions"

# Publication-ready defaults
plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 26,
    'axes.titlesize': 26,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 23,
    'figure.titlesize': 27,
    'lines.linewidth': 1.5,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})


def plot_allegro_diagnostics(root_file, output_dir):
    root_file = Path(root_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    f = uproot.open(root_file)
    tree = f["events"]
    geom = ALLEGROGeometry(version='v03')

    # --- Load contributions (step-level) ---
    cp = f"{CONTRIB_COLLECTION}/{CONTRIB_COLLECTION}"
    contribs = tree.arrays([f"{cp}.energy", f"{cp}.stepPosition.x",
                            f"{cp}.stepPosition.y", f"{cp}.stepPosition.z"], library="ak")
    e_ak = contribs[f"{cp}.energy"]
    x_ak = contribs[f"{cp}.stepPosition.x"]
    y_ak = contribs[f"{cp}.stepPosition.y"]
    z_ak = contribs[f"{cp}.stepPosition.z"]

    has_steps = np.count_nonzero(ak.to_numpy(x_ak[0])) > 0

    # --- Load cell-level hits ---
    hp = f"{ECAL_COLLECTION}/{ECAL_COLLECTION}"
    cells = tree.arrays([f"{hp}.energy", f"{hp}.position.x",
                         f"{hp}.position.y", f"{hp}.position.z"], library="ak")

    # --- MC truth (primary photon only: generatorStatus == 1) ---
    mc = tree.arrays(["MCParticles/MCParticles.momentum.x",
                      "MCParticles/MCParticles.momentum.y",
                      "MCParticles/MCParticles.momentum.z",
                      "MCParticles/MCParticles.vertex.x",
                      "MCParticles/MCParticles.vertex.y",
                      "MCParticles/MCParticles.vertex.z",
                      "MCParticles/MCParticles.generatorStatus"], library="ak")
    mask = mc["MCParticles/MCParticles.generatorStatus"] == 1
    px   = ak.firsts(mc["MCParticles/MCParticles.momentum.x"][mask]).to_numpy()
    py   = ak.firsts(mc["MCParticles/MCParticles.momentum.y"][mask]).to_numpy()
    pz   = ak.firsts(mc["MCParticles/MCParticles.momentum.z"][mask]).to_numpy()
    vx   = ak.firsts(mc["MCParticles/MCParticles.vertex.x"][mask]).to_numpy()
    vy   = ak.firsts(mc["MCParticles/MCParticles.vertex.y"][mask]).to_numpy()
    vz   = ak.firsts(mc["MCParticles/MCParticles.vertex.z"][mask]).to_numpy()
    e_inc = np.sqrt(px**2 + py**2 + pz**2)

    # Derived gun quantities
    r_gun     = np.sqrt(vx**2 + vy**2)                    # radial distance of gun vertex
    theta_gun = np.arctan2(np.sqrt(px**2 + py**2), pz)    # polar angle from +z axis
    phi_gun   = np.arctan2(py, px)                         # azimuthal angle
    cos_theta = np.cos(theta_gun)
    # Expected z from formula: z = r_gun * cot(theta)
    z_expected = r_gun * np.cos(theta_gun) / np.sin(theta_gun)

    THETA_MIN, THETA_MAX = 0.87, 2.27   # rad — LEMURS range

    n_events = len(e_ak)

    if has_steps:
        pos_x, pos_y, pos_z, pos_e = x_ak, y_ak, z_ak, e_ak
        data_label = "step-level"
    else:
        pos_x = cells[f"{hp}.position.x"]
        pos_y = cells[f"{hp}.position.y"]
        pos_z = cells[f"{hp}.position.z"]
        pos_e = cells[f"{hp}.energy"]
        data_label = "cell-level"

    cell_etot = ak.sum(cells[f"{hp}.energy"], axis=1).to_numpy()
    nhits_steps = ak.num(e_ak).to_numpy()
    nhits_cells = ak.num(cells[f"{hp}.energy"]).to_numpy()
    ratio = cell_etot / e_inc

    e_flat = ak.flatten(pos_e).to_numpy()
    x_flat = ak.flatten(pos_x).to_numpy()
    y_flat = ak.flatten(pos_y).to_numpy()
    z_flat = ak.flatten(pos_z).to_numpy()
    rho_flat = np.sqrt(x_flat**2 + y_flat**2)

    print(f"Loaded {n_events} events from {root_file.name} ({data_label})")
    print(f"Steps: {nhits_steps.sum()}, Cells: {nhits_cells.sum()}")
    print(f"Sampling fraction: {ratio.mean():.4f} +/- {ratio.std():.4f}")

    # =====================================================================
    # Figure 1: Overview
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f"ALLEGRO ECal Overview  ({n_events} events, {data_label})")

    # 1a: Incident energy distribution — linear bins to show uniform sampling correctly
    e_bins = np.linspace(e_inc.min(), e_inc.max(), 30)
    axes[0, 0].hist(e_inc, bins=e_bins, histtype='step', lw=1.5, color='darkviolet')
    axes[0, 0].set(xlabel="Incident energy [GeV]", ylabel="Events",
                   title="Incident energy (uniform)")

    # 1b: Deposit vs incident
    axes[0, 1].scatter(e_inc, cell_etot, s=30, alpha=0.7, c='steelblue', edgecolors='none')
    eref = np.linspace(e_inc.min(), e_inc.max(), 100)
    axes[0, 1].plot(eref, eref * ratio.mean(), 'r--', lw=1.5,
                    label=f'SF = {ratio.mean():.3f}')
    axes[0, 1].set(xlabel="Incident energy [GeV]", ylabel="ECal deposit [GeV]",
                   title="Deposit vs incident", xscale='log', yscale='log')
    axes[0, 1].legend()

    # 1c: Number of steps distribution
    axes[1, 0].hist(nhits_steps, bins=25, histtype='stepfilled', color='coral', alpha=0.7)
    axes[1, 0].axvline(nhits_steps.mean(), color='red', ls='--', lw=1.5,
                       label=f'mean = {nhits_steps.mean():.0f}')
    axes[1, 0].set(xlabel="Number of G4 steps", ylabel="Events",
                   title="Steps per event")
    axes[1, 0].legend()

    # 1d: Step energy spectrum
    e_mev = e_flat[e_flat >= 1e-5] * 1e3
    if len(e_mev) > 0:
        bins = np.logspace(np.log10(0.01), np.log10(e_mev.max()), 100)
        axes[1, 1].hist(e_mev, bins=bins, histtype='step', lw=1.5, color='orange')
    axes[1, 1].set(xlabel="Step energy [MeV]", ylabel="Count",
                   title="Step energy spectrum", xscale='log', yscale='log')

    plt.tight_layout()
    fig.savefig(output_dir / "allegro_overview.png"); plt.close()

    # =====================================================================
    # Figure 2: Single shower
    # =====================================================================
    best = np.argmax(nhits_steps)
    ex = ak.to_numpy(pos_e[best])
    xx, yx, zx = (ak.to_numpy(pos_x[best]), ak.to_numpy(pos_y[best]),
                   ak.to_numpy(pos_z[best]))

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    fig.suptitle(f"Single Shower  (evt {best},  $E_{{inc}}$={e_inc[best]:.1f} GeV,  "
                 f"$E_{{dep}}$={cell_etot[best]:.2f} GeV,  {len(ex):,} steps)")
    kw = dict(c=np.log10(ex + 1e-12), s=4, cmap='hot', alpha=0.8, edgecolors='none')
    views = [(xx, yx, "x [mm]", "y [mm]", "x-y view"),
             (xx, zx, "x [mm]", "z [mm]", "x-z view"),
             (zx, yx, "z [mm]", "y [mm]", "z-y view")]
    for ax, (a, b, xl, yl, ti) in zip(axes, views):
        sc = ax.scatter(a, b, **kw)
        ax.set(xlabel=xl, ylabel=yl, title=ti)
        plt.colorbar(sc, ax=ax, label='log$_{10}$(E [GeV])')
    axes[0].set_aspect('equal')

    plt.tight_layout()
    fig.savefig(output_dir / "allegro_single_shower.png"); plt.close()

    # =====================================================================
    # Figure 3: All showers
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    fig.suptitle(f"All Showers  ({n_events} events,  {len(e_flat):,} {data_label} hits)")

    max_pts = 50000
    if len(e_flat) > max_pts:
        idx = np.random.choice(len(e_flat), max_pts, replace=False)
        xp, yp, zp, ep = x_flat[idx], y_flat[idx], z_flat[idx], e_flat[idx]
    else:
        xp, yp, zp, ep = x_flat, y_flat, z_flat, e_flat
    vmin = -5
    kw = dict(c=np.clip(np.log10(ep + 1e-12), vmin, None), s=1, cmap='hot',
              alpha=0.5, edgecolors='none', vmin=vmin)

    sc1 = axes[0].scatter(xp, yp, **kw)
    # Circles = cylindrical barrel cross-section: dotted = inner (r_min), dashed = outer (r_max)
    axes[0].add_patch(plt.Circle((0, 0), geom.r_min, fill=False, color='black', ls=':', lw=1.5))
    axes[0].add_patch(plt.Circle((0, 0), geom.r_max, fill=False, color='black', ls='--', lw=1.5))
    axes[0].set(xlabel="x [mm]", ylabel="y [mm]", title="x-y view", aspect='equal')

    # z-x view: z on x-axis, x on y-axis
    # Barrel boundary in side view: horizontal lines at ±r_min (inner) and ±r_max (outer)
    sc2 = axes[1].scatter(zp, xp, **kw)
    for r, ls in [(geom.r_min, ':'), (geom.r_max, '--')]:
        axes[1].axhline( r, color='black', ls=ls, lw=1.5)
        axes[1].axhline(-r, color='black', ls=ls, lw=1.5)
    axes[1].set(xlabel="z [mm]", ylabel="x [mm]", title="z-x view")

    # z-y view: z on x-axis, y on y-axis — same barrel lines
    sc3 = axes[2].scatter(zp, yp, **kw)
    for r, ls in [(geom.r_min, ':'), (geom.r_max, '--')]:
        axes[2].axhline( r, color='black', ls=ls, lw=1.5)
        axes[2].axhline(-r, color='black', ls=ls, lw=1.5)
    axes[2].set(xlabel="z [mm]", ylabel="y [mm]", title="z-y view")

    for ax, sc in zip(axes, [sc1, sc2, sc3]):
        plt.colorbar(sc, ax=ax, label='log$_{10}$(E [GeV])')

    plt.tight_layout()
    fig.savefig(output_dir / "allegro_all_showers.png"); plt.close()

    # =====================================================================
    # Figure 4: Shower profiles
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"ALLEGRO ECal Shower Profiles  ({n_events} events)")

    # Longitudinal: energy per radial layer (= depth)
    layers_flat = geom.rho_to_layer(rho_flat)
    lbins = np.arange(geom.num_layers+1)
    mean_layer, _ = np.histogram(layers_flat, bins=lbins, weights=e_flat)
    mean_layer /= n_events

    axes[0].hist(np.arange(geom.num_layers), bins=lbins, weights=mean_layer,
                 histtype='step', lw=1.5, color='steelblue')
    axes[0].set(xlabel="Layer (depth)", ylabel="Mean energy [GeV]",
                title="Longitudinal profile",
                xlim=(0, geom.num_layers+1), xticks=np.arange(geom.num_layers))
    axes[0].set_ylim(bottom=0)

    # Lateral: energy vs transverse distance from shower axis
    all_d_perp, all_e_perp = [], []
    for i in range(n_events):
        xi = np.ma.getdata(ak.to_numpy(pos_x[i]))
        yi = np.ma.getdata(ak.to_numpy(pos_y[i]))
        zi = np.ma.getdata(ak.to_numpy(pos_z[i]))
        ei = np.ma.getdata(ak.to_numpy(pos_e[i]))
        if len(ei) == 0:
            continue
        w = ei / ei.sum()
        cx, cy, cz = np.dot(w, xi), np.dot(w, yi), np.dot(w, zi)
        norm = np.sqrt(cx**2 + cy**2 + cz**2)
        if norm < 1e-6:
            continue
        ux, uy, uz = cx / norm, cy / norm, cz / norm
        t = xi * ux + yi * uy + zi * uz
        d_perp = np.sqrt((xi - t * ux)**2 + (yi - t * uy)**2 + (zi - t * uz)**2)
        all_d_perp.append(d_perp)
        all_e_perp.append(ei)

    d_flat = np.concatenate(all_d_perp)
    e_lat = np.concatenate(all_e_perp)
    d_bins = np.linspace(0, np.percentile(d_flat, 99), 40)
    mean_lat, _ = np.histogram(d_flat, bins=d_bins, weights=e_lat)
    mean_lat /= n_events

    axes[1].bar(0.5 * (d_bins[:-1] + d_bins[1:]), mean_lat,
                width=np.diff(d_bins), color='seagreen', alpha=0.8, edgecolor='none')
    axes[1].set(xlabel="Transverse distance [mm]", ylabel="Mean energy [GeV]",
                title="Lateral profile")
    axes[1].set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(output_dir / "allegro_profiles.png"); plt.close()

    # =====================================================================
    # Figure 5: Gun position & angular diagnostics  (2x2)
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f"Gun Position & Angular Diagnostics  ({n_events} events)")

    # (0,0) Gun position x-y — ring at r_min, coloured by theta
    sc = axes[0, 0].scatter(vx, vy, c=theta_gun, cmap='plasma', s=80, zorder=3)
    axes[0, 0].add_patch(plt.Circle((0, 0), geom.r_min, fill=False,
                                     color='black', ls='--', lw=1.5,
                                     label=f'R_min={geom.r_min:.0f} mm'))
    plt.colorbar(sc, ax=axes[0, 0], label='θ [rad]')
    axes[0, 0].set(xlabel="x [mm]", ylabel="y [mm]",
                   title="Gun position x-y  (color = θ)", aspect='equal')
    axes[0, 0].legend(fontsize=14)

    # (0,1) z_gun vs theta — should follow z = R_min * cot(θ)
    theta_line = np.linspace(THETA_MIN, THETA_MAX, 300)
    z_line     = geom.r_min * np.cos(theta_line) / np.sin(theta_line)
    axes[0, 1].scatter(theta_gun, vz, c='steelblue', s=80, zorder=3, label='MC vertex')
    axes[0, 1].plot(theta_line, z_line, 'r--', lw=1.5,
                    label=r'$z = R_{\min}\cot\theta$')
    axes[0, 1].axvline(THETA_MIN, color='grey', ls=':', lw=1)
    axes[0, 1].axvline(THETA_MAX, color='grey', ls=':', lw=1)
    axes[0, 1].set(xlabel="θ [rad]", ylabel="z_gun [mm]",
                   title="Gun z vs θ")
    axes[0, 1].legend(fontsize=14)

    # (1,0) theta distribution — uniform cos(theta) sampling
    theta_bins = np.linspace(0, np.pi, 40)
    axes[1, 0].hist(theta_gun, bins=theta_bins, histtype='stepfilled',
                    color='coral', alpha=0.8, edgecolor='darkred')
    axes[1, 0].axvline(THETA_MIN, color='black', ls='--', lw=1.5,
                       label=f'θ_min={THETA_MIN} rad')
    axes[1, 0].axvline(THETA_MAX, color='black', ls=':',  lw=1.5,
                       label=f'θ_max={THETA_MAX} rad')
    axes[1, 0].set(xlabel="θ [rad]", ylabel="Events",
                   title="Polar angle distribution")
    axes[1, 0].legend(fontsize=14)

    # (1,1) phi distribution — should be uniform in [0, 2π]
    phi_plot = phi_gun % (2 * np.pi)
    phi_bins = np.linspace(0, 2 * np.pi, 20)
    axes[1, 1].hist(phi_plot, bins=phi_bins, histtype='stepfilled',
                    color='mediumpurple', alpha=0.8, edgecolor='indigo')
    axes[1, 1].set(xlabel="φ [rad]", ylabel="Events",
                   title="Azimuthal angle distribution",
                   xticks=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
                   xticklabels=['0', 'π/2', 'π', '3π/2', '2π'])

    plt.tight_layout()
    fig.savefig(output_dir / "allegro_gun_diagnostics.png"); plt.close()

    # =====================================================================
    # Statistics report
    # =====================================================================
    hits_inside = np.sum((rho_flat >= geom.r_min) & (rho_flat <= geom.r_max))
    total_mean = mean_layer.sum()

    layer_table = f"{'Layer':>6} {'R_low':>10} {'R_high':>10} {'Mean E':>12} {'Frac':>8}\n"
    for l in range(geom.num_layers):
        frac = mean_layer[l] / total_mean if total_mean > 0 else 0
        layer_table += f"{l+1:>6} {geom.layer_boundaries[l]:>10.1f} {geom.layer_boundaries[l+1]:>10.1f} {mean_layer[l]:>12.6f} {frac:>8.4f}\n"
    layer_table += f"{'Total':>6} {geom.r_min:>10.1f} {geom.r_max:>10.1f} {total_mean:>12.6f} {'1.0000':>8}"

    z_gun_expected_min = geom.r_min * np.cos(THETA_MAX) / np.sin(THETA_MAX)
    z_gun_expected_max = geom.r_min * np.cos(THETA_MIN) / np.sin(THETA_MIN)

    evt_table = (f"{'Event':>6} {'E_inc':>10} {'E_dep':>10} {'SF':>7}"
                 f" {'theta':>7} {'phi':>7} {'r_gun':>9} {'z_gun':>9}"
                 f" {'Cells':>7} {'Steps':>8}\n")
    for i in range(n_events):
        evt_table += (f"{i:>6} {e_inc[i]:>10.3f} {cell_etot[i]:>10.3f} {ratio[i]:>7.4f}"
                      f" {theta_gun[i]:>7.4f} {phi_gun[i]:>7.4f} {r_gun[i]:>9.2f} {vz[i]:>9.2f}"
                      f" {nhits_cells[i]:>7} {nhits_steps[i]:>8}\n")

    report = f"""{'='*70}
ALLEGRO ECal Dataset - Statistics Report
{'='*70}
ROOT file:    {root_file.name}
Hit type:     {data_label}
Geometry:     ALLEGRO_o1_v03

{'-'*70}
GEOMETRY
{'-'*70}
R_min / R_max:      {geom.r_min:.1f} / {geom.r_max:.1f} mm
Layers:             {geom.num_layers}
Depth:              {geom.r_max - geom.r_min:.1f} mm

{'-'*70}
EVENT SUMMARY
{'-'*70}
Events:             {n_events}
G4 steps:           {nhits_steps.sum()} (mean {nhits_steps.mean():.0f}/evt)
Cells:              {nhits_cells.sum()} (mean {nhits_cells.mean():.0f}/evt)
Steps/cell:         {nhits_steps.sum()/max(nhits_cells.sum(),1):.1f}

{'-'*70}
INCIDENT ENERGY [GeV]
{'-'*70}
Range:              [{e_inc.min():.2f}, {e_inc.max():.2f}]
Mean / Median:      {e_inc.mean():.2f} / {np.median(e_inc):.2f}

{'-'*70}
SAMPLING FRACTION
{'-'*70}
Mean / Std:         {ratio.mean():.4f} / {ratio.std():.4f}
Range:              [{ratio.min():.4f}, {ratio.max():.4f}]

{'-'*70}
GUN POSITION
{'-'*70}
r_gun [mm]:         mean={r_gun.mean():.4f}  std={r_gun.std():.4f}
                    expected = R_min - 1e-8 = {geom.r_min - 1e-8:.4f}
  delta (r-R_min):  [{(r_gun-geom.r_min).min():.4f}, {(r_gun-geom.r_min).max():.4f}]
z_gun [mm]:         [{vz.min():.2f}, {vz.max():.2f}]
                    expected: [{z_gun_expected_min:.2f}, {z_gun_expected_max:.2f}]
z vs formula check: max |z_gun - R_min*cot(θ)| = {np.abs(vz - z_expected).max():.2e} mm

{'-'*70}
ANGULAR DISTRIBUTION (LEMURS range: θ ∈ [0.87, 2.27] rad)
{'-'*70}
θ [rad]:            [{theta_gun.min():.4f}, {theta_gun.max():.4f}]  target [{THETA_MIN}, {THETA_MAX}]
cos θ:              [{cos_theta.min():.4f}, {cos_theta.max():.4f}]  target [{np.cos(THETA_MAX):.4f}, {np.cos(THETA_MIN):.4f}]
φ [rad]:            [{phi_gun.min():.4f}, {phi_gun.max():.4f}]   (wrapped to [0,2π]: uniform expected)
All θ in range:     {np.all((theta_gun >= THETA_MIN) & (theta_gun <= THETA_MAX))}

{'-'*70}
SPATIAL EXTENT OF HITS [mm]
{'-'*70}
x:  [{x_flat.min():.1f}, {x_flat.max():.1f}]
y:  [{y_flat.min():.1f}, {y_flat.max():.1f}]
z:  [{z_flat.min():.1f}, {z_flat.max():.1f}]
rho:[{rho_flat.min():.1f}, {rho_flat.max():.1f}]
Inside barrel:      {hits_inside}/{len(rho_flat)} ({100*hits_inside/len(rho_flat):.1f}%)
  Leaking rho>R_max:{np.sum(rho_flat > geom.r_max)} steps ({100*np.sum(rho_flat > geom.r_max)/len(rho_flat):.2f}%)
  Below  rho<R_min: {np.sum(rho_flat < geom.r_min)} steps ({100*np.sum(rho_flat < geom.r_min)/len(rho_flat):.2f}%)

{'-'*70}
LONGITUDINAL PROFILE (per layer)
{'-'*70}
{layer_table}

{'-'*70}
PER-EVENT TABLE (E in GeV, r/z in mm, θ/φ in rad)
{'-'*70}
{evt_table}{'='*70}
"""
    (output_dir / "allegro_dataset_statistics.txt").write_text(report)
    print(f"Saved plots and stats to {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--root-file', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    plot_allegro_diagnostics(args.root_file, args.output_dir)