#!/usr/bin/env python3
"""
Plot a single ALLEGRO ECal shower in global (x,y,z) coordinates
to visualise the accordion calorimeter geometry.

Uses the raw step-level ROOT contributions (HitCreationMode=2) so the
hit positions trace the actual path through the folded absorbers/LAr.

Environment:  source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2025-05-29

Usage:
    python plot_allegro_accordion.py \
        --root output_dataset/ALLEGRO/root/final/allegro_final_0001.root \
        --event 5 \
        --out output_dataset/ALLEGRO/accordion_shower.png
"""

import argparse
import sys
from pathlib import Path

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import uproot

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.calo_geometry import ALLEGROGeometry

CONTRIB = "ECalBarrelModuleThetaMergedContributions"
CP      = f"{CONTRIB}/{CONTRIB}"


def pick_best_event(tree, geo, n_check=50):
    """Pick the event with the most barrel hits (most interesting to plot)."""
    contribs = tree.arrays([
        f"{CP}.stepPosition.x", f"{CP}.stepPosition.y",
        f"{CP}.stepPosition.z", f"{CP}.energy",
    ], library="ak", entry_stop=n_check)

    best_evt, best_n = 0, 0
    for i in range(min(n_check, len(contribs[f"{CP}.energy"]))):
        x = np.array(contribs[f"{CP}.stepPosition.x"][i])
        y = np.array(contribs[f"{CP}.stepPosition.y"][i])
        rho = np.sqrt(x**2 + y**2)
        n_barrel = int(((rho >= geo.r_min) & (rho <= geo.r_max)).sum())
        if n_barrel > best_n:
            best_n, best_evt = n_barrel, i
    return best_evt


def load_event(tree, geo, event_idx):
    """Load a single event's step contributions inside the barrel."""
    contribs = tree.arrays([
        f"{CP}.stepPosition.x", f"{CP}.stepPosition.y",
        f"{CP}.stepPosition.z", f"{CP}.energy",
        "MCParticles/MCParticles.momentum.x",
        "MCParticles/MCParticles.momentum.y",
        "MCParticles/MCParticles.momentum.z",
        "MCParticles/MCParticles.generatorStatus",
        "MCParticles/MCParticles.vertex.x",
        "MCParticles/MCParticles.vertex.y",
        "MCParticles/MCParticles.vertex.z",
    ], library="ak", entry_start=event_idx, entry_stop=event_idx + 1)

    x = np.array(contribs[f"{CP}.stepPosition.x"][0])
    y = np.array(contribs[f"{CP}.stepPosition.y"][0])
    z = np.array(contribs[f"{CP}.stepPosition.z"][0])
    e = np.array(contribs[f"{CP}.energy"][0])

    # barrel filter
    rho = np.sqrt(x**2 + y**2)
    mask = (rho >= geo.r_min) & (rho <= geo.r_max) & (e > 0)
    x, y, z, e, rho = x[mask], y[mask], z[mask], e[mask], rho[mask]

    # gun info — primary particle (generatorStatus == 1)
    status  = np.array(ak.flatten(contribs["MCParticles/MCParticles.generatorStatus"][0], axis=None))
    primary = status == 1
    px = float(np.array(ak.flatten(contribs["MCParticles/MCParticles.momentum.x"][0], axis=None))[primary][0])
    py = float(np.array(ak.flatten(contribs["MCParticles/MCParticles.momentum.y"][0], axis=None))[primary][0])
    pz = float(np.array(ak.flatten(contribs["MCParticles/MCParticles.momentum.z"][0], axis=None))[primary][0])
    gx = float(np.array(ak.flatten(contribs["MCParticles/MCParticles.vertex.x"][0], axis=None))[primary][0])
    gy = float(np.array(ak.flatten(contribs["MCParticles/MCParticles.vertex.y"][0], axis=None))[primary][0])
    gz = float(np.array(ak.flatten(contribs["MCParticles/MCParticles.vertex.z"][0], axis=None))[primary][0])

    E_inc = np.sqrt(px**2 + py**2 + pz**2)
    phi   = np.arctan2(py, px)
    theta = np.arccos(np.clip(pz / E_inc, -1, 1))

    return dict(x=x, y=y, z=z, e=e, rho=rho,
                gx=gx, gy=gy, gz=gz,
                E_inc=E_inc, phi_gun=np.degrees(phi), theta_gun=np.degrees(theta))


def layer_of(rho, geo):
    """Map rho → layer index (0-indexed)."""
    layers = np.full(len(rho), geo.num_layers - 1, dtype=int)
    for l in range(geo.num_layers - 1):
        mask = rho < geo.layer_boundaries[l + 1]
        layers[mask & (layers == geo.num_layers - 1)] = l
    return layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",    required=True)
    ap.add_argument("--event",   type=int, default=None,
                    help="Event index (default: auto-pick most hits)")
    ap.add_argument("--out",     default="accordion_shower.png")
    args = ap.parse_args()

    geo  = ALLEGROGeometry(version="v03")
    tree = uproot.open(args.root)["events"]
    print(f"Opened: {args.root}  ({len(tree)} events)")

    evt_idx = args.event
    if evt_idx is None:
        evt_idx = pick_best_event(tree, geo)
        print(f"Auto-selected event {evt_idx} (most barrel hits)")
    else:
        print(f"Using event {evt_idx}")

    d = load_event(tree, geo, evt_idx)
    x, y, z, e, rho = d["x"], d["y"], d["z"], d["e"], d["rho"]
    layers = layer_of(rho, geo)
    phi_hit = np.arctan2(y, x)

    print(f"  {len(e):,} barrel hits   E_inc={d['E_inc']:.1f} GeV")
    print(f"  θ={d['theta_gun']:.1f}°  φ={d['phi_gun']:.1f}°")
    print(f"  E_dep={e.sum()*1e3:.1f} MeV   rho [{rho.min():.0f}, {rho.max():.0f}] mm")

    phi_gun_rad = np.radians(d["phi_gun"])
    phi_rel     = phi_hit - phi_gun_rad          # angular offset from gun [rad]
    arc         = rho * phi_rel                  # arc-length offset [mm]

    # ── Publication-ready font settings ───────────────────────────────────────
    FS_LABEL  = 18
    FS_TITLE  = 17
    FS_TICK   = 15
    FS_CB     = 15
    FS_LGND   = 14
    FS_ANNOT  = 13   # in-plot layer labels
    FS_STRIP  = 15   # bottom strip panels

    plt.rcParams.update({
        "font.family":      "serif",
        "font.size":        16,
        "axes.labelsize":   FS_LABEL,
        "axes.titlesize":   FS_TITLE,
        "xtick.labelsize":  FS_TICK,
        "ytick.labelsize":  FS_TICK,
        "legend.fontsize":  FS_LGND,
    })

    cmap     = "inferno"
    VMIN_GLB = -6     # global lower bound for log10 colorbar

    fig = plt.figure(figsize=(22, 22))
    fig.suptitle(
        f"ALLEGRO ECal accordion — event {evt_idx}  "
        f"$E_{{\\rm inc}}$={d['E_inc']:.1f} GeV,  "
        f"$\\theta$={d['theta_gun']:.1f}°,  $\\phi$={d['phi_gun']:.1f}°\n"
        f"{len(e):,} barrel steps,  $E_{{\\rm dep}}$={e.sum()*1e3:.0f} MeV",
        fontsize=19, y=0.995,
    )

    # Layout: 3 rows × 4 cols
    #   Rows 0–1 col 0–1 : main unrolled accordion (full height)
    #   Row  0   col 2   : x–y zoom L4
    #   Row  0   col 3   : x–y zoom L6
    #   Row  1   col 2–3 : r–z longitudinal
    #   Row  2   col 0–3 : per-layer zoomed strips
    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.52, wspace=0.40,
                           height_ratios=[1, 1, 0.85])

    r_bins = np.linspace(geo.r_min, geo.r_max, 300)

    def add_cb(im, ax, label):
        cb = plt.colorbar(im, ax=ax, pad=0.01)
        cb.set_label(label, fontsize=FS_CB)
        cb.ax.tick_params(labelsize=FS_TICK)
        return cb

    def clipped_vmin(H_log):
        """Return vmin clipped to VMIN_GLB, vmax from data."""
        valid = H_log[~np.isnan(H_log)]
        if len(valid) == 0:
            return VMIN_GLB, 0.0
        return max(VMIN_GLB, float(valid.min())), float(valid.max())

    # ── 1.  Unrolled accordion: r·Δφ vs r  (MAIN VIEW) ───────────────────────
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    arc_lim  = 250
    arc_bins = np.linspace(-arc_lim, arc_lim, 600)

    H, xe, ye = np.histogram2d(arc, rho, bins=[arc_bins, r_bins], weights=e)
    H_log = np.where(H > 0, np.log10(H + 1e-12), np.nan)
    vmin1, vmax1 = clipped_vmin(H_log)
    im1 = ax1.pcolormesh(xe, ye, H_log.T, cmap=cmap, shading="auto",
                         vmin=vmin1, vmax=vmax1)
    for r in geo.layer_boundaries:
        ax1.axhline(r, color="white", lw=1.0, ls="--", alpha=0.65)
    for i, r in enumerate(0.5*(geo.layer_boundaries[:-1]+geo.layer_boundaries[1:])):
        ax1.text(arc_lim - 12, r, f"L{i}", color="white", fontsize=FS_ANNOT,
                 va="center", ha="right", fontweight="bold")
    ax1.axvline(0, color="cyan", lw=1.5, ls=":", label="Shower axis")
    ax1.set_xlabel("Arc-length offset  $r\\cdot\\Delta\\phi$  [mm]")
    ax1.set_ylabel("Radius  $r$  [mm]")
    ax1.set_title("Unrolled accordion view  ($r$ vs $r\\cdot\\Delta\\phi$)\n"
                  "Bright = LAr energy deposit  ·  Dark = Pb absorbers")
    ax1.legend(loc="upper right")
    add_cb(im1, ax1, "$\\log_{10}(\\Sigma E\\ [\\mathrm{GeV}])$")

    # helper: draw layer-boundary arcs on an x–y panel (centre = lab origin)
    def draw_layer_arcs(ax, r_boundaries, phi_center_rad, win_mm):
        from matplotlib.patches import Arc as MArc
        phi_deg  = np.degrees(phi_center_rad)
        for r_bnd in r_boundaries:
            ang_half = np.degrees(win_mm / r_bnd) * 1.6   # slightly wider than viewport
            patch = MArc((0, 0), 2*r_bnd, 2*r_bnd,
                         angle=0,
                         theta1=phi_deg - ang_half,
                         theta2=phi_deg + ang_half,
                         color="white", lw=1.5, ls="--", alpha=0.9, zorder=5)
            ax.add_patch(patch)

    def xy_layers_panel(ax, l_start, l_end, win=200, nbins=400):
        """Show layers l_start..l_end (inclusive) in x-y with all boundary arcs."""
        r_lo  = geo.layer_boundaries[l_start]
        r_hi  = geo.layer_boundaries[l_end + 1]
        r_mid = 0.5 * (r_lo + r_hi)
        cx    = r_mid * np.cos(phi_gun_rad)
        cy    = r_mid * np.sin(phi_gun_rad)
        mask  = (rho >= r_lo) & (rho <= r_hi)
        Hxy, xexy, yexy = np.histogram2d(
            x[mask], y[mask],
            bins=[np.linspace(cx - win, cx + win, nbins),
                  np.linspace(cy - win, cy + win, nbins)],
            weights=e[mask])
        Hxy_log = np.where(Hxy > 0, np.log10(Hxy + 1e-12), np.nan)
        vm, vM  = clipped_vmin(Hxy_log)
        im = ax.pcolormesh(xexy, yexy, Hxy_log.T, cmap=cmap, shading="auto",
                           vmin=vm, vmax=vM)
        draw_layer_arcs(ax, list(geo.layer_boundaries[l_start: l_end + 2]),
                        phi_gun_rad, win)
        ax.set_xlim(cx - win, cx + win)
        ax.set_ylim(cy - win, cy + win)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$  [mm]")
        ax.set_ylabel("$y$  [mm]")
        ax.set_title(f"$x$–$y$ zoom — Layers {l_start}–{l_end}  "
                     f"($r$ = {r_lo:.0f}–{r_hi:.0f} mm)\n"
                     "Dashed arcs = layer boundaries")
        add_cb(im, ax, "$\\log_{10}(\\Sigma E\\ [\\mathrm{GeV}])$")
        return cx, cy   # expose centre for title overrides

    # ── 2.  x–y zoom — Layers 4–6, full height (same as main accordion) ────────
    ax2 = fig.add_subplot(gs[0:2, 2:4])
    xy_layers_panel(ax2, 4, 6)

    # ── 4.  Per-layer zoomed accordion strips ─────────────────────────────────
    strip_layers = np.linspace(0, geo.num_layers - 1, 4, dtype=int)
    arc_zoom  = 60
    arc_zbins = np.linspace(-arc_zoom, arc_zoom, 300)

    for col, li in enumerate(strip_layers):
        ax = fig.add_subplot(gs[2, col])
        r_lo_s   = geo.layer_boundaries[li]
        r_hi_s   = geo.layer_boundaries[li + 1]
        r_bins_s = np.linspace(r_lo_s, r_hi_s, 80)
        mask_s   = (rho >= r_lo_s) & (rho <= r_hi_s)
        arc_s, rho_s, e_s = arc[mask_s], rho[mask_s], e[mask_s]

        if len(e_s) > 0:
            Hs, xes, yes = np.histogram2d(arc_s, rho_s,
                                           bins=[arc_zbins, r_bins_s],
                                           weights=e_s)
            Hs_log = np.where(Hs > 0, np.log10(Hs + 1e-12), np.nan)
            vmins, vmaxs = clipped_vmin(Hs_log)
            ims = ax.pcolormesh(xes, yes, Hs_log.T, cmap=cmap, shading="auto",
                                vmin=vmins, vmax=vmaxs)
            cbs = plt.colorbar(ims, ax=ax, pad=0.01, fraction=0.046)
            cbs.set_label("$\\log_{10}(E)$", fontsize=FS_CB - 1)
            cbs.ax.tick_params(labelsize=FS_TICK - 1)
        ax.axvline(0, color="cyan", lw=1.0, ls=":")
        ax.set_xlabel("$r\\cdot\\Delta\\phi$  [mm]", fontsize=FS_STRIP)
        ax.set_ylabel("$r$  [mm]", fontsize=FS_STRIP)
        ax.set_title(f"Layer {li}  ($r$ = {r_lo_s:.0f}–{r_hi_s:.0f} mm)\n"
                     "Accordion fold — zoomed", fontsize=FS_STRIP)
        ax.tick_params(labelsize=FS_TICK)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # ── FIGURE 2: progressive zoom into 1×1 mm cells ──────────────────────────
    # Goal: show individual cells as clearly visible colored squares.
    # Three zoom levels: overview → ±40 mm corridor → ±12 mm ultra-close per layer
    fig2 = plt.figure(figsize=(24, 20))
    fig2.suptitle(
        f"ALLEGRO ECal — shower projected into 1×1 mm cells — event {evt_idx}  "
        f"$E_{{\\rm inc}}$={d['E_inc']:.1f} GeV,  "
        f"$\\theta$={d['theta_gun']:.1f}°,  $\\phi$={d['phi_gun']:.1f}°",
        fontsize=19, y=0.995,
    )
    # 2-row layout: top = overview + zoom-in corridor, bottom = 4 per-layer close-ups
    gs2 = gridspec.GridSpec(2, 4, figure=fig2,
                            hspace=0.45, wspace=0.38,
                            height_ratios=[1.6, 1])

    arc_bins_1mm = np.arange(-arc_lim, arc_lim + 1.0, 1.0)
    r_bins_1mm   = np.arange(geo.r_min, geo.r_max + 1.0, 1.0)
    Hg, xeg, yeg = np.histogram2d(arc, rho,
                                   bins=[arc_bins_1mm, r_bins_1mm], weights=e)
    Hg_log = np.where(Hg > 0, np.log10(Hg + 1e-12), np.nan)
    vming, vmaxg = clipped_vmin(Hg_log)

    # ── G1: overview unrolled, 1 mm bins (top-left, 2 cols) ───────────────────
    axG1 = fig2.add_subplot(gs2[0, 0:2])
    imG1 = axG1.pcolormesh(xeg, yeg, Hg_log.T, cmap=cmap, shading="auto",
                            vmin=vming, vmax=vmaxg)
    for r in geo.layer_boundaries:
        axG1.axhline(r, color="white", lw=0.8, ls="--", alpha=0.6)
    for i, r in enumerate(0.5*(geo.layer_boundaries[:-1]+geo.layer_boundaries[1:])):
        axG1.text(arc_lim - 12, r, f"L{i}", color="white", fontsize=FS_ANNOT,
                  va="center", ha="right", fontweight="bold")
    axG1.axvline(0, color="cyan", lw=1.2, ls=":", label="Shower axis")
    # mark the ±40 mm corridor zoomed in panel G2
    for sign in (-1, 1):
        axG1.axvline(sign * 40, color="lime", lw=1.2, ls="--", alpha=0.8)
    axG1.set_xlabel("Arc-length offset  $r\\cdot\\Delta\\phi$  [mm]")
    axG1.set_ylabel("Radius  $r$  [mm]")
    axG1.set_title("Overview — 1×1 mm cell projection  (full shower)\n"
                   "Green dashed = region zoomed in right panel")
    axG1.legend(loc="upper right", fontsize=FS_LGND)
    add_cb(imG1, axG1, "$\\log_{10}(\\Sigma E\\ [\\mathrm{GeV}])$")

    # ── G2: corridor zoom ±40 mm — cells become distinguishable (top-right) ───
    axG2 = fig2.add_subplot(gs2[0, 2:4])
    zoom_arc = 40
    # reuse the full 1mm histogram, just restrict x range:
    imG2 = axG2.pcolormesh(xeg, yeg, Hg_log.T, cmap=cmap, shading="auto",
                            vmin=vming, vmax=vmaxg)
    for r in geo.layer_boundaries:
        axG2.axhline(r, color="white", lw=0.8, ls="--", alpha=0.6)
    for i, r in enumerate(0.5*(geo.layer_boundaries[:-1]+geo.layer_boundaries[1:])):
        axG2.text(zoom_arc - 2, r, f"L{i}", color="white", fontsize=FS_ANNOT,
                  va="center", ha="right", fontweight="bold")
    axG2.axvline(0, color="cyan", lw=1.2, ls=":", label="Shower axis")
    axG2.set_xlim(-zoom_arc, zoom_arc)
    axG2.set_xlabel("Arc-length offset  $r\\cdot\\Delta\\phi$  [mm]")
    axG2.set_ylabel("Radius  $r$  [mm]")
    axG2.set_title(f"Zoomed corridor — $|r\\cdot\\Delta\\phi|$ ≤ {zoom_arc} mm\n"
                   "Individual 1 mm cells become visible")
    axG2.legend(loc="upper right", fontsize=FS_LGND)
    add_cb(imG2, axG2, "$\\log_{10}(\\Sigma E\\ [\\mathrm{GeV}])$")

    # ── G3-G6: ultra-close per-layer strips ±12 mm — cells are clear squares ──
    # 12 mm arc × ~37 mm r per layer → cells are ~20–30 px wide in output
    close_arc  = 12
    close_bins = np.arange(-close_arc, close_arc + 1.0, 1.0)   # 24 bins
    for col, li in enumerate(strip_layers):
        ax = fig2.add_subplot(gs2[1, col])
        r_lo_s   = geo.layer_boundaries[li]
        r_hi_s   = geo.layer_boundaries[li + 1]
        r_bins_s = np.arange(r_lo_s, r_hi_s + 1.0, 1.0)
        mask_s   = (rho >= r_lo_s) & (rho <= r_hi_s)
        arc_s, rho_s, e_s = arc[mask_s], rho[mask_s], e[mask_s]
        if len(e_s) > 0:
            Hs, xes, yes = np.histogram2d(arc_s, rho_s,
                                           bins=[close_bins, r_bins_s],
                                           weights=e_s)
            Hs_log = np.where(Hs > 0, np.log10(Hs + 1e-12), np.nan)
            vmins, vmaxs = clipped_vmin(Hs_log)
            ims = ax.pcolormesh(xes, yes, Hs_log.T, cmap=cmap, shading="auto",
                                vmin=vmins, vmax=vmaxs)
            cbs = plt.colorbar(ims, ax=ax, pad=0.01, fraction=0.046)
            cbs.set_label("$\\log_{10}(E\\ [\\mathrm{GeV}])$",
                          fontsize=FS_CB - 1)
            cbs.ax.tick_params(labelsize=FS_TICK - 1)
        ax.set_aspect("equal")   # ensures cells appear as true squares
        ax.axvline(0, color="cyan", lw=1.0, ls=":")
        ax.set_xlabel("$r\\cdot\\Delta\\phi$  [mm]", fontsize=FS_STRIP)
        ax.set_ylabel("$r$  [mm]", fontsize=FS_STRIP)
        ax.set_title(f"Layer {li}  ($r$ = {r_lo_s:.0f}–{r_hi_s:.0f} mm)\n"
                     f"$\\pm${close_arc} mm — each square = 1 mm cell",
                     fontsize=FS_STRIP)
        ax.tick_params(labelsize=FS_TICK)

    fig2.tight_layout()
    out2 = out.parent / (out.stem + "_1mm_cells" + out.suffix)
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
