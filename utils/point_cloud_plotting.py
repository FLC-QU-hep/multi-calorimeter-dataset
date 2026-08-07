import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

def plot_dataset_overview(data, geometry=None, figsize=(18, 12)):
    """
    Overview plots for entire dataset - showing complete distributions of ALL hits.
    
    Parameters:
    -----------
    data : dict
        Point cloud data from load_pointcloud_h5
    geometry : CaloGeometry or None
        Geometry parameters (optional, for reference lines)
    """
    points = data['events']
    n_points = data['n_points']
    total_energy = data['total_energy']
    incident_energy = data.get('incident_energy', None)
    n_showers = len(n_points)
    
    # Flatten all showers into one big point cloud
    all_hits = []
    for i in range(n_showers):
        n = n_points[i]
        all_hits.append(points[i, :n])
    all_hits = np.vstack(all_hits)  # Shape: (total_hits, 5)
    
    x_all = all_hits[:, 0]
    y_all = all_hits[:, 1]
    z_all = all_hits[:, 2]
    t_all = all_hits[:, 3]
    e_all = all_hits[:, 4]
    
    layer_all = z_all.astype(int)
    r_all = np.sqrt(y_all**2 + z_all**2)
    
    # Energy normalization for coloring
    norm = mcolors.LogNorm(vmin=max(e_all.min()*1e3, 0.01), vmax=e_all.max()*1e3)
    
    fig = plt.figure(figsize=figsize)
    fig.suptitle(f"Dataset Overview - {n_showers} Showers, {len(e_all):,} Total Hits", 
                 fontsize=14, fontweight='bold')
    
    # --- 1. Hit Energy Spectrum ---
    ax1 = plt.subplot(3, 4, 1)
    ax1.hist(e_all * 1e3, bins=np.logspace(-2, np.log10(e_all.max()*1e3), 100), 
             histtype='step', log=True, linewidth=1.5)
    ax1.set_xscale('log')
    ax1.set_xlabel("Hit Energy [MeV]")
    ax1.set_ylabel("Counts")
    ax1.set_title("Hit Energy Spectrum (All Hits)")
    ax1.grid(True, alpha=0.3)
    
    # --- 2. Total Energy per Event ---
    ax2 = plt.subplot(3, 4, 2)
    ax2.hist(total_energy, bins=50, histtype='step', linewidth=1.5, color='blue')
    ax2.set_xlabel("Energy sum per Event [GeV]")
    ax2.set_ylabel("Events")
    ax2.set_title("Energy Sum Distribution")
    ax2.grid(True, alpha=0.3)
    
    # --- 3. Hit Multiplicity ---
    ax3 = plt.subplot(3, 4, 3)
    ax3.hist(n_points, bins=100, histtype='step', linewidth=1.5, color='green')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel("Number of Hits per Event")
    ax3.set_ylabel("Events")
    ax3.set_title("Hit Multiplicity")
    ax3.grid(True, alpha=0.3)
    
    # --- 4. Transverse View (x-y) ---
    ax4 = plt.subplot(3, 4, 4)
    # Subsample for performance if too many points
    if len(e_all) > 100000:
        idx = np.random.choice(len(e_all), 100000, replace=False)
        sc = ax4.scatter(x_all[idx], y_all[idx], c=e_all[idx]*1e3, s=0.5, 
                        cmap='plasma', alpha=0.3, norm=norm)
    else:
        sc = ax4.scatter(x_all, y_all, c=e_all*1e3, s=1, 
                        cmap='plasma', alpha=0.5, norm=norm)
    ax4.set_xlabel("x [mm]")
    ax4.set_ylabel("y [mm]")
    ax4.set_title("Transverse View (All Showers)")
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax4, label="E [MeV]")
    
    # --- 5. Hits colored by Layer ---
    ax5 = plt.subplot(3, 4, 5)
    if geometry is not None:
        vmax_layer = geometry.num_layers - 1
    else:
        vmax_layer = layer_all.max()
    
    if len(e_all) > 100000:
        sc2 = ax5.scatter(x_all[idx], y_all[idx], c=layer_all[idx], s=0.5, 
                         cmap='viridis', alpha=0.3, vmin=0, vmax=vmax_layer)
    else:
        sc2 = ax5.scatter(x_all, y_all, c=layer_all, s=1, 
                         cmap='viridis', alpha=0.5, vmin=0, vmax=vmax_layer)
    ax5.set_xlabel("x [mm]")
    ax5.set_ylabel("y [mm]")
    ax5.set_title("Hits Colored by Layer")
    ax5.grid(True, alpha=0.3)
    plt.colorbar(sc2, ax=ax5, label="Layer Number")
    
    # --- 6. x-z View ---
    ax6 = plt.subplot(3, 4, 6)
    if len(e_all) > 100000:
        sc3 = ax6.scatter(x_all[idx], z_all[idx], c=e_all[idx]*1e3, s=0.5, 
                         cmap='plasma', alpha=0.3, norm=norm)
    else:
        sc3 = ax6.scatter(x_all, z_all, c=e_all*1e3, s=1, 
                         cmap='plasma', alpha=0.5, norm=norm)
    ax6.set_xlabel("x [mm]")
    ax6.set_ylabel("z [mm]")
    ax6.set_title("x-z View (All Showers)")
    ax6.grid(True, alpha=0.3)
    plt.colorbar(sc3, ax=ax6, label="E [MeV]")
    
    # --- 7. Layer vs x Position ---
    ax7 = plt.subplot(3, 4, 7)
    if len(e_all) > 100000:
        sc4 = ax7.scatter(x_all[idx], layer_all[idx], c=e_all[idx]*1e3, s=0.5, 
                         cmap='plasma', alpha=0.3, norm=norm)
    else:
        sc4 = ax7.scatter(x_all, layer_all, c=e_all*1e3, s=1, 
                         cmap='plasma', alpha=0.5, norm=norm)
    ax7.set_xlabel("x [mm]")
    ax7.set_ylabel("Layer Number")
    ax7.set_title("Longitudinal Development (All)")
    if geometry is not None:
        ax7.set_ylim(-0.5, geometry.num_layers - 0.5)
    ax7.grid(True, alpha=0.3)
    plt.colorbar(sc4, ax=ax7, label="E [MeV]")
    
    # --- 8. Energy per Layer ---
    ax8 = plt.subplot(3, 4, 8)
    unique_layers = np.unique(layer_all)
    energy_per_layer = np.array([e_all[layer_all == lay].sum() for lay in unique_layers])
    ax8.bar(unique_layers, energy_per_layer, color='orangered', alpha=0.7, 
            edgecolor='black', linewidth=0.5)
    shower_max = unique_layers[np.argmax(energy_per_layer)]
    ax8.axvline(shower_max, color='blue', linestyle='--', linewidth=2, 
                label=f'Shower max: layer {shower_max}')
    ax8.set_xlabel("Layer Number")
    ax8.set_ylabel("Total Energy [GeV]")
    ax8.set_title("Energy Profile (All Showers)")
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # --- 9. Hits per Layer ---
    ax9 = plt.subplot(3, 4, 9)
    hits_per_layer = np.array([np.sum(layer_all == lay) for lay in unique_layers])
    ax9.bar(unique_layers, hits_per_layer, color='forestgreen', alpha=0.7, 
            edgecolor='black', linewidth=0.5)
    ax9.set_xlabel("Layer Number")
    ax9.set_ylabel("Number of Hits")
    ax9.set_title("Hit Multiplicity per Layer (All)")
    ax9.grid(True, alpha=0.3, axis='y')
    
    # --- 10. Radius Distribution ---
    ax10 = plt.subplot(3, 4, 10)
    ax10.hist(r_all, bins=100, histtype='step', linewidth=1.5, color='purple')
    if geometry is not None:
        ax10.axvline(geometry.rmin, color='red', linestyle='--', alpha=0.7,
                    label=f'r_min = {geometry.rmin} mm')
        ax10.axvline(geometry.rmax, color='red', linestyle='--', alpha=0.7,
                    label=f'r_max = {geometry.rmax:.0f} mm')
        ax10.legend(fontsize=8)
    ax10.set_xlabel("Radius r [mm]")
    ax10.set_ylabel("Hits")
    ax10.set_title("Radial Distribution (All Hits)")
    ax10.grid(True, alpha=0.3)
    
    # --- 11. Time Distribution ---
    ax11 = plt.subplot(3, 4, 11)
    time_bins = np.logspace(np.log10(t_all.min() + 0.01), np.log10(t_all.max()), 100)
    ax11.hist(t_all, bins=time_bins, histtype='step', linewidth=1.5, color='brown', log=True)
    ax11.set_xscale('log')
    ax11.set_xlabel("Hit Time [ns]")
    ax11.set_ylabel("Hits")
    ax11.set_title("Time Distribution (All Hits)")
    
    # --- 12. 2D Histogram: Layer vs x ---
    ax12 = plt.subplot(3, 4, 12)
    if geometry is not None:
        h = ax12.hist2d(x_all, layer_all, bins=[100, geometry.num_layers],
                       cmap='hot', cmin=1)
        ax12.set_ylim(-0.5, geometry.num_layers - 0.5)
    else:
        h = ax12.hist2d(x_all, layer_all, bins=[100, 50], cmap='hot', cmin=1)
    ax12.set_xlabel("x [mm]")
    ax12.set_ylabel("Layer Number")
    ax12.set_title("2D Shower Profile (All)")
    plt.colorbar(h[3], ax=ax12, label="Hits")
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"DATASET OVERVIEW")
    print(f"{'='*60}")
    print(f"Number of showers:       {n_showers}")
    print(f"Total hits:              {len(e_all):,}")
    print(f"Hits per shower:         {len(e_all)/n_showers:.1f} (mean)")
    print(f"")
    print(f"Energy statistics:")
    print(f"  Total energy (all):    {e_all.sum():.3f} GeV")
    print(f"  Mean energy/shower:    {total_energy.mean():.3f} GeV")
    print(f"  Energy range/shower:   {total_energy.min():.3f} - {total_energy.max():.3f} GeV")
    print(f"")
    print(f"Hit energy statistics:")
    print(f"  Min hit energy:        {e_all.min()*1e3:.3f} MeV")
    print(f"  Max hit energy:        {e_all.max()*1e3:.3f} MeV")
    print(f"  Mean hit energy:       {e_all.mean()*1e3:.3f} MeV")
    print(f"")
    print(f"Spatial extent:")
    print(f"  x range:               {x_all.min():.1f} - {x_all.max():.1f} mm")
    print(f"  y range:               {y_all.min():.1f} - {y_all.max():.1f} mm")
    print(f"  z range:               {z_all.min():.1f} - {z_all.max():.1f} mm")
    print(f"  r range:               {r_all.min():.1f} - {r_all.max():.1f} mm")
    print(f"")
    print(f"Layer statistics:")
    print(f"  Layer range:           {layer_all.min()} - {layer_all.max()}")
    print(f"  Shower max (all):      {shower_max}")
    print(f"")
    print(f"Time statistics:")
    print(f"  Time range:            {t_all.min():.2f} - {t_all.max():.2f} ns")
    print(f"  Mean time:             {t_all.mean():.2f} ns")
    print(f"  Median time:           {np.median(t_all):.2f} ns")
    print(f"{'='*60}\n")
    
    plt.tight_layout()
    return fig

def plot_dataset_overview_extended(data, geometry=None, figsize=(20, 14)):
    """
    Extended dataset overview with additional physics plots.
    """
    points = data['events']
    n_points = data['n_points']
    total_energy = data['total_energy']
    incident_energy = data.get('incident_energy', None)
    n_showers = len(n_points)
    
    # Flatten all showers
    all_hits = []
    for i in range(n_showers):
        n = n_points[i]
        all_hits.append(points[i, :n])
    all_hits = np.vstack(all_hits)
    
    x_all = all_hits[:, 0]
    y_all = all_hits[:, 1]
    z_all = all_hits[:, 2]
    t_all = all_hits[:, 3]
    e_all = all_hits[:, 4]
    
    layer_all = z_all.astype(int)
    r_all = np.sqrt(x_all**2 + y_all**2) #+ geometry.rmin if geometry is not None else np.sqrt(x_all**2 + y_all**2)
    
    fig = plt.figure(figsize=figsize)
    fig.suptitle(f"Extended Dataset Analysis - {n_showers} Showers", 
                 fontsize=16, fontweight='bold')
    
    # --- Row 1: Energy distributions ---
    
    # 1. Hit energy spectrum
    ax1 = plt.subplot(3, 5, 1)
    ax1.hist(e_all * 1e3, bins=np.logspace(-2, np.log10(e_all.max()*1e3), 100), 
             histtype='step', log=True, linewidth=2, color='blue')
    ax1.set_xscale('log')
    ax1.set_xlabel("Hit Energy [MeV]")
    ax1.set_ylabel("Counts")
    ax1.set_title("Hit Energy Spectrum")
    ax1.grid(True, alpha=0.3)
    
    # 2. Energy per event
    ax2 = plt.subplot(3, 5, 2)
    ax2.hist(total_energy, bins=50, histtype='step', linewidth=2, color='green')
    ax2.set_xlabel("Total Energy [GeV]")
    ax2.set_ylabel("Showers")
    ax2.set_title("Energy per Shower")
    ax2.grid(True, alpha=0.3)
    
    # 3. Energy vs incident (if available)
    ax3 = plt.subplot(3, 5, 3)
    if incident_energy is not None:
        ax3.scatter(incident_energy.flatten(), total_energy, s=5, alpha=0.5)
        ax3.set_xlabel("Incident Energy [GeV]")
        ax3.set_ylabel("Deposited Energy [GeV]")
        ax3.set_title("Energy Response")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No incident energy', ha='center', va='center', 
                transform=ax3.transAxes)
    
    # 4. Energy vs hits
    ax4 = plt.subplot(3, 5, 4)
    ax4.scatter(n_points, total_energy, s=5, alpha=0.5, color='purple')
    ax4.set_xlabel("Hits per Shower")
    ax4.set_ylabel("Total Energy [GeV]")
    ax4.set_title("Energy vs Multiplicity")
    ax4.grid(True, alpha=0.3)
    
    # 5. Energy per layer
    ax5 = plt.subplot(3, 5, 5)
    unique_layers = np.unique(layer_all)
    energy_per_layer = np.array([e_all[layer_all == lay].sum() for lay in unique_layers])
    ax5.plot(unique_layers, energy_per_layer, 'o-', linewidth=2, markersize=5)
    ax5.set_xlabel("Layer")
    ax5.set_ylabel("Energy [GeV]")
    ax5.set_title("Longitudinal Energy")
    ax5.grid(True, alpha=0.3)
    
    # --- Row 2: Spatial distributions ---
    
    # 6. Transverse view
    ax6 = plt.subplot(3, 5, 6)
    if len(e_all) > 50000:
        idx = np.random.choice(len(e_all), 50000, replace=False)
        ax6.scatter(x_all[idx], y_all[idx], s=0.5, alpha=0.2, color='blue')
    else:
        ax6.scatter(x_all, y_all, s=1, alpha=0.3, color='blue')
    ax6.set_xlabel("x [mm]")
    ax6.set_ylabel("y [mm]")
    ax6.set_title("Transverse Distribution")
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)
    
    # 7. Longitudinal view
    ax7 = plt.subplot(3, 5, 7)
    if len(e_all) > 50000:
        ax7.scatter(x_all[idx], z_all[idx], s=0.5, alpha=0.2, color='green')
    else:
        ax7.scatter(x_all, z_all, s=1, alpha=0.3, color='green')
    ax7.set_xlabel("x [mm]")
    ax7.set_ylabel("z [mm]")
    ax7.set_title("Longitudinal Distribution")
    ax7.grid(True, alpha=0.3)
    
    # 8. Radial distribution
    ax8 = plt.subplot(3, 5, 8)
    ax8.hist(r_all, bins=100, histtype='step', linewidth=2, color='purple', log=True)
    if geometry is not None:
        ax8.axvline(0, color='r', linestyle='--', alpha=0.5)
        ax8.axvline(r_all.max(), color='r', linestyle='--', alpha=0.5)
    ax8.set_xlabel("Radius [mm]")
    ax8.set_ylabel("Hits")
    ax8.set_title("Radial Distribution")
    ax8.grid(True, alpha=0.3)
    
    # 9. Hits per layer
    ax9 = plt.subplot(3, 5, 9)
    hits_per_layer = np.array([np.sum(layer_all == lay) for lay in unique_layers])
    ax9.plot(unique_layers, hits_per_layer, 'o-', linewidth=2, markersize=5, color='orange')
    ax9.set_xlabel("Layer")
    ax9.set_ylabel("Hits")
    ax9.set_title("Hit Multiplicity per Layer")
    
    # 10. 2D shower profile
    ax10 = plt.subplot(3, 5, 10)
    n_layer_bins = geometry.num_layers if geometry is not None else int(layer_all.max()) + 1
    h = ax10.hist2d(x_all, layer_all, bins=[50, n_layer_bins], cmap='hot', cmin=1)
    ax10.set_xlabel("x [mm]")
    ax10.set_ylabel("Layer")
    ax10.set_title("2D Shower Profile")
    plt.colorbar(h[3], ax=ax10, label="Hits")
    
    # --- Row 3: Time and advanced ---
    
    # 11. Time distribution
    ax11 = plt.subplot(3, 5, 11)
    time_bins = np.logspace(np.log10(t_all.min() + 0.01), np.log10(t_all.max()), 50)
    ax11.hist(t_all, bins=time_bins, histtype='step', linewidth=2, log=True, color='brown')
    ax11.set_xscale('log')
    ax11.set_xlabel("Time [ns]")
    ax11.set_ylabel("Hits")
    ax11.set_title("Time Distribution")

    
    # 12. Energy vs time
    ax12 = plt.subplot(3, 5, 12)
    h = ax12.hist2d(t_all, e_all*1e3, bins=[50, 50], cmap='viridis', 
                   norm=mcolors.LogNorm(), cmin=1)
    ax12.set_xscale('log')
    ax12.set_yscale('log')
    ax12.set_xlabel("Time [ns]")
    ax12.set_ylabel("Hit Energy [MeV]")
    ax12.set_title("Energy vs Time")
    plt.colorbar(h[3], ax=ax12, label="Hits")
    
    # 13. Hit multiplicity
    ax13 = plt.subplot(3, 5, 13)
    ax13.hist(n_points, bins=50, histtype='step', linewidth=2, color='red')
    ax13.set_xlabel("Hits per Shower")
    ax13.set_ylabel("Showers")
    ax13.set_title("Hit Multiplicity")

    
    # 14. Shower max distribution
    shower_max_per_event = []
    for i in range(n_showers):
        n = n_points[i]
        shower = points[i, :n]
        layers = shower[:, 2].astype(int)
        energies = shower[:, 4]
        unique = np.unique(layers)
        e_per_lay = np.array([energies[layers == lay].sum() for lay in unique])
        shower_max_per_event.append(unique[np.argmax(e_per_lay)])
    
    ax14 = plt.subplot(3, 5, 14)
    ax14.hist(shower_max_per_event, bins=np.arange(0, 31), histtype='step', 
             linewidth=2, color='darkgreen')
    ax14.set_xlabel("Shower Max Layer")
    ax14.set_ylabel("Showers")
    ax14.set_title("Shower Max Distribution")

    
    # 15. Summary text
    ax15 = plt.subplot(3, 5, 15)
    ax15.axis('off')
    summary = f"""
    Dataset Summary
    
    Showers: {n_showers}
    Total Hits: {len(e_all):,}
    
    Energy:
      Mean: {total_energy.mean():.2f} GeV
      Range: {total_energy.min():.2f}-{total_energy.max():.2f}
    
    Hits/Shower:
      Mean: {n_points.mean():.0f}
      Range: {n_points.min()}-{n_points.max()}
    
    Layers: {layer_all.min()}-{layer_all.max()}
    
    Time: {t_all.min():.1f}-{t_all.max():.1f} ns
    """
    ax15.text(0.1, 0.5, summary, transform=ax15.transAxes,
             fontsize=10, verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    return fig

def plot_single_shower(data, shower_idx=0, figsize=(18, 12)):

    """
    Comprehensive visualization of a single shower.
    
    Parameters:
    -----------
    data : dict
        Point cloud data from load_pointcloud_h5
    shower_idx : int
        Index of shower to plot
    """
    points = data['events']
    n_points = data['n_points']
    
    # Extract shower
    n = n_points[shower_idx]
    shower = points[shower_idx, :n]
    x, y, z, t, e = shower[:, 0], shower[:, 1], shower[:, 2], shower[:, 3], shower[:, 4]
    
    # Compute radius
    r = np.sqrt(y**2 + z**2)
    
    fig = plt.figure(figsize=figsize)
    fig.suptitle(f"Shower {shower_idx} - {n} hits, E={e.sum():.3f} GeV", 
                 fontsize=14, fontweight='bold')
    
    # Energy normalization for coloring
    norm = mcolors.LogNorm(vmin=max(e.min()*1e3, 0.01), vmax=e.max()*1e3)
    
    # --- 1. Hit Energy Spectrum ---
    ax1 = plt.subplot(3, 4, 1)
    ax1.hist(e * 1e3, bins=np.logspace(-2, 2, 50), histtype='step', linewidth=2)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel("Hit Energy [MeV]")
    ax1.set_ylabel("Counts")
    ax1.set_title("Hit Energy Spectrum")
    
    # --- 2. Transverse View (x-y) ---
    ax2 = plt.subplot(3, 4, 2)
    sc = ax2.scatter(x, y, c=e*1e3, s=3, cmap='plasma', alpha=0.6, norm=norm)
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("y [mm]")
    ax2.set_title("Transverse View")
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax2, label="E [MeV]")
    
    # --- 3. Longitudinal View (x-z) ---
    ax3 = plt.subplot(3, 4, 3)
    sc = ax3.scatter(x, z, c=e*1e3, s=3, cmap='plasma', alpha=0.6, norm=norm)
    ax3.set_xlabel("x [mm]")
    ax3.set_ylabel("z [mm]")
    ax3.set_title("Longitudinal View (x-z)")
    ax3.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax3, label="E [MeV]")
    
    # --- 4. Layer vs x Position ---
    ax4 = plt.subplot(3, 4, 4)
    layer = z.astype(int)
    sc = ax4.scatter(x, layer, c=e*1e3, s=3, cmap='plasma', alpha=0.6, norm=norm)
    ax4.set_xlabel("x [mm]")
    ax4.set_ylabel("Layer Number")
    ax4.set_title("Longitudinal Development")
    ax4.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax4, label="E [MeV]")
    
    # --- 5. Energy per Layer ---
    ax5 = plt.subplot(3, 4, 5)
    unique_layers = np.unique(layer)
    energy_per_layer = np.array([e[layer == lay].sum() for lay in unique_layers])
    ax5.bar(unique_layers, energy_per_layer, color='orangered', alpha=0.7)
    shower_max = unique_layers[np.argmax(energy_per_layer)]
    ax5.axvline(shower_max, color='blue', linestyle='--', linewidth=2, 
                label=f'Shower max: layer {shower_max}')
    ax5.set_xlabel("Layer Number")
    ax5.set_ylabel("Energy [GeV]")
    ax5.set_title("Energy Profile")
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # --- 6. Hits per Layer ---
    ax6 = plt.subplot(3, 4, 6)
    hits_per_layer = np.array([np.sum(layer == lay) for lay in unique_layers])
    ax6.bar(unique_layers, hits_per_layer, color='forestgreen', alpha=0.7)
    ax6.set_xlabel("Layer Number")
    ax6.set_ylabel("Number of Hits")
    ax6.set_title("Hit Multiplicity per Layer")
    ax6.grid(True, alpha=0.3, axis='y')
    
    # --- 7. Radial Distribution ---
    ax7 = plt.subplot(3, 4, 7)
    ax7.hist(r, bins=50, histtype='step', linewidth=2, color='purple')
    ax7.set_xlabel("Radius r [mm]")
    ax7.set_ylabel("Hits")
    ax7.set_title("Radial Distribution")
    ax7.grid(True, alpha=0.3)
    
    # --- 8. Time Distribution ---
    ax8 = plt.subplot(3, 4, 8)
    time_bins = np.logspace(np.log10(t.min() + 0.01), np.log10(t.max()), 50)
    ax8.hist(t, bins=time_bins, histtype='step', linewidth=2, color='brown', log=True)
    ax8.set_xscale('log')
    ax8.set_xlabel("Hit Time [ns]")
    ax8.set_ylabel("Hits")
    ax8.set_title("Time Distribution")
    ax8.grid(True, alpha=0.3)
    
    # --- 9. Energy vs Time ---
    ax9 = plt.subplot(3, 4, 9)
    h = ax9.hist2d(t, e*1e3, bins=[50, 50], cmap='hot', norm=mcolors.LogNorm(), cmin=1)
    ax9.set_xlabel("Time [ns]")
    ax9.set_ylabel("Hit Energy [MeV]")
    ax9.set_yscale('log')
    ax9.set_xscale('log')
    ax9.set_title("Energy vs Time")
    plt.colorbar(h[3], ax=ax9, label="Hits")
    
    # --- 10. Time vs Depth ---
    ax10 = plt.subplot(3, 4, 10)
    sc = ax10.scatter(x, t, c=e*1e3, s=2, cmap='plasma', alpha=0.5, norm=norm)
    ax10.set_xlabel("Depth x [mm]")
    ax10.set_ylabel("Time [ns]")
    ax10.set_yscale('log')
    ax10.set_title("Time vs Depth")
    ax10.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax10, label="E [MeV]")
    
    # --- 11. 2D Histogram: Layer vs x ---
    ax11 = plt.subplot(3, 4, 11)
    h = ax11.hist2d(x, layer, bins=[50, len(unique_layers)], cmap='hot', cmin=1)
    ax11.set_xlabel("x [mm]")
    ax11.set_ylabel("Layer Number")
    ax11.set_title("2D Shower Profile")
    plt.colorbar(h[3], ax=ax11, label="Hits")
    
    # --- 12. Summary Statistics ---
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    stats_text = f"""
    Total Hits: {n:,}
    Total Energy: {e.sum():.3f} GeV
    
    Shower Max: Layer {shower_max}
    Depth Range: {x.min():.1f} - {x.max():.1f} mm
    Radius Range: {r.min():.1f} - {r.max():.1f} mm
    
    Time Range: {t.min():.2f} - {t.max():.2f} ns
    Mean Time: {t.mean():.2f} ns
    
    Mean x: {x.mean():.1f} mm
    Mean y: {y.mean():.1f} mm
    Std x: {x.std():.1f} mm
    Std y: {y.std():.1f} mm
    """
    ax12.text(0.1, 0.5, stats_text, transform=ax12.transAxes,
             fontsize=11, verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    return fig

def plot_multi_dataset_comparison(dataset_dict, figsize=(20, 14)):
    """
    Compare multiple datasets from different configurations on the same plots.
    """

    n_configs = len(dataset_dict)
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_configs))

    fig = plt.figure(figsize=figsize)


    # Process each dataset
    all_data = {}
    for idx, (config_id, data) in enumerate(dataset_dict.items()):
        points = data['events']
        n_points = data['n_points']
        total_energy = data['total_energy']

        # Extract scalar value from sampling_fraction array
        sampling_fraction_array = data['sampling_fraction']
        if isinstance(sampling_fraction_array, np.ndarray):
            sampling_fraction = float(sampling_fraction_array[0, 0])  # Extract first value
        else:
            sampling_fraction = float(sampling_fraction_array)

        # Extract measured sampling fraction
        measured_sf = data.get('measured_sampling_fraction', sampling_fraction)
        if isinstance(measured_sf, np.ndarray):
            measured_sf = float(measured_sf[0, 0])
        else:
            measured_sf = float(measured_sf)

        incident_energy = data.get('incident_energy', None)
        n_showers = len(n_points)

        # Flatten all showers for this config
        all_hits = []
        for i in range(n_showers):
            n = n_points[i]
            all_hits.append(points[i, :n])
        all_hits = np.vstack(all_hits)

        x_all = all_hits[:, 0]
        y_all = all_hits[:, 1]
        z_all = all_hits[:, 2]
        t_all = all_hits[:, 3]
        e_all = all_hits[:, 4]

        # Use index 2 for the layer number (assuming z_all is the layer depth/index)
        # If z_all is a float depth, you might need to convert it to an integer layer index
        layer_all = z_all.astype(int) 
        r_all = np.sqrt(x_all**2 + y_all**2)

        # Store hits separated by event/shower for correct longitudinal calculation
        # Store individual shower data (hits, energy, layer, incident energy)
        shower_data = []
        if incident_energy is not None:
            incident_energy = incident_energy.flatten()
        
        start_idx = 0
        for i in range(n_showers):
            end_idx = start_idx + n_points[i]
            E_inc = incident_energy[i] if incident_energy is not None else 1.0 # Use 1.0 for mean normalization if E_inc is missing
            
            shower_data.append({
                'layer': points[i, :n_points[i], 2].astype(int),
                'e': points[i, :n_points[i], 4],
                'E_inc': E_inc
            })
            start_idx = end_idx

        all_data[config_id] = {
            'x': x_all, 'y': y_all, 'z': z_all, 't': t_all, 'e': e_all,
            'layer': layer_all, 'r': r_all,
            'n_points': n_points,
            'total_energy': total_energy,
            'incident_energy': incident_energy,
            'sampling_fraction': sampling_fraction,
            'measured_sampling_fraction': measured_sf,
            'color': colors[idx],
            'label': f'f={measured_sf:.4f}',  # Use measured sampling fraction
            'shower_data': shower_data, # NEW: Individual shower data
            'n_showers': n_showers # NEW: Number of showers
        }
    # --- Create 3x3 grid for 7 plots + legend ---

    # 1. Hit energy spectrum
    ax1 = plt.subplot(3, 3, 1)
    for config_id, d in all_data.items():
        ax1.hist(d['e'] * 1e3,
                bins=np.logspace(np.log10(d['e'].min()*1e3), np.log10(d['e'].max()*1e3), 100),
                 histtype='step', log=True, linewidth=2.5,
                 color=d['color'], label=d['label'], alpha=0.8)
    ax1.set_xscale('log')
    ax1.set_xlabel("Hit Energy [MeV]", fontsize=18, fontweight='bold')
    ax1.set_ylabel("Counts", fontsize=18, fontweight='bold')
    ax1.set_title("Hit Energy Spectrum", fontsize=22, fontweight='bold')
    ax1.set_xlim(5e-3, None)
    ax1.tick_params(axis='both', which='major', labelsize=20)

    # 2. Energy Response
    ax2 = plt.subplot(3, 3, 2)
    has_incident = any(d['incident_energy'] is not None for d in all_data.values())
    if has_incident:
        for config_id, d in all_data.items():
            if d['incident_energy'] is not None:
                ax2.scatter(d['incident_energy'], d['total_energy'],
                           s=10, alpha=0.5, color=d['color'], label=d['label'])
        ax2.set_xlabel("Incident Energy [GeV]", fontsize=18, fontweight='bold')
        ax2.set_ylabel("Deposited Energy [GeV]", fontsize=18, fontweight='bold')
        ax2.set_title("Energy Response", fontsize=22, fontweight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=20)
    else:
        ax2.text(0.5, 0.5, 'No incident energy', ha='center', va='center',
                transform=ax2.transAxes, fontsize=18)
    
    # ----------------------------------------------------------------------
    # 3. Longitudinal Energy Distribution (CORRECTED)
    # ----------------------------------------------------------------------
    # 3. Longitudinal Energy Distribution (REPLICATING ORIGINAL LOGIC FOR AX3)
    ax3 = plt.subplot(3, 3, 3)
    
    # --- Data preparation to replicate original Z-binning ---
    # The original logic used all Z hits across all configurations to define the global bins.
    z_all_combined = np.concatenate([d['z'] for d in all_data.values()])
    z_valid = z_all_combined[~np.isnan(z_all_combined)] 
    
    if len(z_valid) > 0:
        # Define 31 bins based on the full Z range
        Z_BINS = 31
        z_bins = np.linspace(z_valid.min(), z_valid.max(), Z_BINS)
        z_centers = (z_bins[:-1] + z_bins[1:]) / 2
        
        # Iterate over each configuration to compute its profile
        for config_id, d in all_data.items():
            
            # --- Re-segment Flattened Hits into Showers ---
            # We must reconstruct the per-shower structure (Z and E hits per event).
            shower_hits = []
            current_start = 0
            for n_hit in dataset_dict[config_id]['n_points']:
                current_end = current_start + n_hit
                if n_hit > 0:
                    shower_hits.append({
                        'z': d['z'][current_start:current_end],
                        'E': d['e'][current_start:current_end],
                    })
                current_start = current_end
                
            n_showers = len(shower_hits) # Number of showers with hits
            energy_profile = []
            
            # --- Apply Original Longitudinal Profile Logic ---
            for j in range(len(z_bins) - 1):
                energies_in_bin = []
                
                # Iterate over every single event (shower) in this configuration
                for shower in shower_hits:
                    
                    # Mask hits for the current Z bin: [z_bins[j], z_bins[j+1])
                    mask_event_in_bin = (shower['z'] >= z_bins[j]) & (shower['z'] < z_bins[j+1])
                    
                    if mask_event_in_bin.sum() > 0:
                        # Sum the energy of all hits that fall into this bin for this event
                        E_in_bin_for_event = shower['E'][mask_event_in_bin].sum()
                        energies_in_bin.append(E_in_bin_for_event)
                
                # Apply the original averaging logic: mean over events that HAD hits in this bin
                if len(energies_in_bin) > 0:
                    energy_profile.append(np.mean(energies_in_bin))
                else:
                    # If no events had hits in this specific bin, plot as NaN
                    energy_profile.append(np.nan)
            
            energy_profile = np.array(energy_profile)
            
            # Plotting using Z centers
            ax3.plot(z_centers, energy_profile, 'o-',
                    linewidth=2.5, markersize=5, color=d['color'],
                    label=d['label'], alpha=0.8)

    ax3.set_xlabel("z [layer]", fontsize=18, fontweight='bold')
    ax3.set_ylabel("Energy [GeV]", fontsize=18, fontweight='bold')
    ax3.set_title("Longitudinal Energy Distribution", fontsize=22, fontweight='bold')
    ax3.tick_params(axis='both', which='major', labelsize=20)
    # ----------------------------------------------------------------------


    # 4. Transverse Distribution
    ax4 = plt.subplot(3, 3, 4)
    for config_id, d in all_data.items():
        if len(d['e']) > 10000:
            idx = np.random.choice(len(d['e']), 10000, replace=False)
            ax4.scatter(d['x'][idx], d['y'][idx], s=0.75,
                       alpha=1, color=d['color'], label=d['label'])
        else:
            ax4.scatter(d['x'], d['y'], s=1, alpha=0.5,
                       color=d['color'], label=d['label'])
    ax4.set_xlabel("x [mm]", fontsize=18, fontweight='bold')
    ax4.set_ylabel("y [mm]", fontsize=18, fontweight='bold')
    ax4.set_title("Transverse Distribution", fontsize=22, fontweight='bold')
    ax4.set_aspect('equal')
    ax4.tick_params(axis='both', which='major', labelsize=20)

    # 5. Longitudinal Distribution (x-z view)
    ax5 = plt.subplot(3, 3, 5)
    for config_id, d in all_data.items():
        if len(d['e']) > 10000:
            idx = np.random.choice(len(d['e']), 10000, replace=False)
            ax5.scatter(d['x'][idx], d['z'][idx], s=0.75,
                       alpha=1, color=d['color'], label=d['label'])
        else:
            ax5.scatter(d['x'], d['z'], s=1, alpha=1,
                       color=d['color'], label=d['label'])
    ax5.set_xlabel("x [mm]", fontsize=18, fontweight='bold')
    ax5.set_ylabel("z [layer]", fontsize=18, fontweight='bold')
    ax5.set_title("Longitudinal Distribution", fontsize=22, fontweight='bold')
    ax5.tick_params(axis='both', which='major', labelsize=20)

    # 6. Radial Distribution
    ax6 = plt.subplot(3, 3, 6)
    for config_id, d in all_data.items():
        ax6.hist(d['r'], bins=50, histtype='step', linewidth=2.5,
                color=d['color'], label=d['label'], log=True, alpha=0.8)
    ax6.set_xlabel("Radius [mm]", fontsize=18, fontweight='bold')
    ax6.set_ylabel("Hits", fontsize=18, fontweight='bold')
    ax6.set_title("Radial Distribution", fontsize=22, fontweight='bold')
    ax6.tick_params(axis='both', which='major', labelsize=20)

    # 7. Hit Multiplicity per Layer
    ax7 = plt.subplot(3, 3, 7)
    for config_id, d in all_data.items():
        # Get number of showers for this config
        n_showers = len(dataset_dict[config_id]['n_points'])

        unique_layers = np.unique(d['layer'])
        hits_per_layer = np.array([np.sum(d['layer'] == lay)
                                for lay in unique_layers])
        # Normalize by number of showers
        hits_per_layer_normalized = hits_per_layer / n_showers

        ax7.plot(unique_layers, hits_per_layer_normalized, 'o-',
                linewidth=2.5, markersize=5, color=d['color'],
                label=d['label'], alpha=0.8)
    ax7.set_xlabel("Layer", fontsize=18, fontweight='bold')
    ax7.set_ylabel("Hits per Shower", fontsize=18, fontweight='bold')
    ax7.set_title("Hit Multiplicity per Layer", fontsize=22, fontweight='bold')
    ax7.tick_params(axis='both', which='major', labelsize=20)

    # 8. Time Distribution
    ax8 = plt.subplot(3, 3, 8)
    for config_id, d in all_data.items():
        t_min = max(d['t'].min(), 0.01)
        t_max = d['t'].max()
        if t_max > t_min:
            time_bins = np.logspace(np.log10(t_min), np.log10(t_max), 30)
            ax8.hist(d['t'], bins=time_bins, histtype='step',
                     linewidth=2.5, log=True, color=d['color'],
                     label=d['label'], alpha=0.8)
    ax8.set_xscale('log')
    ax8.set_xlabel("Time [ns]", fontsize=18, fontweight='bold')
    ax8.set_ylabel("Hits", fontsize=18, fontweight='bold')
    ax8.set_title("Time Distribution", fontsize=22, fontweight='bold')
    ax8.tick_params(axis='both', which='major', labelsize=20)

    # 9. Legend panel
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    # Create custom legend handles
    from matplotlib.patches import Patch
    legend_handles = []
    for config_id, d in all_data.items():
        label = (
                f"f_meas = {d['measured_sampling_fraction']:.4f}\n"
                f"Max Hits: {d['n_points'].max():.0f}"
                )
        legend_handles.append(Patch(facecolor=d['color'], label=label))

    ax9.legend(handles=legend_handles, loc='center', fontsize=24,
            frameon=False)

    plt.tight_layout()
    return fig

def plot_energy_spectrum(dataset_dict, title='Energy Spectrum of Geant4 Hits', output_path=None):
    """
    Plot energy spectrum of G4 steps from point cloud data.
    
    Parameters:
    -----------
    dataset_dict : dict
        Dictionary with config_id as keys, each containing:
        - 'events': point cloud array (n_showers, max_points, 5)
        - 'sampling_fraction': sampling fraction value
    output_path : str or None
        Path to save the figure (optional)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Generate distinct colors for each configuration
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(dataset_dict)))
    
    print("="*70)
    print("ENERGY SPECTRUM ANALYSIS")
    print("="*70)
    
    # Loop through each configuration
    for idx, (config_id, data) in enumerate(dataset_dict.items()):
        # Extract energy from point cloud
        # Points shape: (n_showers, max_points, 5) where last dimension = [x, y, z, time, energy]
        points = data['events']
        energy_all = points[:, :, 4].flatten()  # Extract energy feature (index 4)
        
        # Remove padding zeros (points that don't exist)
        energy_all = energy_all[energy_all > 0]
        
        # Convert from GeV to MeV for better readability
        energy_all_mev = energy_all * 1e3
        
        # Get sampling fraction for this configuration (fallback to sampling_fraction if measured not available)
        measured_sf = data.get('measured_sampling_fraction', data.get('sampling_fraction'))
        f_samp = measured_sf[0, 0] if isinstance(measured_sf, np.ndarray) else measured_sf
        
        # Filter energy range: exclude extreme outliers
        # Keep energies between 0.01 MeV (10 keV) and 1000 MeV (1 GeV)
        # energy_range = energy_all_mev[(energy_all_mev > 1e-3) & (energy_all_mev < 1e3)]
        
        # Create logarithmically-spaced bins for histogram
        # 200 bins spanning from minimum to maximum energy in the filtered range
        bins = np.logspace(np.log10(energy_all_mev.min()), np.log10(energy_all_mev.max()), 200)
        
        # Plot histogram using matplotlib's hist function
        # histtype='step' creates line-style histogram (no fill)
        # This automatically handles the binning and counting
        ax.hist(energy_all_mev, bins=bins, histtype='step', linewidth=3,
                color=colors[idx], alpha=0.85, label=f'f = {f_samp:.4f}')
        
        # Print statistical information for this configuration
        print(f"\nConfig {config_id} (f={f_samp:.4f}):")
        print(f"  Total hits:      {len(energy_all_mev):,}")
        print(f"  Energy range:    {energy_all_mev.min():.3f} - {energy_all_mev.max():.1f} MeV")
        print(f"  Median:          {np.median(energy_all_mev):.2f} MeV")
    
    # Configure plot formatting
    ax.set_xscale('log')  # Logarithmic x-axis (energy)
    ax.set_yscale('log')  # Logarithmic y-axis (counts)
    ax.set_xlabel('Energy per G4 Step [MeV]', fontsize=20, fontweight='bold')
    ax.set_ylabel('Counts', fontsize=20, fontweight='bold')
    ax.set_title(title, fontsize=24, fontweight='bold')
    ax.legend(fontsize=20, loc='best', framealpha=0.9)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlim(5e-3, 5e2)  # Fixed x-axis range: 0.01 to 1000 MeV
    ax.set_ylim(1, 1e5)    # y-axis minimum = 1 (log scale requirement)
    
    # Finalize layout to prevent label cutoff
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Figure saved to: {output_path}")
    print("="*70)
    
    return fig, ax