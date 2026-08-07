import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils.calo_geometry import SimpleBoxGeometry

plt.rcParams.update({
        # Use a serif font that's likely available
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Liberation Serif', 'Computer Modern Roman', 'Bitstream Vera Serif'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 600,  # Higher DPI for publication quality
        'savefig.format': 'pdf',  # PDF format is often preferred for publications
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.linewidth': 0.8,  # Slightly thinner axes lines
        'lines.linewidth': 1.5,  # Slightly thicker plot lines
        'lines.markersize': 4,  # Slightly smaller markers
        # 'axes.grid': True,
        'grid.alpha': 0.3
    })

def analyze_layers(data, geometry):
    """Analyze energy deposition by layer."""
    energy_per_layer = np.zeros(geometry.num_layers)
    hits_per_layer = np.bincount(data['layer'], minlength=geometry.num_layers)
    
    for layer in range(geometry.num_layers):
        mask = data['layer'] == layer
        energy_per_layer[layer] = data['energy'][mask].sum()
    
    shower_max_layer = np.argmax(energy_per_layer)
    
    return {
        'energy_per_layer': energy_per_layer,
        'hits_per_layer': hits_per_layer,
        'shower_max_layer': shower_max_layer,
        'total_energy': data['energy'].sum(),
        'num_layers': geometry.num_layers
    }

def plot_calo_analysis(data, figsize=(16, 10)):
    """Create comprehensive calorimeter analysis plots."""
    energy_flat = data['energy']
    x_flat = data['x']
    y_flat = data['y']
    z_flat = data['z']
    t_flat = data['time']
    total_energy = data['total_energy']
    nhits = data['nhits']
    r_flat = np.sqrt(x_flat**2 + y_flat**2)
    
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    
    # 1. Hit Energy Spectrum
    axes[0,0].hist(energy_flat * 1e3, bins=np.logspace(np.log10(energy_flat.min()*1e3), np.log10(energy_flat.max()*1e3), 100), 
                   histtype='step', log=True, linewidth=1.5)
    axes[0,0].set_xscale('log')
    axes[0,0].set_xlabel("Hit Energy [MeV]")
    axes[0,0].set_ylabel("Counts")
    axes[0,0].set_title("Hit Energy Spectrum")
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Total Energy per Event
    axes[0,1].hist(total_energy, bins=50, histtype='step', linewidth=1.5)
    axes[0,1].set_xlabel("Energy sum per Event [GeV]")
    axes[0,1].set_ylabel("Events")
    axes[0,1].set_title("Energy Sum Distribution")
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Hit Multiplicity
    axes[0,2].hist(nhits, bins=100, histtype='step', linewidth=1.5, color='green')
    axes[0,2].set_xscale('log')
    axes[0,2].set_yscale('log')
    axes[0,2].set_xlabel("Number of Hits per Event")
    axes[0,2].set_ylabel("Events")
    axes[0,2].set_title("Hit Multiplicity")
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Transverse View (x-y)
    norm = mcolors.LogNorm(vmin=energy_flat.min()*1e3 + 1e-3, vmax=energy_flat.max()*1e3)

    sc = axes[1,0].scatter(x_flat, y_flat, 
                           c=energy_flat * 1e3,
                           s=1, cmap="plasma", 
                           alpha=0.5, 
                           norm=norm)
    axes[1,0].set_xlabel("x [mm]")
    axes[1,0].set_ylabel("y [mm]")
    axes[1,0].set_title("Transverse View")
    axes[1,0].set_aspect('equal')
    axes[1,0].grid(True, alpha=0.3)
    plt.colorbar(sc, ax=axes[1,0], label="E [MeV]")
    
    # 5. Longitudinal View (z-y)
    sc2 = axes[1,1].scatter(z_flat, y_flat,
                            c=energy_flat * 1e3,
                            s=1, cmap="plasma", alpha=0.5, norm=norm)
    axes[1,1].set_xlabel("z [mm] (beam direction)")
    axes[1,1].set_ylabel("y [mm]")
    axes[1,1].set_title("Longitudinal View")
    axes[1,1].grid(True, alpha=0.3)
    plt.colorbar(sc2, ax=axes[1,1], label="E [MeV]")
    
    # 6. z vs x view
    sc3 = axes[1,2].scatter(z_flat, x_flat,
                            c=energy_flat * 1e3,
                            s=1, cmap="plasma", alpha=0.5, norm=norm)
    axes[1,2].set_xlabel("z [mm] (beam direction)")
    axes[1,2].set_ylabel("x [mm]")
    axes[1,2].set_title("z vs x View")
    axes[1,2].grid(True, alpha=0.3)
    plt.colorbar(sc3, ax=axes[1,2], label="E [MeV]")
    
    # 7. Longitudinal Energy Profile
    z_bins = np.linspace(z_flat.min(), z_flat.max(), 50)
    e_prof_z, z_edges = np.histogram(z_flat, bins=z_bins, weights=energy_flat)
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    axes[2,0].plot(z_centers, e_prof_z, 'o-', markersize=4)
    axes[2,0].set_xlabel("z [mm] (depth)")
    axes[2,0].set_ylabel("Energy Deposited [GeV]")
    axes[2,0].set_title("Longitudinal Profile")
    axes[2,0].grid(True, alpha=0.3)
    
    # 8. Transverse Energy Profile
    r_bins = np.linspace(0, r_flat.max(), 50)
    e_prof_r, r_edges = np.histogram(r_flat, bins=r_bins, weights=energy_flat)
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2
    axes[2,1].plot(r_centers, e_prof_r, 'o-', markersize=4, color='orange')
    axes[2,1].set_xlabel("Transverse Radius [mm]")
    axes[2,1].set_ylabel("Energy Deposited [GeV]")
    axes[2,1].set_title("Transverse Profile")
    axes[2,1].grid(True, alpha=0.3)
    
    # 9. Hit Time Distribution
    axes[2,2].hist(t_flat, bins=100, histtype='step', 
                   linewidth=1.5, color='purple', log=True)
    axes[2,2].set_xlabel("Hit Time [ns]")
    axes[2,2].set_ylabel("Number of Hits")
    axes[2,2].set_title("Hit Time Distribution")
    axes[2,2].grid(True, alpha=0.3)

    # Print statistics
    stats_text = f"""
    Total events: {len(total_energy)}
    Total hits: {len(energy_flat)}
    
    Mean E/event: {np.mean(total_energy):.3f} GeV
    Std E/event: {np.std(total_energy):.3f} GeV
    
    Mean hits/event: {np.mean(nhits):.1f}
    Std hits/event: {np.std(nhits):.1f}
    
    Time range: {t_flat.min():.2f} - {t_flat.max():.2f} ns
    Mean time: {t_flat.mean():.2f} ns
    """
    print(stats_text)
    
    plt.tight_layout()
    return fig


def _create_scatter(ax, x, y, energy, title, xlabel, ylabel, **kwargs):
    """Create energy-colored scatter plot."""
    sc = ax.scatter(x, y, 
                    c=np.log10(energy * 1e3 + 0.1), 
                   cmap="hot", norm=mcolors.Normalize(), 
                   rasterized=True,
                 **kwargs
                   )
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
    plt.colorbar(sc, ax=ax, label="log₁₀(E [MeV])")  # Commented out since no color mapping


def _downsample(data, max_points=100000):
    """Randomly downsample data for faster plotting."""
    n = len(data['x'])
    if n <= max_points:
        return data
    
    indices = np.random.choice(n, max_points, replace=False)
    
    # Only downsample arrays with same length as 'x' (hit-level data)
    downsampled = {}
    for key, val in data.items():
        if isinstance(val, np.ndarray) and len(val) == n:
            downsampled[key] = val[indices]
        else:
            # Keep scalars and other arrays unchanged
            downsampled[key] = val
    
    return downsampled


def _print_statistics(data_cut, cone_angle):
    """Print cut statistics."""
    print(f"\n{'='*60}")
    print(f"CUT ANALYSIS")
    print(f"{'='*60}")
    print(f"Total hits (original):   {data_cut['n_hits_original']:,}")
    print(f"Total hits (after cuts): {data_cut['n_hits_final']:,}")
    print(f"Fraction kept:           {100*data_cut['n_hits_final']/data_cut['n_hits_original']:.2f}%\n")
    print(f"3D cone cut:             {cone_angle}° opening (±{cone_angle/2:.1f}° from z-axis)")
    print(f"Total energy (cut data): {data_cut['energy'].sum():.3f} GeV")
    print(f"Mean hit energy:         {data_cut['energy'].mean()*1e3:.2f} MeV")
    print(f"{'='*60}\n")


def plot_cone_cut_comparison(data_original, data_cut, figsize=(12, 16), downsample=None):
    """
    Visualize 3D cone cut: original vs cut data in x-y, y-z, x-z projections.
    
    Parameters:
    -----------
    data_original : dict
        Original data (with rotation applied if needed)
    data_cut : dict
        Data after cone cut (from process_calo_hits)
    figsize : tuple
        Figure size
    downsample : int or None
        Maximum number of points to plot (for speed). None means plot all.
    """
    cone_angle = data_cut['cone_angle']
    
    # Downsample ONLY for visualization
    if downsample:
        data_original_plot = _downsample(data_original, max_points=downsample)
        data_cut_plot = _downsample(data_cut, max_points=downsample)
    else:
        data_original_plot = data_original
        data_cut_plot = data_cut

    fig, axes = plt.subplots(3, 2, figsize=figsize)

    views = [
        (0, 'x', 'y', "x [mm]", "y [mm]", "x-y view"),
        (1, 'y', 'z', "y [mm]", "z [mm]", "y-z view"),
        (2, 'z', 'x', "z [mm]", "x [mm]", "x-z view")
    ]

    for row, coord1, coord2, xlabel, ylabel, view_name in views:
        _create_scatter(axes[row, 0], data_original_plot[coord1], data_original_plot[coord2],
                       data_original_plot['energy'], f"Original Data - {view_name}",
                       xlabel, ylabel, s=0.5, alpha=0.4)
        
        _create_scatter(axes[row, 1], data_cut_plot[coord1], data_cut_plot[coord2],
                       data_cut_plot['energy'], f"After Cone Cut ({cone_angle}°) - {view_name}",
                       xlabel, ylabel, s=0.5, alpha=0.4)
    
    _print_statistics(data_cut, cone_angle)
    
    plt.tight_layout()
    return fig


def plot_layer_analysis(layer_data, figsize=(12, 5)):
    """
    Plot layer-by-layer energy deposition and hit distribution.
    
    Parameters:
    -----------
    layer_data : dict
        Output from analyze_layers()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    centers = layer_data['layer_centers']
    
    # Energy per layer
    ax1.plot(centers, layer_data['energy_per_layer'], 'o-', linewidth=2, markersize=5)
    ax1.axvline(layer_data['shower_max_depth'], color='red', linestyle='--',
                label=f"Shower max @ {layer_data['shower_max_depth']:.1f} mm")
    ax1.set_xlabel("Depth z [mm]", fontsize=12)
    ax1.set_ylabel("Energy [GeV]", fontsize=12)
    ax1.set_title("Longitudinal Energy Profile", fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Hits per layer
    ax2.bar(centers, layer_data['hits_per_layer'], width=np.diff(layer_data['layer_edges'])[0],
            alpha=0.7, edgecolor='black')
    ax2.axvline(layer_data['shower_max_depth'], color='red', linestyle='--',
                label=f"Shower max (layer {layer_data['shower_max_layer']})")
    ax2.set_xlabel("Depth z [mm]", fontsize=12)
    ax2.set_ylabel("Number of Hits", fontsize=12)
    ax2.set_title("Hit Distribution by Layer", fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    print(f"\n{'='*50}")
    print(f"LAYER ANALYSIS")
    print(f"{'='*50}")
    print(f"Number of layers:        {layer_data['num_layers']}")
    print(f"Shower max layer:        {layer_data['shower_max_layer']}")
    print(f"Shower max depth:        {layer_data['shower_max_depth']:.1f} mm")
    print(f"Total energy:            {layer_data['total_energy']:.3f} GeV")
    print(f"{'='*50}\n")
    
    plt.tight_layout()
    return fig

def plot_comprehensive_analysis(data, geometry, total_energy_all=None, nhits_all=None,
                                figsize=(18, 12)):
    """
    Create comprehensive 12-panel calorimeter analysis.
    Supports both cylindrical (CaloGeometry) and planar (SimpleBoxGeometry).

    Parameters:
    -----------
    data : dict
        Data with 'x', 'y', 'z', 'energy', 'layer' keys
        Also 'r' for cylindrical geometry or 'z' for planar geometry
    geometry : CaloGeometry or SimpleBoxGeometry
        Geometry parameters
    total_energy_all : array or None
        Per-event total energies (for event-level plots)
    nhits_all : array or None
        Per-event hit counts (for event-level plots)
    """
    # Detect geometry type
    is_simplebox = isinstance(geometry, SimpleBoxGeometry)

    fig = plt.figure(figsize=figsize)

    energy_cut = data['energy']
    x_cut = data['x']
    y_cut = data['y']
    z_cut = data['z']
    layer_cut = data['layer']
    time = data['time']

    # For cylindrical geometry, use 'r'; for planar, use 'z'
    if is_simplebox:
        depth_coord = z_cut  # z is the depth coordinate for SimpleBox
        coord_label = "z"
    else:
        depth_coord = data['r']  # r is the depth coordinate for cylindrical
        coord_label = "r"

    norm = mcolors.LogNorm(vmin=max(energy_cut.min()*1e3 + 1e-3, 1e-3),
                            vmax=energy_cut.max()*1e3)

    # --- 1. Hit Energy Spectrum ---
    ax1 = plt.subplot(3, 4, 1)
    ax1.hist(energy_cut * 1e3, bins=np.logspace(-2, np.log10(energy_cut.max()*1e3), 100), 
             histtype='step', log=True, linewidth=1.5)
    ax1.set_xscale('log')
    # ax1.set_xlim(-2, energy_cut.max()*1e3)
    ax1.set_xlabel("Hit Energy [MeV]")
    ax1.set_ylabel("Counts")
    ax1.set_title("Hit Energy Spectrum")
    
    # --- 2. Total Energy per Event ---
    ax2 = plt.subplot(3, 4, 2)
    if total_energy_all is not None:
        ax2.hist(total_energy_all, bins=50, histtype='step', linewidth=1.5, color='blue')
        ax2.set_xlabel("Energy sum per Event [GeV]")
        ax2.set_ylabel("Events")
        ax2.set_title("Energy Sum Distribution")
    else:
        ax2.text(0.5, 0.5, 'No per-event data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Energy Sum Distribution")
    
    # --- 3. Hit Multiplicity ---
    ax3 = plt.subplot(3, 4, 3)
    if nhits_all is not None:
        ax3.hist(nhits_all, bins=100, histtype='step', linewidth=1.5, color='green')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel("Number of Hits per Event")
        ax3.set_ylabel("Events")
        ax3.set_title("Hit Multiplicity")
    else:
        ax3.text(0.5, 0.5, 'No per-event data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title("Hit Multiplicity")
    
    # --- 4. Transverse View (x-y) ---
    ax4 = plt.subplot(3, 4, 4)
    sc = ax4.scatter(x_cut, y_cut, c=energy_cut * 1e3, s=1, 
                     cmap="plasma", alpha=0.6, norm=norm)
    ax4.set_xlabel("x [mm] (beam direction)")
    ax4.set_ylabel("y [mm]")
    ax4.set_title("Transverse View")
    ax4.set_aspect('equal')
    plt.colorbar(sc, ax=ax4, label="E [MeV]")
    
    # --- 5. Hits colored by Layer ---
    ax5 = plt.subplot(3, 4, 5)
    sc2 = ax5.scatter(x_cut, y_cut, c=layer_cut, s=1, cmap="viridis", 
                      alpha=0.5, vmin=0, vmax=geometry.num_layers-1)
    ax5.set_xlabel("x [mm] (beam direction)")
    ax5.set_ylabel("y [mm]")
    ax5.set_title("Hits Colored by Layer")
    plt.colorbar(sc2, ax=ax5, label="Layer Number")
    
    # --- 6. x-z View ---
    ax6 = plt.subplot(3, 4, 6)
    sc3 = ax6.scatter(x_cut, z_cut, c=energy_cut * 1e3, s=1, 
                      cmap="plasma", alpha=0.6, norm=norm)
    ax6.set_xlabel("x [mm] (beam direction)")
    ax6.set_ylabel("z [mm]")
    ax6.set_title("x-z View")
    plt.colorbar(sc3, ax=ax6, label="E [MeV]")
    
    # --- 7. Layer vs x Position ---
    ax7 = plt.subplot(3, 4, 7)
    sc4 = ax7.scatter(x_cut, layer_cut, c=energy_cut * 1e3, s=1, 
                      cmap="plasma", alpha=0.6, norm=norm)
    ax7.set_xlabel("x [mm] (beam direction)")
    ax7.set_ylabel("Layer Number")
    ax7.set_title("Longitudinal Shower Development")
    ax7.set_ylim(-0.5, geometry.num_layers - 0.5)
    plt.colorbar(sc4, ax=ax7, label="E [MeV]")
    
    # --- 8. Energy per Layer ---
    ax8 = plt.subplot(3, 4, 8)
    layer_analysis = analyze_layers(data, geometry)
    ax8.bar(range(1, geometry.num_layers+1), layer_analysis['energy_per_layer'], 
            color='orangered', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax8.axvline(layer_analysis['shower_max_layer']+1, color='blue', 
                linestyle='--', linewidth=2, label=f"Shower max (layer {layer_analysis['shower_max_layer']+1})")
    ax8.set_xlabel("Layer Number")
    ax8.set_ylabel("Energy Deposited [GeV]")
    ax8.set_title("Radial Energy Profile")
    ax8.set_xlim(0.5, geometry.num_layers + 0.5)
    ax8.legend(fontsize=8)
    
    # --- 9. Hits per Layer ---
    ax9 = plt.subplot(3, 4, 9)
    ax9.bar(range(1, geometry.num_layers+1), layer_analysis['hits_per_layer'], 
            color='forestgreen', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax9.set_xlabel("Layer Number")
    ax9.set_ylabel("Number of Hits")
    ax9.set_title("Hit Multiplicity per Layer")
    ax9.set_xlim(0.5, geometry.num_layers + 0.5)
    
    # --- 10. Depth Distribution (Radius or Z) ---
    ax10 = plt.subplot(3, 4, 10)
    ax10.hist(depth_coord, bins=50, histtype='step', linewidth=1.5, color='purple')
    if is_simplebox:
        ax10.axvline(geometry.z_start, color='red', linestyle='--',
                     label=f'z_start = {geometry.z_start} mm')
        ax10.axvline(geometry.z_end, color='red', linestyle='--',
                     label=f'z_end = {geometry.z_end:.0f} mm')
        ax10.set_xlabel("z [mm]")
        ax10.set_title("Depth (z) Distribution")
    else:
        ax10.axvline(geometry.rmin, color='red', linestyle='--',
                     label=f'r_min = {geometry.rmin} mm')
        ax10.axvline(geometry.rmax, color='red', linestyle='--',
                     label=f'r_max = {geometry.rmax:.0f} mm')
        ax10.set_xlabel("Radius r [mm]")
        ax10.set_title("Radial Distribution")
    ax10.set_ylabel("Hits")
    ax10.legend(fontsize=8)
    
    # --- 11. Energy vs Radius ---
    ax11 = plt.subplot(3, 4, 11)
    ax11.hist(time, bins=np.logspace(np.log10(time.min()), np.log10(time.max()), 100), histtype='step', linewidth=1.5, color='brown', log=True)  # Add log=True
    ax11.set_xscale('log')
    ax11.set_xlabel("Hit Time [ns]")
    ax11.set_ylabel("Number of Hits")
    ax11.set_title("Hit Time Distribution")
    
    # --- 12. 2D Histogram: Layer vs x ---
    ax12 = plt.subplot(3, 4, 12)
    h = ax12.hist2d(x_cut, layer_cut, bins=[100, geometry.num_layers],
                    cmap='hot', cmin=1)
    ax12.set_xlabel("x [mm] (beam direction)")
    ax12.set_ylabel("Layer Number")
    ax12.set_title("2D Shower Profile")
    ax12.set_ylim(-0.5, geometry.num_layers - 0.5)
    plt.colorbar(h[3], ax=ax12, label="Hits")
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE ANALYSIS")
    print(f"{'='*60}")
    print(f"Total hits:              {len(energy_cut):,}")
    print(f"Total energy:            {layer_analysis['total_energy']:.3f} GeV")
    print(f"Shower max layer:        {layer_analysis['shower_max_layer']}")
    print(f"Layer range:             {layer_cut.min()+1} - {layer_cut.max()+1}")
    print(f"Unique layers hit:       {len(np.unique(layer_cut))}")
    if is_simplebox:
        print(f"Depth (z) range:         {depth_coord.min():.1f} - {depth_coord.max():.1f} mm")
    else:
        print(f"Radius range:            {depth_coord.min():.1f} - {depth_coord.max():.1f} mm")
    print(f'Time range:              {time.min():.2f} - {time.max():.2f} ns')
    print(f"{'='*60}\n")
    
    plt.tight_layout()
    return fig

def plot_clustering_comparison(data_before, data_after, layer_to_plot=None, figsize=(16, 12)):
    """
    Visualize hits before and after clustering.
    
    Parameters:
    -----------
    data_before : dict
        Original unclustered data
    data_after : dict
        Clustered data
    layer_to_plot : int or None
        Specific layer to show detailed view. If None, shows aggregate.
    """
    fig = plt.figure(figsize=figsize)
    
    # Extract data
    x_before = data_before['x']
    y_before = data_before['y']
    e_before = data_before['energy']
    layer_before = data_before['layer']
    
    x_after = data_after['x']
    y_after = data_after['y']
    e_after = data_after['energy']
    layer_after = data_after['layer']
    
    # --- Row 1: Full detector views ---
    
    # Before clustering - all layers
    ax1 = plt.subplot(3, 3, 1)
    sc1 = ax1.scatter(x_before, y_before, c=np.log10(e_before*1e3 + 0.1), 
                      s=0.5, cmap='plasma', alpha=0.4, vmin=-1, vmax=2)
    ax1.set_xlim(x_before.min()-10, x_before.max()+10)
    ax1.set_ylim(y_before.min()-10, y_before.max()+10)
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.set_title(f"Before Clustering\n({len(e_before):,} hits)")
    ax1.set_aspect('equal')
    plt.colorbar(sc1, ax=ax1, label="log₁₀(E [MeV])")
    
    # After clustering - all layers
    ax2 = plt.subplot(3, 3, 2)
    sc2 = ax2.scatter(x_after, y_after, c=np.log10(e_after*1e3 + 0.1),
                      s=2, cmap='plasma', alpha=0.6, vmin=-1, vmax=2)
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("y [mm]")
    ax2.set_title(f"After Clustering ({data_after.get('cell_size', 1):.1f}mm cells)\n({len(e_after):,} hits)")
    ax2.set_aspect('equal')
    plt.colorbar(sc2, ax=ax2, label="log₁₀(E [MeV])")
    
    # Hits per layer comparison
    ax3 = plt.subplot(3, 3, 3)
    unique_layers = np.unique(layer_before)
    hits_before = [np.sum(layer_before == lay) for lay in unique_layers]
    hits_after = [np.sum(layer_after == lay) for lay in unique_layers]
    
    x_pos = np.arange(len(unique_layers))
    width = 0.35
    ax3.bar(x_pos - width/2, hits_before, width, label='Before', alpha=0.7, color='blue')
    ax3.bar(x_pos + width/2, hits_after, width, label='After', alpha=0.7, color='red')
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Number of Hits")
    ax3.set_title("Hits per Layer")
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # --- Row 2: Single layer detailed view ---
    if layer_to_plot is None:
        # Find layer with most energy
        energy_per_layer = [e_before[layer_before == lay].sum() for lay in unique_layers]
        layer_to_plot = unique_layers[np.argmax(energy_per_layer)]
    
    mask_before = layer_before == layer_to_plot
    mask_after = layer_after == layer_to_plot
    
    # Before - single layer
    ax4 = plt.subplot(3, 3, 4)
    sc4 = ax4.scatter(x_before[mask_before], y_before[mask_before], 
                      c=np.log10(e_before[mask_before]*1e3 + 0.1),
                      s=1, cmap='plasma', alpha=0.5, vmin=-1, vmax=2)
    ax4.set_xlabel("x [mm]")
    ax4.set_ylabel("y [mm]")
    ax4.set_title(f"Layer {layer_to_plot} - Before\n({np.sum(mask_before):,} hits)")
    ax4.set_aspect('equal')
    plt.colorbar(sc4, ax=ax4, label="log₁₀(E [MeV])")
    
    # After - single layer
    ax5 = plt.subplot(3, 3, 5)
    sc5 = ax5.scatter(x_after[mask_after], y_after[mask_after],
                      c=np.log10(e_after[mask_after]*1e3 + 0.1),
                      s=5, cmap='plasma', alpha=0.7, vmin=-1, vmax=2)
    ax5.set_xlabel("x [mm]")
    ax5.set_ylabel("y [mm]")
    ax5.set_title(f"Layer {layer_to_plot} - After\n({np.sum(mask_after):,} cells)")
    ax5.set_aspect('equal')
    plt.colorbar(sc5, ax=ax5, label="log₁₀(E [MeV])")
    
    # Compression ratio per layer
    ax6 = plt.subplot(3, 3, 6)
    compression = [hits_before[i]/hits_after[i] if hits_after[i] > 0 else 0 
                   for i in range(len(unique_layers))]
    ax6.plot(unique_layers, compression, 'o-', linewidth=2, markersize=6)
    ax6.set_xlabel("Layer")
    ax6.set_ylabel("Compression Ratio")
    ax6.set_title("Clustering Compression")
    ax6.grid(True, alpha=0.3)
    ax6.axhline(1, color='red', linestyle='--', alpha=0.5)
    
    # --- Row 3: Energy distributions ---
    
    # Hit energy spectrum
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(e_before*1e3, bins=np.logspace(-2, np.log10(e_before.max()*1e3), 50),
             histtype='step', label='Before', linewidth=2, alpha=0.7)
    ax7.hist(e_after*1e3, bins=np.logspace(-2, np.log10(e_after.max()*1e3), 50),
             histtype='step', label='After', linewidth=2, alpha=0.7)
    ax7.set_xscale('log')
    ax7.set_yscale('log')
    ax7.set_xlabel("Hit/Cell Energy [MeV]")
    ax7.set_ylabel("Counts")
    ax7.set_title("Energy Spectrum")
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Energy per layer
    ax8 = plt.subplot(3, 3, 8)
    energy_before = [e_before[layer_before == lay].sum() for lay in unique_layers]
    energy_after = [e_after[layer_after == lay].sum() for lay in unique_layers]
    ax8.plot(unique_layers, energy_before, 'o-', label='Before', linewidth=2, markersize=6)
    ax8.plot(unique_layers, energy_after, 's-', label='After', linewidth=2, markersize=6)
    ax8.set_xlabel("Layer")
    ax8.set_ylabel("Energy [GeV]")
    ax8.set_title("Energy per Layer")
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 2D histogram of single layer (after clustering)
    ax9 = plt.subplot(3, 3, 9)
    if np.sum(mask_after) > 0:
        x_range = [x_after[mask_after].min(), x_after[mask_after].max()]
        y_range = [y_after[mask_after].min(), y_after[mask_after].max()]
        h = ax9.hist2d(x_after[mask_after], y_after[mask_after],
                       weights=e_after[mask_after]*1e3,
                       bins=50, cmap='hot', cmin=0.1)
        ax9.set_xlabel("x [mm]")
        ax9.set_ylabel("y [mm]")
        ax9.set_title(f"Layer {layer_to_plot} Energy Density\n(Clustered)")
        plt.colorbar(h[3], ax=ax9, label="Energy [MeV]")
    
    plt.tight_layout()
    return fig

def plot_event_views(x, y, z, energy, figsize=(12, 4), cmap_name="plasma"):
    """
    Show XY, XZ, and YZ projections of a calorimeter shower in one row.
    
    Parameters
    ----------
    x, y, z : array-like
        Hit coordinates.
    energy : array-like
        Energy deposit per hit (used for color).
    figsize : tuple
        Size of the matplotlib figure.
    cmap_name : str
        Matplotlib colormap name.
    """
    # Convert to numpy arrays and mask invalid values
    x, y, z, energy = map(np.asarray, (x, y, z, energy))
    mask = np.isfinite(energy) & (energy > 0)
    x, y, z, energy = x[mask], y[mask], z[mask], energy[mask]

    if len(energy) == 0:
        raise ValueError("No valid energy hits to plot.")

    # Normalize color scale (shared for all panels)
    vmin, vmax = np.min(energy), np.max(energy)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    # Keep handle to one scatter for colorbar
    sc0 = axes[0].scatter(x, y, c=energy, s=8, cmap=cmap, norm=norm)
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    axes[0].set_title("XY view (Transverse)")

    sc1 = axes[1].scatter(z, x, c=energy, s=8, cmap=cmap, norm=norm)
    axes[1].set_xlabel("z [layers]")
    axes[1].set_ylabel("x [mm]")
    axes[1].set_title("XZ view (Side)")

    sc2 = axes[2].scatter( y, z, c=energy, s=8, cmap=cmap, norm=norm)
    axes[2].set_xlabel("y [mm]")
    axes[2].set_ylabel("z [layers]")
    axes[2].set_title("YZ view (Side)")

    # Shared colorbar using one of the scatter plots
    cbar = fig.colorbar(sc0, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("Energy [GeV]")

    plt.show()