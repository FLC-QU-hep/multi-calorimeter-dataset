# utils/plotting_configs.py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

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

def plot_configs(configs, output_dir, num_layers):
    """
    Plot sampling fraction vs active/passive/total thicknesses and save metadata.

    Parameters
    ----------
    configs : list of tuples
        Each entry: (sampling_fraction, active_thickness, passive_thickness)
    output_dir : Path
        Directory where plots and metadata will be saved.
    num_layers : int
        Number of calorimeter layers.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Unpack data
    sf, at, pt = zip(*configs)
    sf, at, pt = np.array(sf), np.array(at), np.array(pt)
    total_thickness = (at + pt) * (num_layers-1)  # Exclude last overshooting layer

    # Write metadata file
    info_file = output_dir / "sampling_fraction_thickness_summary.txt"
    with open(info_file, "w") as fh:
        fh.write(f"Plot generation timestamp: {datetime.now().isoformat()}\n")
        fh.write(f"Number of configurations: {len(configs)}\n")
        fh.write(f"Number of layers: {num_layers-1}\n\n")

        fh.write("Sampling fraction statistics:\n")
        fh.write(f"  Mean: {np.mean(sf):.4f}\n")
        fh.write(f"  Median: {np.median(sf):.4f}\n")
        fh.write(f"  Min: {np.min(sf):.4f}\n")
        fh.write(f"  Max: {np.max(sf):.4f}\n\n")

        fh.write("Thickness summary per configuration:\n")
        for i, (f, a, p, t) in enumerate(zip(sf, at, pt, total_thickness)):
            fh.write(f"  Config {i:03d}: f={f:.4f}, active={a:.4f}mm, passive={p:.4f}mm, total={t:.2f}mm\n")

    # Create figure
    fig = plt.figure(figsize=(10, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    
    # Top row: active + passive
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    # Bottom row: total
    ax3 = fig.add_subplot(grid[1, :])

    # Passive
    ax1.scatter(sf, pt, color='blue', s=1)
    ax1.set_xlabel("Sampling fraction")
    ax1.set_ylabel("Passive thickness [mm]")

    # Active
    ax2.scatter(sf, at, color='green', s=1)
    ax2.set_xlabel("Sampling fraction")
    ax2.set_ylabel("Active thickness [mm]")

    # Total
    ax3.scatter(sf, total_thickness, color='red', s=1)
    ax3.set_xlabel("Sampling fraction")
    ax3.set_ylabel("Total calorimeter thickness [mm]")

    plt.tight_layout()
    plot_path = output_dir / "sampling_fraction_vs_thicknesses.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Plots saved to {plot_path}")