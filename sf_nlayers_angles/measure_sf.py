# measure_sf.py
import os
import numpy as np
import awkward as ak
import uproot
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SUBFIX = "SiW_final"

REPO_ROOT = Path(os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[1])))
CONFIG_DIR = REPO_ROOT / f"calo_configs/par04/SimpleBox/{SUBFIX}"
BASE_DATA_DIR = REPO_ROOT / "output_dataset/SimpleBox/root/"


DATA_DIR = BASE_DATA_DIR / SUBFIX
PLOT_DIR = CONFIG_DIR / "plots"
DIAG_DIR = PLOT_DIR / "individual"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# Load metadata
with open(CONFIG_DIR / "final_metadata.json") as f:
    configs = json.load(f)

def get_incident_energies(data):
    """Extract incident energies from MCParticles"""
    px = data["MCParticles/MCParticles.momentum.x"]
    py = data["MCParticles/MCParticles.momentum.y"]
    pz = data["MCParticles/MCParticles.momentum.z"]
    gen_status = data["MCParticles/MCParticles.generatorStatus"]

    # Filter: keep only primary particles (generatorStatus == 1)
    mask = gen_status == 1

    px_primary = px[mask]
    py_primary = py[mask]
    pz_primary = pz[mask]

    # Calculate momentum magnitude for primary particles
    p_mag_primary = np.sqrt(px_primary**2 + py_primary**2 + pz_primary**2)

    # For photons/electrons: E = |p| (assuming massless or relativistic)
    incident_energy = ak.flatten(p_mag_primary)

    return np.array(incident_energy)

results = []

print(f"\n{'='*60}")
print("MEASURING SAMPLING FRACTIONS")
print(f"{'='*60}\n")

for config in configs:
    xml_id = config['id']
    file_path = DATA_DIR / f"100_1-100GeV_SiW_xml_{xml_id:03d}.root"
    
    if not file_path.exists():
        print(f"❌ Missing: {file_path.name}")
        continue
    
    # Read ROOT file
    file = uproot.open(file_path)
    tree = file["events"]
    
    # Get incident energy
    data = tree.arrays([
        "MCParticles/MCParticles.momentum.x",
        "MCParticles/MCParticles.momentum.y",
        "MCParticles/MCParticles.momentum.z",
        "MCParticles/MCParticles.generatorStatus",
        "MCParticles/MCParticles.PDG",
    ], library="ak")
    E_inc = get_incident_energies(data)
    
    # Get deposited energy
    arrays = tree.arrays([
        "ECalBarrelCollectionContributions/ECalBarrelCollectionContributions.energy"
    ], library="ak")
    E_dep = ak.sum(
        arrays["ECalBarrelCollectionContributions/ECalBarrelCollectionContributions.energy"], 
        axis=1
    ).to_numpy()
    
    # Calculate SF
    sf = E_dep / E_inc

    # Filter out invalid values (inf, nan) caused by zero E_inc
    valid_mask = np.isfinite(sf) & (E_inc > 0)
    sf_valid = sf[valid_mask]
    E_inc_valid = E_inc[valid_mask]
    E_dep_valid = E_dep[valid_mask]

    if len(sf_valid) == 0:
        print(f"⚠️  Config {xml_id:02d}: No valid events (all E_inc = 0)")
        continue

    results.append({
        'id': xml_id,
        'active_mm': config['active_mm'],
        'passive_mm': config['passive_mm'],
        'sf_mean': float(sf_valid.mean()),
        'sf_std': float(sf_valid.std())
    })

    # Individual diagnostic plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(sf_valid, bins=50, alpha=1, color='b', histtype='step')
    ax.axvline(sf_valid.mean(), color='r', ls='--', lw=2, label=f'Mean={sf_valid.mean():.4f}')
    ax.fill_betweenx([0, ax.get_ylim()[1]], sf_valid.mean()-sf_valid.std(), sf_valid.mean()+sf_valid.std(), 
                     color='orange', alpha=0.2, label=f'Std={sf_valid.std():.4f}')

    ax.set_xlabel('E_dep / E_inc', fontweight='bold', fontsize=16)
    ax.set_ylabel('Events', fontweight='bold', fontsize=16)
    ax.set_title(f'Config {xml_id:02d}: Energy Ratio', fontweight='bold', fontsize=20)
    ax.legend(fontsize=14)

    ax = axes[1]
    ax.scatter(E_inc_valid, E_dep_valid, alpha=0.5, s=30, c=sf_valid, cmap='viridis')
    ax.plot([0, 100], [0, 100*sf_valid.mean()], 'r--', lw=2)
    ax.set_xlabel('E_incident [GeV]', fontweight='bold', fontsize=16)
    ax.set_ylabel('E_deposited [GeV]', fontweight='bold', fontsize=16)
    ax.set_title(f'Energy Correlation (SF={sf_valid.mean():.4f})', fontweight='bold', fontsize=20)
    plt.colorbar(ax.collections[0], ax=ax, label='SF')

    plt.tight_layout()
    plt.savefig(DIAG_DIR / f"config_{xml_id:02d}.png", dpi=120)
    plt.close()

    print(f"Config {xml_id:02d}: SF={sf_valid.mean():.4f}±{sf_valid.std():.4f}, "
          f"t_a={config['active_mm']:.2f}mm, t_p={config['passive_mm']:.2f}mm")

# Save results
with open(CONFIG_DIR / "measured_sf.json", 'w') as f:
    json.dump(results, f, indent=2)

# Summary plot
sf_vals = [r['sf_mean'] for r in results]
t_a = [r['active_mm'] for r in results]
t_p = [r['passive_mm'] for r in results]

if len(results) == 0:
    print(f"\n{'='*60}")
    print("❌ ERROR: No valid configurations were processed!")
    print("All configurations had zero incident energy (E_inc = 0)")
    print(f"{'='*60}\n")
    exit(1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
ax.scatter(sf_vals, t_a, s=150, alpha=0.7, edgecolors='black', linewidths=2)
ax.set_xlabel('Measured SF', fontweight='bold', fontsize=16)
ax.set_ylabel('Active [mm]', fontweight='bold', fontsize=16)
ax.set_title('Active vs SF', fontweight='bold', fontsize=20)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(sf_vals, t_p, s=150, alpha=0.7, edgecolors='black', linewidths=2)
ax.set_xlabel('Measured SF', fontweight='bold', fontsize=16)
ax.set_ylabel('Passive [mm]', fontweight='bold', fontsize=16)
ax.set_title('Passive vs SF', fontweight='bold', fontsize=20)
ax.grid(True, alpha=0.3)

ax = axes[2]
sc = ax.scatter(t_a, t_p, c=sf_vals, s=200, edgecolors='black', linewidths=2, cmap='plasma')
plt.colorbar(sc, ax=ax, label='SF')
ax.set_xlabel('Active [mm]', fontweight='bold', fontsize=16)
ax.set_ylabel('Passive [mm]', fontweight='bold', fontsize=16)
ax.set_title('Thickness Space', fontweight='bold', fontsize=20)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "sf_summary.png", dpi=150)
plt.close()

print(f"\n{'='*60}")
print(f"SF range: [{min(sf_vals):.4f}, {max(sf_vals):.4f}]")
print(f"Saved: {CONFIG_DIR / 'measured_sf.json'}")
print(f"Plots: {DIAG_DIR} (individual), {PLOT_DIR / 'sf_summary.png'}")
print(f"{'='*60}\n")