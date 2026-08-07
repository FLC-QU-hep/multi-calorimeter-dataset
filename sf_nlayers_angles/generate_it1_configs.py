# generate_iteration1_configs.py
import os
import numpy as np
from pathlib import Path
from string import Template
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[1])))

# Material properties (Radiation lengths)
X0_SI = 93.6   # mm (Silicon)
X0_W = 3.5     # mm (Tungsten)

# Fixed detector parameters
NUM_LAYERS = 30
N_X0_TOTAL = 30  # Total radiation lengths in detector

# Calibration sampling
N_CONFIGS = 100
# A_RANGE = [0.0018, 0.016]  # Fraction parameter range
A_RANGE = [0.0023, 0.0152]  # Updated fraction parameter range

OUTPUT_DIR = REPO_ROOT / "calo_configs/par04/SimpleBox/SiW_iter1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

template = Template((REPO_ROOT / "calo_configs/base/SimpleBox_template.xml").read_text())

print(f"\n{'='*60}")
print("PHYSICS-BASED CALIBRATION CONFIG GENERATION")
print(f"{'='*60}")
print(f"Approach: Sample fraction parameter 'a'")
print(f"  t_active  = n_X0 × X0_Si × a / n_layers")
print(f"  t_passive = n_X0 × X0_W × (1-a) / n_layers")
print(f"\nParameters:")
print(f"  X0_Si = {X0_SI} mm")
print(f"  X0_W  = {X0_W} mm")
print(f"  n_X0  = {N_X0_TOTAL}")
print(f"  n_layers = {NUM_LAYERS}")
print(f"{'='*60}\n")

# Sample 'a' parameter uniformly
np.random.seed(911)
a_values = np.random.uniform(A_RANGE[0], A_RANGE[1], N_CONFIGS)

configs = []
for i, a in enumerate(a_values):
    # Calculate thicknesses from radiation length formula
    t_active = N_X0_TOTAL * X0_SI * a / NUM_LAYERS
    t_passive = N_X0_TOTAL * X0_W * (1 - a) / NUM_LAYERS
    
    configs.append({
        'id': i,
        'a_parameter': float(a),
        'active_mm': float(t_active),
        'passive_mm': float(t_passive)
    })

# Generate XMLs
for config in configs:
    xml_id = config['id']
    t_a = config['active_mm']
    t_p = config['passive_mm']
    
    # Placeholder SF
    placeholder_sf = 0.03
    
    xml = template.substitute(
        num_layers=NUM_LAYERS,
        sampling_fraction=f"{placeholder_sf:.10f}",
        active_thickness=f"{t_a:.10f}",
        passive_thickness=f"{t_p:.10f}",
        elements_path="../../../base/elements.xml",
        materials_path="../../../base/materials.xml"
    )
    
    (OUTPUT_DIR / f"SimpleBox_config_{xml_id:03d}.xml").write_text(xml)

# Save metadata
with open(OUTPUT_DIR / "configs_metadata.json", 'w') as f:
    json.dump(configs, f, indent=2)

# Diagnostic plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Parameter space
ax = axes[0]
a_vals = [c['a_parameter'] for c in configs]
ax.hist(a_vals, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
ax.set_xlabel('Parameter a', fontsize=14, fontweight='bold')
ax.set_ylabel('Count', fontsize=14, fontweight='bold')
ax.set_title('Sampled Parameter Distribution', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Thickness space
ax = axes[1]
t_a = [c['active_mm'] for c in configs]
t_p = [c['passive_mm'] for c in configs]
ax.scatter(t_a, t_p, s=200, alpha=0.7, edgecolors='black', linewidths=2, c=a_vals, cmap='viridis')
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('Parameter a', fontsize=12, fontweight='bold')
ax.set_xlabel('Active [mm]', fontsize=14, fontweight='bold')
ax.set_ylabel('Passive [mm]', fontsize=14, fontweight='bold')
ax.set_title('Thickness Space (colored by a)', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 3: a vs thickness components
ax = axes[2]
ax.scatter(a_vals, t_a, s=100, alpha=0.7, label='Active', edgecolors='black', linewidths=1.5)
ax.scatter(a_vals, t_p, s=100, alpha=0.7, label='Passive', edgecolors='black', linewidths=1.5)
ax.set_xlabel('Parameter a', fontsize=14, fontweight='bold')
ax.set_ylabel('Thickness [mm]', fontsize=14, fontweight='bold')
ax.set_title('Thickness vs Parameter', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "parameter_space.png", dpi=150)
plt.close()

print(f"{'='*60}")
print("GENERATED CONFIGURATIONS")
print(f"{'='*60}")
print(f"N configs: {N_CONFIGS}")
print(f"Parameter a: [{min(a_vals):.5f}, {max(a_vals):.5f}]")
print(f"Active:      [{min(t_a):.3f}, {max(t_a):.3f}] mm")
print(f"Passive:     [{min(t_p):.3f}, {max(t_p):.3f}] mm")
print(f"Output: {OUTPUT_DIR}")
print(f"Plot: {PLOT_DIR / 'parameter_space.png'}")
print(f"{'='*60}\n")