# generate_final_configs.py
import os
import numpy as np
import json
import pickle
from pathlib import Path
from string import Template
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[1])))

# Material properties
X0_SI = 93.6   # mm (Silicon)
X0_W = 3.5     # mm (Tungsten)

# Fixed detector parameters
NUM_LAYERS = 30
N_X0_TOTAL = 30

# Configuration
MODEL_PATH = REPO_ROOT / "calo_configs/par04/SimpleBox/SiW_iter1/model.pkl"
OUTPUT_DIR = REPO_ROOT / "calo_configs/par04/SimpleBox/sf_nlayers_angles/SiW_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SF_RANGE = [0.01, 0.05]
N_CONFIGS = 10000  # 10000 configs x 400 events = 4M showers

print(f"\n{'='*60}")
print(f"PHYSICS-BASED FINAL CONFIGURATION GENERATION")
print(f"{'='*60}")
print(f"Target SF range: {SF_RANGE}")
print(f"N configs: {N_CONFIGS}")
print(f"{'='*60}\n")

# Load model: a = f(SF)
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Sample target SF values uniformly
np.random.seed(1234)
target_sf = np.random.uniform(SF_RANGE[0], SF_RANGE[1], N_CONFIGS)

# Add small random perturbation for diversity
target_sf += np.random.uniform(-0.001, 0.001, N_CONFIGS)
target_sf = np.clip(target_sf, SF_RANGE[0], SF_RANGE[1])

configs = []
template = Template((REPO_ROOT / "calo_configs/base/SimpleBox_template.xml").read_text())

for i, sf_target in enumerate(target_sf):
    # Predict parameter 'a' from target SF
    a_pred = model.predict(np.array([[sf_target]]))[0]
    
    # Calculate thicknesses from radiation length formula
    t_active = N_X0_TOTAL * X0_SI * a_pred / NUM_LAYERS
    t_passive = N_X0_TOTAL * X0_W * (1 - a_pred) / NUM_LAYERS
    
    configs.append({
        'id': i,
        'target_sf': float(sf_target),
        'predicted_a': float(a_pred),
        'active_mm': float(t_active),
        'passive_mm': float(t_passive),
        'layer_thickness_mm': float(t_active + t_passive),
        'total_depth_mm': float((t_active + t_passive) * NUM_LAYERS)
    })
    
    # Generate XML
    xml = template.substitute(
        num_layers=NUM_LAYERS,
        sampling_fraction=f"{sf_target:.10f}",
        active_thickness=f"{t_active:.10f}",
        passive_thickness=f"{t_passive:.10f}",
        elements_path="../../../base/elements.xml",
        materials_path="../../../base/materials.xml"
    )
    (OUTPUT_DIR / f"SimpleBox_config_{i:04d}.xml").write_text(xml)

# Save metadata
with open(OUTPUT_DIR / "final_metadata.json", 'w') as f:
    json.dump(configs, f, indent=2)

# Diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

sf_vals = [c['target_sf'] for c in configs]
a_vals = [c['predicted_a'] for c in configs]
t_a = [c['active_mm'] for c in configs]
t_p = [c['passive_mm'] for c in configs]

# 1. SF distribution
ax = axes[0,0]
ax.hist(sf_vals, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
ax.set_xlabel('Target SF', fontsize=14, fontweight='bold')
ax.set_ylabel('Count', fontsize=14, fontweight='bold')
ax.set_title('Target SF Distribution', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 2. Parameter 'a' distribution
ax = axes[0,1]
ax.hist(a_vals, bins=20, edgecolor='black', alpha=0.7, color='coral')
ax.set_xlabel('Predicted a', fontsize=14, fontweight='bold')
ax.set_ylabel('Count', fontsize=14, fontweight='bold')
ax.set_title('Predicted Parameter Distribution', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 3. Thickness space (colored by SF)
ax = axes[1,0]
sc = ax.scatter(t_a, t_p, c=sf_vals, s=100, edgecolors='black', linewidths=1.5, cmap='plasma')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Target SF', fontsize=12, fontweight='bold')
ax.set_xlabel('Active [mm]', fontsize=14, fontweight='bold')
ax.set_ylabel('Passive [mm]', fontsize=14, fontweight='bold')
ax.set_title('Final Configs: Thickness Space', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)

# 4. SF vs a (final mapping)
ax = axes[1,1]
ax.scatter(sf_vals, a_vals, s=100, alpha=0.7, edgecolors='black', linewidths=1.5)
ax.set_xlabel('Target SF', fontsize=14, fontweight='bold')
ax.set_ylabel('Predicted a', fontsize=14, fontweight='bold')
ax.set_title('SF → a Mapping', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "final_configs_summary.png", dpi=150)
plt.close()

print(f"{'='*60}")
print("GENERATION COMPLETE")
print(f"{'='*60}")
print(f"Generated: {N_CONFIGS} configurations")
print(f"SF range: [{min(sf_vals):.4f}, {max(sf_vals):.4f}]")
print(f"Active:   [{min(t_a):.3f}, {max(t_a):.3f}] mm")
print(f"Passive:  [{min(t_p):.3f}, {max(t_p):.3f}] mm")
print(f"Output: {OUTPUT_DIR}")
print(f"Plot: {OUTPUT_DIR / 'final_configs_summary.png'}")
print(f"{'='*60}\n")