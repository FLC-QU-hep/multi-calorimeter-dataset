#!/usr/bin/env python3
"""
Process a single calorimeter configuration.
Designed to run as SLURM job array.
"""

import argparse
import os
import sys
from pathlib import Path
import uproot
import json

REPO_ROOT = Path(os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO_ROOT))
from utils import preprocessing_data as prep
from utils import calo_geometry

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = REPO_ROOT

def build_config(ref_dir="sf_nlayers_angles", sub_dir="SiW_test", n_digits=4, n_showers=400):
    """Build CONFIG dict for the given reference directory."""
    fmt = f"0{n_digits}d"
    return {
        'root_dir': BASE_DIR / f"output_dataset/SimpleBox/root/{ref_dir}/{sub_dir}",
        'h5_temp_dir': BASE_DIR / f"output_dataset/SimpleBox/h5/{ref_dir}/{sub_dir}/temp",
        'xml_dir': BASE_DIR / f"calo_configs/par04/SimpleBox/{ref_dir}/{sub_dir}",
        'metadata_json': BASE_DIR / f"calo_configs/par04/SimpleBox/{ref_dir}/{sub_dir}/final_metadata.json",
        'root_pattern': f"{n_showers}_1-100GeV_SiW_xml_{{config_id:{fmt}}}.root",
        'xml_pattern': f"SimpleBox_config_{{config_id:{fmt}}}.xml",
        'h5_pattern': f"config_{{config_id:{fmt}}}.h5",

        # Processing parameters
        'cell_size': 1.0,
        'energy_threshold': 0.01,
        'apply_clustering': True,
        'apply_threshold_after_clustering': True,
        'cone_angle': None,
    }


# ============================================================================
# MAIN
# ============================================================================

def process_config(config_id, config):
    """Process a single configuration."""

    # Paths (using patterns from config)
    root_file = config['root_dir'] / config['root_pattern'].format(config_id=config_id)
    xml_file = config['xml_dir'] / config['xml_pattern'].format(config_id=config_id)
    temp_h5_file = config['h5_temp_dir'] / config['h5_pattern'].format(config_id=config_id)

    # Create temp directory
    config['h5_temp_dir'].mkdir(parents=True, exist_ok=True)

    # Skip if already processed
    if temp_h5_file.exists():
        print(f"✓ Config {config_id}: Already processed, skipping...")
        return

    # Check files exist
    if not root_file.exists():
        print(f"✗ Config {config_id}: ROOT file not found!")
        sys.exit(1)

    if not xml_file.exists():
        print(f"✗ Config {config_id}: XML file not found!")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"Processing Config ID: {config_id}")
    print(f"{'='*60}")

    # Open ROOT file
    file = uproot.open(root_file)
    tree = file["events"]

    # Load ECal hit data
    ecal_steps = tree.arrays([
        "ECalBarrelCollectionContributions/ECalBarrelCollectionContributions.energy",
        "ECalBarrelCollectionContributions/ECalBarrelCollectionContributions.stepPosition.x",
        "ECalBarrelCollectionContributions/ECalBarrelCollectionContributions.stepPosition.y",
        "ECalBarrelCollectionContributions/ECalBarrelCollectionContributions.stepPosition.z",
        "ECalBarrelCollectionContributions/ECalBarrelCollectionContributions.time",
    ], library="ak")

    # Load geometry
    geometry = calo_geometry.SimpleBoxGeometry.from_xml(xml_file)

    # Load MCParticle data
    mc_data = tree.arrays([
        "MCParticles/MCParticles.momentum.x",
        "MCParticles/MCParticles.momentum.y",
        "MCParticles/MCParticles.momentum.z",
        "MCParticles/MCParticles.generatorStatus",
        "MCParticles/MCParticles.PDG",
    ], library="ak")

    # Extract incident energy and momentum 3-vector
    incident_energy_np, incident_momentum_np = prep.get_incident_momentum(mc_data, verbose=False)

    # Process and save
    prep.process_and_save_batch(
        arrays=ecal_steps,
        geometry=geometry,
        output_path=temp_h5_file,
        cone_angle=config['cone_angle'],
        cell_size=config['cell_size'],
        incident_energies=incident_energy_np,
        incident_momentum=incident_momentum_np,
        apply_clustering=config['apply_clustering'],
        apply_threshold_after_clustering=config['apply_threshold_after_clustering'],
        energy_threshold=config['energy_threshold'],
    )

    print(f"✓ Config {config_id} processed successfully!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process single calorimeter config')
    parser.add_argument('--config_id', type=int, required=True,
                       help='Configuration ID to process')
    parser.add_argument('--ref-dir', type=str, default='sf_nlayers_angles',
                       help='Reference directory (default: sf_nlayers_angles)')
    parser.add_argument('--sub-dir', type=str, default='SiW_test',
                       help='Sub-directory (default: SiW_test)')
    parser.add_argument('--n-digits', type=int, default=4,
                       help='Number of digits for config ID formatting (default: 4)')
    parser.add_argument('--n-showers', type=int, default=400,
                       help='Number of showers per config, must match N_EVENTS in sim script (default: 400)')
    args = parser.parse_args()

    config = build_config(args.ref_dir, args.sub_dir, args.n_digits, args.n_showers)
    process_config(args.config_id, config)