#!/usr/bin/env python3
"""
ROOT -> per-config temp H5 -> showerdata training-ready format.

Usage:
    python multi-dataset_h5_creation.py                              # sf_nlayers_angles (default)
    python multi-dataset_h5_creation.py --ref-dir sf_nlayers_angles  # angles variant
"""

import argparse
import json
import os
import h5py
import uproot
from pathlib import Path
from tqdm import tqdm
import sys

REPO_ROOT = Path(os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO_ROOT))
from utils import preprocessing_data as prep
from utils import calo_geometry
from utils.showerdata_utils import combine_temp_to_showerdata

BASE_DIR = REPO_ROOT
N_SHOWERS_DEFAULT = 200


def build_config(ref_dir="sf_nlayers", sub_dir="SiW_final", n_digits=4,
                 n_showers=None, metadata_file="final_metadata.json", output_name=None):
    """Build CONFIG dict for the given reference directory."""
    n_showers = n_showers or N_SHOWERS_DEFAULT
    fmt = f"0{n_digits}d"
    if output_name is None:
        output_name = "1000simplebox_1-100GeV_sf-nlayers.h5"
    return {
        'root_dir':          BASE_DIR / f"output_dataset/SimpleBox/root/{ref_dir}/{sub_dir}",
        'h5_temp_dir':       BASE_DIR / f"output_dataset/SimpleBox/h5/{ref_dir}/{sub_dir}/temp",
        'xml_dir':           BASE_DIR / f"calo_configs/par04/SimpleBox/{ref_dir}/{sub_dir}",
        'metadata_json':     BASE_DIR / f"calo_configs/par04/SimpleBox/{ref_dir}/{sub_dir}/{metadata_file}",
        'showerdata_output': BASE_DIR / f"output_dataset/SimpleBox/showerdata/{ref_dir}/{sub_dir}/{output_name}",
        'root_pattern': f"{n_showers}_1-100GeV_SiW_xml_{{config_id:{fmt}}}.root",
        'xml_pattern':  f"SimpleBox_config_{{config_id:{fmt}}}.xml",
        'h5_pattern':   f"config_{{config_id:{fmt}}}.h5",
        # Processing parameters
        'cell_size': 1.0,
        'energy_threshold': 0.01,  # MeV
        'apply_clustering': True,
        'apply_threshold_after_clustering': True,
        'cone_angle': None,
    }


def process_individual_configs(CONFIG):
    """
    Process each config's ROOT file -> individual temp H5.
    Fast-paths configs whose temp file already exists.
    Returns (max_points_global, successful_config_ids).
    """
    print("\nSTEP 1: PROCESSING INDIVIDUAL CONFIGURATIONS\n")
    CONFIG['h5_temp_dir'].mkdir(parents=True, exist_ok=True)

    with open(CONFIG['metadata_json']) as f:
        metadata = json.load(f)

    max_points_global = 0
    successful_configs = []

    for config_id in tqdm(range(len(metadata)), desc="configs"):
        temp_h5 = CONFIG['h5_temp_dir'] / CONFIG['h5_pattern'].format(config_id=config_id)

        # Fast path: temp file already exists
        if temp_h5.exists():
            try:
                with h5py.File(temp_h5, 'r') as f:
                    max_points_global = max(max_points_global, f['events'].shape[1])
                successful_configs.append(config_id)
                continue
            except OSError:
                print(f"Warning: corrupt {temp_h5.name}, regenerating...")
                os.remove(temp_h5)

        # Slow path: process from ROOT
        try:
            root_file = CONFIG['root_dir'] / CONFIG['root_pattern'].format(config_id=config_id)
            xml_file  = CONFIG['xml_dir']  / CONFIG['xml_pattern'].format(config_id=config_id)
            if not root_file.exists() or not xml_file.exists():
                continue

            tree = uproot.open(root_file)["events"]
            n_entries = tree.num_entries
            # Read in chunks of 20k to avoid uproot int32 overflow on large files
            CHUNK = 20_000
            ecal_branches = [
                "ECalBarrelCollectionContributions/ECalBarrelCollectionContributions.energy",
                "ECalBarrelCollectionContributions/ECalBarrelCollectionContributions.stepPosition.x",
                "ECalBarrelCollectionContributions/ECalBarrelCollectionContributions.stepPosition.y",
                "ECalBarrelCollectionContributions/ECalBarrelCollectionContributions.stepPosition.z",
                "ECalBarrelCollectionContributions/ECalBarrelCollectionContributions.time",
            ]
            mc_branches = [
                "MCParticles/MCParticles.momentum.x",
                "MCParticles/MCParticles.momentum.y",
                "MCParticles/MCParticles.momentum.z",
                "MCParticles/MCParticles.generatorStatus",
                "MCParticles/MCParticles.PDG",
            ]
            geometry = calo_geometry.SimpleBoxGeometry.from_xml(xml_file)
            chunk_paths = []
            for start in range(0, n_entries, CHUNK):
                stop = min(start + CHUNK, n_entries)
                ecal_steps = tree.arrays(ecal_branches, library="ak", entry_start=start, entry_stop=stop)
                mc_data    = tree.arrays(mc_branches,   library="ak", entry_start=start, entry_stop=stop)
                inc_e, inc_p = prep.get_incident_momentum(mc_data, verbose=False)
                chunk_path = temp_h5.parent / f"{temp_h5.stem}_chunk{start}.h5"
                prep.process_and_save_batch(
                    arrays=ecal_steps,
                    geometry=geometry,
                    output_path=chunk_path,
                    cone_angle=CONFIG['cone_angle'],
                    cell_size=CONFIG['cell_size'],
                    incident_energies=inc_e,
                    incident_momentum=inc_p,
                    apply_clustering=CONFIG['apply_clustering'],
                    apply_threshold_after_clustering=CONFIG['apply_threshold_after_clustering'],
                    energy_threshold=CONFIG['energy_threshold'],
                )
                chunk_paths.append(chunk_path)

            # Merge chunks into single temp_h5
            if len(chunk_paths) == 1:
                chunk_paths[0].rename(temp_h5)
            else:
                import numpy as np
                chunks_data = []
                first_attrs = {}
                for cp in chunk_paths:
                    with h5py.File(cp, 'r') as f:
                        chunks_data.append({k: f[k][:] for k in f.keys()})
                        if not first_attrs:
                            first_attrs = dict(f.attrs)
                    cp.unlink()
                max_pts = max(c['events'].shape[1] for c in chunks_data)
                total_n = sum(c['events'].shape[0] for c in chunks_data)
                with h5py.File(temp_h5, 'w') as out:
                    # Copy geometry attributes from first chunk (required by combine_temp_to_showerdata)
                    for ak, av in first_attrs.items():
                        out.attrs[ak] = av
                    for k in chunks_data[0].keys():
                        if k == 'events':
                            merged = np.zeros((total_n, max_pts, chunks_data[0]['events'].shape[2]),
                                              dtype=chunks_data[0]['events'].dtype)
                            row = 0
                            for c in chunks_data:
                                n, p, d = c['events'].shape
                                merged[row:row+n, :p, :] = c['events']
                                row += n
                            out.create_dataset('events', data=merged)
                        else:
                            out.create_dataset(k, data=np.concatenate([c[k] for c in chunks_data]))

            with h5py.File(temp_h5, 'r') as f:
                max_points_global = max(max_points_global, f['events'].shape[1])
            successful_configs.append(config_id)

        except Exception:
            raise

    print(f"\nmax_points={max_points_global} | successful configs={len(successful_configs)}")
    return max_points_global, successful_configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref-dir',       type=str, default='sf_nlayers_angles',
                        help='Reference directory (default: sf_nlayers_angles)')
    parser.add_argument('--sub-dir',       type=str, default='SiW_final')
    parser.add_argument('--n-digits',      type=int, default=4)
    parser.add_argument('--n-showers',     type=int, default=None)
    parser.add_argument('--metadata-file', type=str, default='final_metadata.json')
    parser.add_argument('--output-name',   type=str, default=None)
    args = parser.parse_args()

    CONFIG = build_config(args.ref_dir, args.sub_dir, args.n_digits,
                          args.n_showers, args.metadata_file, args.output_name)
    _, configs = process_individual_configs(CONFIG)

    if configs:
        combine_temp_to_showerdata(
            h5_temp_dir=str(CONFIG['h5_temp_dir']),
            dst_h5=str(CONFIG['showerdata_output']),
        )
    else:
        print("No configs processed — nothing to combine.")


if __name__ == "__main__":
    main()
