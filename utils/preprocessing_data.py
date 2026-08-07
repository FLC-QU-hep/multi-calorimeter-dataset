# utils/preprocessing_data.py

import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import xml.etree.ElementTree as ET
import re
import h5py
from tqdm import tqdm

from . import calo_geometry as calo_geom

# ==================== DATA PREPARATION ====================

DEFAULT_CONTRIB_PREFIX = "ECalBarrelCollectionContributions/ECalBarrelCollectionContributions"

def process_calo_hits(arrays, event_index=None, apply_rotation=False,
                      cone_angle=None, phi_max=None,
                      energy_threshold_mev=1e-2, verbose=False,
                      contrib_prefix=None):
    """
    Extract and process calorimeter hit data with optional filters.
    
    Parameters:
    -----------
    arrays : dict
        Awkward array containing calorimeter hit data
    event_index : int or None
        If specified, process only this event. If None, process all events.
    apply_rotation : bool
        If True, rotate from detector (x,y,z) to global (z,-y,x) coordinates
        After rotation: z = beam direction, x-y = transverse plane
    cone_angle : float or None
        Full 3D cone opening angle around beam (z-axis after rotation) in degrees.
        Accepts all hits with theta < cone_angle/2 from z-axis.
        This is rotationally symmetric (360° in phi).
    phi_min : float or None
        Minimum azimuthal angle in x-y transverse plane in degrees (default: None)
        phi = 0° is along +x, phi = 90° is along +y
    phi_max : float or None
        Maximum azimuthal angle in x-y transverse plane in degrees (default: None)
    energy_threshold_mev : float
        Minimum hit energy in MeV (default: 0.01 MeV)
    verbose : bool
        If True, print filtering statistics and geometry info
    
    Returns:
    --------
    dict with processed hit data
    
    Coordinate system after rotation (if apply_rotation=True):
    - z: beam direction (was x in detector frame)
    - x: transverse (was z in detector frame)  
    - y: transverse (was -y in detector frame)
    - theta: polar angle from z-axis (beam)
    - phi: azimuthal angle in x-y plane (transverse)
    """
    # Extract data
    p = contrib_prefix if contrib_prefix else DEFAULT_CONTRIB_PREFIX
    energy = arrays[f"{p}.energy"]
    x = arrays[f"{p}.stepPosition.x"]
    y = arrays[f"{p}.stepPosition.y"]
    z = arrays[f"{p}.stepPosition.z"]
    t = arrays[f"{p}.time"]
    
    # Select event if specified
    if event_index is not None:
        energy = energy[event_index:event_index+1]
        x = x[event_index:event_index+1]
        y = y[event_index:event_index+1]
        z = z[event_index:event_index+1]
        t = t[event_index:event_index+1]
    
    # Rotate to global coordinates if requested
    if apply_rotation:
        x, y, z = z, -y, x
    
    # Track original hit count
    n_original = int(ak.sum(ak.num(energy)))
    
    # Build combined mask
    mask = (energy * 1e3) >= energy_threshold_mev
    n_after_energy = int(ak.sum(ak.num(energy[mask])))
    
    # Apply 3D cone cut (theta from z-axis = beam) if specified
    if cone_angle is not None:
        r_transverse = np.sqrt(x**2 + y**2)  # distance from z-axis (beam)
        theta = np.arctan2(r_transverse, z)  # polar angle from z-axis
        cone_mask = theta < np.radians(cone_angle / 2)
        mask = mask & cone_mask
        n_after_cone = int(ak.sum(ak.num(energy[mask])))
    else:
        n_after_cone = n_after_energy
    
    # Apply filters
    energy = energy[mask]
    x, y, z, t = x[mask], y[mask], z[mask], t[mask]
    
    # Compute totals
    total_energy = ak.sum(energy, axis=1).to_numpy()
    nhits = ak.num(energy).to_numpy()
    n_final = int(ak.sum(nhits))
    
    # Print statistics if verbose
    if verbose:
        print(f"\n{'='*60}")
        print(f"PROCESSING SUMMARY")
        print(f"{'='*60}")
        if event_index is not None:
            print(f"Event:                   {event_index}")
        print(f"Rotation applied:        {apply_rotation}")
        if apply_rotation:
            print(f"  (z=beam, x-y=transverse)")
        print(f"\nFiltering chain:")
        print(f"  Original hits:         {n_original:,}")
        print(f"  After E ≥ {energy_threshold_mev} MeV:  {n_after_energy:,} ({100*n_after_energy/n_original:.1f}%)")
        
        if cone_angle is not None:
            print(f"  After 3D cone {cone_angle}°:  {n_after_cone:,} ({100*n_after_cone/n_original:.1f}%)")
            print(f"    (±{cone_angle/2:.1f}° from beam/z-axis)")
        print(f"  Final hits:            {n_final:,} ({100*n_final/n_original:.1f}%)")
        print(f"\nTotal energy:            {total_energy.sum():.3f} GeV")
        print(f"{'='*60}\n")
    
    return {
        'energy': ak.flatten(energy).to_numpy(),
        'x': ak.flatten(x).to_numpy(),
        'y': ak.flatten(y).to_numpy(),
        'z': ak.flatten(z).to_numpy(),
        'time': ak.flatten(t).to_numpy(),
        'total_energy': total_energy,
        'nhits': nhits,
        'n_hits_original': n_original,
        'n_hits_final': n_final,
        'cone_angle': cone_angle,
    }


def get_incident_energies(data, verbose=True):
    """
    Compute incident energies per event from raw calorimeter data.
    
    Parameters:
    -----------
    data : dict
        Awkward array containing calorimeter hit data
    
    Returns:
    --------
    np.ndarray of incident energies per event in GeV
    """
    px = data["MCParticles/MCParticles.momentum.x"]
    py = data["MCParticles/MCParticles.momentum.y"]
    pz = data["MCParticles/MCParticles.momentum.z"]
    gen_status = data["MCParticles/MCParticles.generatorStatus"]
    pdg = data["MCParticles/MCParticles.PDG"]

    # Filter: keep only primary particles (generatorStatus == 1)
    mask = gen_status == 1

    px_primary = px[mask]
    py_primary = py[mask]
    pz_primary = pz[mask]

    # Calculate momentum magnitude for primary particles only
    p_mag_primary = np.sqrt(px_primary**2 + py_primary**2 + pz_primary**2)

    print("Number of primary particles per event:", ak.num(p_mag_primary))

    # For photons: E = |p| (assuming momentum is in GeV/c)
    incident_energy = ak.flatten(p_mag_primary)  # Should be 1 particle per event

    incident_energy_np = np.array(incident_energy)

    if verbose:
        print(f"\n{'='*60}")
        print(f"INCIDENT ENERGY SUMMARY")
        print(f"{'='*60}")
        print(f"Shape: {incident_energy_np.shape}")
        print(f"Min energy: {incident_energy_np.min():.3f} GeV")
        print(f"Max energy: {incident_energy_np.max():.3f} GeV")
        print(f"Mean energy: {incident_energy_np.mean():.3f} GeV")
        print(f"PDG: {pdg[mask]}")
        print(f"{'='*60}\n")
    return incident_energy_np


def get_incident_momentum(data, verbose=True):
    """
    Extract incident energy and momentum 3-vector per event from MCParticles.

    Parameters:
    -----------
    data : dict
        Awkward array containing MCParticle data
    verbose : bool
        If True, print summary statistics

    Returns:
    --------
    tuple of (incident_energy, incident_momentum)
        incident_energy: np.ndarray of shape (n_events,) in GeV
        incident_momentum: np.ndarray of shape (n_events, 3) -- (px, py, pz) in GeV/c
    """
    px = data["MCParticles/MCParticles.momentum.x"]
    py = data["MCParticles/MCParticles.momentum.y"]
    pz = data["MCParticles/MCParticles.momentum.z"]
    gen_status = data["MCParticles/MCParticles.generatorStatus"]

    # Filter: keep only primary particles (generatorStatus == 1)
    mask = gen_status == 1

    px_flat = np.array(ak.flatten(px[mask]))
    py_flat = np.array(ak.flatten(py[mask]))
    pz_flat = np.array(ak.flatten(pz[mask]))

    p_mag = np.sqrt(px_flat**2 + py_flat**2 + pz_flat**2)
    momentum = np.stack([px_flat, py_flat, pz_flat], axis=1)  # (n_events, 3)

    if verbose:
        p_hat = momentum / p_mag[:, None]
        theta = np.arccos(np.clip(pz_flat / p_mag, -1, 1))
        print(f"\n{'='*60}")
        print(f"INCIDENT MOMENTUM SUMMARY")
        print(f"{'='*60}")
        print(f"Energy |p|: [{p_mag.min():.3f}, {p_mag.max():.3f}] GeV, mean={p_mag.mean():.3f}")
        print(f"Theta: [{theta.min():.4f}, {theta.max():.4f}] rad")
        print(f"Momentum shape: {momentum.shape}")
        print(f"{'='*60}\n")

    return p_mag, momentum


# ==================== CLUSTERING ====================

def cluster_to_grid(data, cell_size_mm=1.0, geometry=None, verbose=True):
    """
    Cluster hits into regular x-y grid within each layer.
    Position is taken from the highest-energy hit in each cell.
    Supports both cylindrical (CaloGeometry) and planar (SimpleBoxGeometry).
    """
    from utils.calo_geometry import SimpleBoxGeometry

    x = data['x']
    y = data['y']
    z = data['z']
    layer = data['layer']
    energy = data['energy']
    time = data['time']

    # Detect geometry type
    is_simplebox = geometry is not None and isinstance(geometry, SimpleBoxGeometry)
    
    # Handle empty input
    if len(energy) == 0:
        print(f"\n{'='*60}")
        print(f"WARNING: No hits to cluster (empty input)")
        print(f"{'='*60}\n")
        result = {
            'energy': np.array([], dtype=np.float32),
            'x': np.array([], dtype=np.float32),
            'y': np.array([], dtype=np.float32),
            'z': np.array([], dtype=np.float32),
            'layer': np.array([], dtype=np.int32),
            'time': np.array([], dtype=np.float32),
            'total_energy': data.get('total_energy', np.array([0.0])),
            'nhits': data.get('nhits', np.array([0])),
            'cell_size': cell_size_mm,
            'is_clustered': True
        }
        # Add coordinate based on geometry type
        if not is_simplebox:
            result['r'] = np.array([], dtype=np.float32)
        return result
    
    # Lists to accumulate clustered hits
    x_clustered = []
    y_clustered = []
    z_clustered = []
    layer_clustered = []
    energy_clustered = []
    time_clustered = []
    
    # Process each layer separately
    unique_layers = np.unique(layer)
    
    for lay in unique_layers:
        # Select hits in this layer
        mask = layer == lay
        x_layer = x[mask]
        y_layer = y[mask]
        z_layer = z[mask]
        e_layer = energy[mask]
        t_layer = time[mask]
        
        if len(x_layer) == 0:
            continue
        
        # Convert to grid coordinates
        x_grid = np.floor(x_layer / cell_size_mm).astype(np.int32)
        y_grid = np.floor(y_layer / cell_size_mm).astype(np.int32)
        
        # Create unique cell identifier
        grid_idx = np.stack([x_grid, y_grid], axis=1)
        
        # Find unique cells
        unique_cells, inverse_idx = np.unique(grid_idx, axis=0, return_inverse=True)
        
        # For each cell, find highest energy hit and sum energies
        x_cell_pos = np.zeros(len(unique_cells), dtype=np.float32)
        y_cell_pos = np.zeros(len(unique_cells), dtype=np.float32)
        z_cell_pos = np.zeros(len(unique_cells), dtype=np.float32)
        t_cell_pos = np.zeros(len(unique_cells), dtype=np.float32)
        e_summed = np.zeros(len(unique_cells), dtype=np.float32)
        
        # Process each unique cell
        for cell_idx in range(len(unique_cells)):
            hits_in_cell_mask = (inverse_idx == cell_idx)
            energies_in_cell = e_layer[hits_in_cell_mask]
            max_energy_idx_local = np.argmax(energies_in_cell)
            indices_in_layer = np.where(hits_in_cell_mask)[0]
            max_energy_hit_global_idx = indices_in_layer[max_energy_idx_local]
            
            x_cell_pos[cell_idx] = x_layer[max_energy_hit_global_idx]
            y_cell_pos[cell_idx] = y_layer[max_energy_hit_global_idx]
            z_cell_pos[cell_idx] = z_layer[max_energy_hit_global_idx]
            t_cell_pos[cell_idx] = t_layer[max_energy_hit_global_idx]
            e_summed[cell_idx] = energies_in_cell.sum()
        
        # Accumulate results
        x_clustered.append(x_cell_pos)
        y_clustered.append(y_cell_pos)
        z_clustered.append(z_cell_pos)
        layer_clustered.append(np.full(len(unique_cells), lay, dtype=np.int32))
        energy_clustered.append(e_summed)
        time_clustered.append(t_cell_pos)
    
    # Handle case where no layers had hits after processing
    if len(x_clustered) == 0:
        print(f"\n{'='*60}")
        print(f"WARNING: No cells after clustering")
        print(f"{'='*60}\n")
        result = {
            'energy': np.array([], dtype=np.float32),
            'x': np.array([], dtype=np.float32),
            'y': np.array([], dtype=np.float32),
            'z': np.array([], dtype=np.float32),
            'layer': np.array([], dtype=np.int32),
            'time': np.array([], dtype=np.float32),
            'total_energy': data.get('total_energy', np.array([0.0])),
            'nhits': data.get('nhits', np.array([0])),
            'cell_size': cell_size_mm,
            'is_clustered': True
        }
        # Add coordinate based on geometry type
        if not is_simplebox:
            result['r'] = np.array([], dtype=np.float32)
        return result
    
    # Concatenate all layers
    x_clustered = np.concatenate(x_clustered)
    y_clustered = np.concatenate(y_clustered)
    z_clustered = np.concatenate(z_clustered)
    layer_clustered = np.concatenate(layer_clustered)
    energy_clustered = np.concatenate(energy_clustered)
    time_clustered = np.concatenate(time_clustered)

    # Compute r only for cylindrical geometry
    if not is_simplebox:
        r_clustered = np.sqrt(y_clustered**2 + z_clustered**2)
    
    # Calculate energy statistics
    energy_before = energy.sum()
    energy_after = energy_clustered.sum()
    abs_diff = abs(energy_before - energy_after)
    rel_diff = abs_diff / energy_before if energy_before > 0 else 0

    # Get per-shower statistics if available
    nhits_original = data.get('nhits', None)
    max_hits_before = None
    min_hits_before = None
    avg_hits_after = None
    num_showers = None

    if nhits_original is not None:
        nhits_array = np.asarray(nhits_original)
        if nhits_array.size > 0:
            num_showers = len(nhits_array)
            max_hits_before = int(np.max(nhits_array))
            min_hits_before = int(np.min(nhits_array))
            avg_hits_after = len(energy_clustered) / num_showers

    if verbose:
        print(f"\n{'='*60}")
        print(f"CLUSTERING SUMMARY")
        print(f"{'='*60}")
        print(f"Cell size:               {cell_size_mm} mm × {cell_size_mm} mm")
        print(f"Original hits (total):   {len(energy):,}")
        print(f"Clustered cells (total): {len(energy_clustered):,}")
        print(f"Compression ratio:       {len(energy)/len(energy_clustered):.1f}x")

        if max_hits_before is not None and num_showers is not None:
            if num_showers == 1:
                # Single shower
                print(f"Hits per shower:         {max_hits_before:,} (before) → {len(energy_clustered):,} (after)")
            else:
                # Multiple showers
                print(f"Hits per shower (before): max = {max_hits_before:,}, range = {min_hits_before:,} - {max_hits_before:,}")
                print(f"Hits per shower (after):  avg = {avg_hits_after:,.1f}")

        print(f"Energy conserved:        {rel_diff < 1e-6} ({100*rel_diff:.4f}%)")
        print(f"{'='*60}\n")
    
    result = {
        'energy': energy_clustered,
        'x': x_clustered,
        'y': y_clustered,
        'z': z_clustered,
        'layer': layer_clustered,
        'time': time_clustered,
        'total_energy': data.get('total_energy', np.array([energy_clustered.sum()])),
        'nhits': data.get('nhits', np.array([len(energy_clustered)])),
        'cell_size': cell_size_mm,
        'is_clustered': True
    }

    # Add 'r' coordinate only for cylindrical geometry
    if not is_simplebox:
        result['r'] = r_clustered

    return result

# ==================== POINT CLOUD CONVERSION ====================
def convert_to_pointcloud(clustered_data_list, z_as_layer=True):
    """
    Convert list of clustered shower data to point cloud format.
    Returns ONLY the events array - no extra computations.
    """
    n_showers = len(clustered_data_list)

    # Determine max_points
    max_points = max(len(data['energy']) for data in clustered_data_list)
    print(f"Auto-detected max_points: {max_points}")

    # Initialize array
    events = np.zeros((n_showers, max_points, 5), dtype=np.float32)

    # Fill arrays
    for i, data in enumerate(clustered_data_list):
        x = data['x']
        y = data['y']
        energy = data['energy']
        t = data.get('time', np.zeros_like(energy))

        # Choose z coordinate
        if z_as_layer:
            z = data['layer'].astype(np.float32)
        else:
            z = data['z']

        n_points = min(len(energy), max_points)

        # Fill point cloud: [x, y, z, time, E]
        events[i, :n_points, 0] = x[:n_points]
        events[i, :n_points, 1] = y[:n_points]
        events[i, :n_points, 2] = z[:n_points]
        events[i, :n_points, 3] = t[:n_points]
        events[i, :n_points, 4] = energy[:n_points]

    print(f"\n{'='*60}")
    print(f"POINT CLOUD CONVERSION")
    print(f"{'='*60}")
    print(f"Number of showers:       {n_showers}")
    print(f"Max points per shower:   {max_points}")
    print(f"Point cloud shape:       {events.shape}")
    print(f"Features:                [x, y, z, time, energy]")
    print(f"{'='*60}\n")

    return {'events': events}

def save_pointcloud_h5(pointcloud_dict, output_path, incident_energy=None,
                       incident_momentum=None, sampling_fraction=None,
                       metadata=None):
    """
    Save point cloud data to HDF5 file.
    Streamlined version - only saves essential data.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Main datasets
        f.create_dataset('events', data=pointcloud_dict['events'],
                         compression='gzip', compression_opts=5)

        # Incident energy (n, 1)
        if incident_energy is not None:
            incident_energy = np.array(incident_energy).reshape(-1, 1)
            f.create_dataset('incident_energy', data=incident_energy)

        # Incident momentum 3-vector (n, 3)
        if incident_momentum is not None:
            incident_momentum = np.array(incident_momentum).reshape(-1, 3)
            f.create_dataset('incident_momentum', data=incident_momentum)

        # Sampling fraction (n, 1)
        if sampling_fraction is not None:
            sampling_fraction = np.array(sampling_fraction).reshape(-1, 1)
            f.create_dataset('sampling_fraction', data=sampling_fraction)

        # Metadata as attributes
        n_showers = pointcloud_dict['events'].shape[0]
        max_points = pointcloud_dict['events'].shape[1]
        n_features = pointcloud_dict['events'].shape[2]
        
        f.attrs['n_showers'] = n_showers
        f.attrs['max_points'] = max_points
        f.attrs['n_features'] = n_features
        f.attrs['feature_names'] = ['x', 'y', 'z', 'time', 'energy']

        if metadata is not None:
            for key, value in metadata.items():
                f.attrs[key] = value

    print(f"Saved point cloud to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1e6:.2f} MB")

def load_pointcloud_h5(input_path):
    """
    Load point cloud data from HDF5 file.
    """
    input_path = Path(input_path)
    
    with h5py.File(input_path, 'r') as f:
        data = {
            'events': f['events'][:],
        }
        
        # Optional datasets
        if 'incident_energy' in f:
            data['incident_energy'] = f['incident_energy'][:]
        if 'sampling_fraction' in f:  
            data['sampling_fraction'] = f['sampling_fraction'][:]
        
        # Metadata
        data['metadata'] = dict(f.attrs)
    
    print(f"\n{'='*60}")
    print(f"LOADED POINT CLOUD")
    print(f"{'='*60}")
    print(f"File: {input_path.name}")
    print(f"Number of showers:       {data['metadata']['n_showers']}")
    print(f"Max points per shower:   {data['metadata']['max_points']}")
    print(f"Shape:                   {data['events'].shape}")
    print(f"Features:                {data['metadata']['feature_names']}")
    if 'sampling_fraction' in data:  
        print(f"Sampling fraction:       {data['sampling_fraction'][0, 0]:.4f}")
    print(f"{'='*60}\n")
    
    return data

def process_and_save_batch(arrays, geometry, output_path,
                           cone_angle=None, cell_size=1.0, incident_energies=None,
                           incident_momentum=None,
                           event_indices=None, energy_threshold=None,
                           apply_clustering=True,
                           apply_threshold_after_clustering=False,
                           contrib_prefix=None):
    """
    Process multiple showers and save as point cloud HDF5.
    Supports both cylindrical (CaloGeometry) and planar (SimpleBoxGeometry).

    Parameters:
    -----------
    apply_clustering : bool
        If True, cluster hits into grid cells (default: True)
    apply_threshold_after_clustering : bool
        If True, apply energy threshold AFTER clustering (recommended)
        If False, apply energy threshold BEFORE clustering (old behavior)
    energy_threshold : float or None
        Energy threshold in MeV
    """
    from utils.calo_geometry import SimpleBoxGeometry

    # Detect geometry type
    is_simplebox = isinstance(geometry, SimpleBoxGeometry)

    # Determine which events to process
    if event_indices is None:
        _p = contrib_prefix if contrib_prefix else DEFAULT_CONTRIB_PREFIX
        n_events = len(arrays[f"{_p}.energy"])
        event_indices = list(range(n_events))
    else:
        event_indices = list(event_indices)

    print(f"Processing {len(event_indices)} showers...")
    print(f"Geometry type: {'SimpleBox (planar)' if is_simplebox else 'Cylindrical'}")
    print(f"Clustering: {'ENABLED' if apply_clustering else 'DISABLED'}")
    print(f"Energy threshold: {energy_threshold} MeV {'(after clustering)' if apply_threshold_after_clustering else '(before clustering)'}")
    
    # Get incident energies and momentum
    incident_energies = np.array(incident_energies)[event_indices]
    if incident_momentum is not None:
        incident_momentum = np.array(incident_momentum)[event_indices]
    samp_fract = geometry.sampling_fraction
    
    # Process each shower
    processed_data_list = []
    valid_indices = []
    hits_before_clustering = []
    hits_after_clustering = []
    hits_after_threshold = []  # NEW: track hits after threshold
    
    for i, evt_idx in enumerate(tqdm(event_indices, desc="Processing showers")):
        # Decide threshold for process_calo_hits
        if apply_threshold_after_clustering:
            # Use very low threshold initially (keep almost all hits)
            initial_threshold = 1e-6  # 0.001 keV - essentially no cut
        else:
            # Use specified threshold
            initial_threshold = energy_threshold if energy_threshold is not None else 1e-6
        
        # Process shower with rotation
        data = process_calo_hits(
            arrays,
            cone_angle=cone_angle,
            event_index=evt_idx,
            apply_rotation=False,
            energy_threshold_mev=initial_threshold,
            verbose=False,
            contrib_prefix=contrib_prefix
        )
        
        data = calo_geom.add_layer_info(data, geometry, verbose=False, 
                                        remove_overflow=False)  
        
        # Skip if no hits after filtering
        if len(data['energy']) == 0:
            print(f"Warning: Event {evt_idx} has no hits after filtering, skipping...")
            continue
        
        n_hits_before = len(data['energy'])
        hits_before_clustering.append(n_hits_before)
        
        if apply_clustering:
            # Cluster hits
            processed = cluster_to_grid(data, cell_size_mm=cell_size, 
                                       geometry=geometry, verbose=False)
            
            if len(processed['energy']) == 0:
                print(f"Warning: Event {evt_idx} has no hits after clustering, skipping...")
                continue
            
            hits_after_clustering.append(len(processed['energy']))
            
            # NEW: Apply energy threshold AFTER clustering
            if apply_threshold_after_clustering and energy_threshold is not None:
                # Filter cells by energy
                energy_mask = (processed['energy'] * 1e3) >= energy_threshold  # Convert GeV to MeV

                # Apply mask to all arrays
                processed['energy'] = processed['energy'][energy_mask]
                processed['x'] = processed['x'][energy_mask]
                processed['y'] = processed['y'][energy_mask]
                processed['z'] = processed['z'][energy_mask]
                if not is_simplebox:
                    processed['r'] = processed['r'][energy_mask]
                processed['layer'] = processed['layer'][energy_mask]
                processed['time'] = processed['time'][energy_mask]
                
                # Check if any hits remain
                if len(processed['energy']) == 0:
                    print(f"Warning: Event {evt_idx} has no hits after threshold, skipping...")
                    continue
                
                hits_after_threshold.append(len(processed['energy']))
            else:
                hits_after_threshold.append(len(processed['energy']))
        else:
            # Use raw hits
            processed = data.copy()
            processed['layer'] = processed['layer'].astype(np.int32)
            
            # Apply threshold if requested (even for raw hits)
            if apply_threshold_after_clustering and energy_threshold is not None:
                energy_mask = (processed['energy'] * 1e3) >= energy_threshold

                processed['energy'] = processed['energy'][energy_mask]
                processed['x'] = processed['x'][energy_mask]
                processed['y'] = processed['y'][energy_mask]
                processed['z'] = processed['z'][energy_mask]
                if not is_simplebox:
                    processed['r'] = processed['r'][energy_mask]
                processed['layer'] = processed['layer'][energy_mask]
                processed['time'] = processed['time'][energy_mask]
                
                if len(processed['energy']) == 0:
                    print(f"Warning: Event {evt_idx} has no hits after threshold, skipping...")
                    continue
            
            hits_after_clustering.append(len(processed['energy']))
            hits_after_threshold.append(len(processed['energy']))
        
        processed_data_list.append(processed)
        valid_indices.append(i)
    
    if len(processed_data_list) == 0:
        raise ValueError("No valid events after processing!")
    
    print(f"\nSuccessfully processed {len(processed_data_list)}/{len(event_indices)} events")
    
    # Print statistics
    hits_before_clustering = np.array(hits_before_clustering)
    hits_after_clustering = np.array(hits_after_clustering)
    hits_after_threshold = np.array(hits_after_threshold)
    
    print(f"\n{'='*60}")
    print(f"PROCESSING STATISTICS")
    print(f"{'='*60}")
    print(f"Hits BEFORE clustering:")
    print(f"  Max:     {hits_before_clustering.max():,}")
    print(f"  Mean:    {hits_before_clustering.mean():.1f}")
    print(f"  Min:     {hits_before_clustering.min():,}")
    
    if apply_clustering:
        print(f"\nHits AFTER clustering:")
        print(f"  Max:     {hits_after_clustering.max():,}")
        print(f"  Mean:    {hits_after_clustering.mean():.1f}")
        print(f"  Min:     {hits_after_clustering.min():,}")
        print(f"\nCompression ratio:  {hits_before_clustering.mean() / hits_after_clustering.mean():.2f}x")
    
    if apply_threshold_after_clustering and energy_threshold is not None:
        print(f"\nHits AFTER threshold ({energy_threshold} MeV):")
        print(f"  Max:     {hits_after_threshold.max():,}")
        print(f"  Mean:    {hits_after_threshold.mean():.1f}")
        print(f"  Min:     {hits_after_threshold.min():,}")
        print(f"\nThreshold rejection: {100*(1 - hits_after_threshold.mean()/hits_after_clustering.mean()):.1f}%")
    
    print(f"{'='*60}\n")
    
    # Filter arrays to match valid events
    incident_energies = incident_energies[valid_indices]
    if incident_momentum is not None:
        incident_momentum = incident_momentum[valid_indices]
    samp_fract_array = np.full((len(valid_indices), 1), samp_fract)
    
    # Convert to point cloud
    pointcloud = convert_to_pointcloud(processed_data_list)
    
    # Save to HDF5
    metadata = {
        'cone_angle': cone_angle,
        'cell_size_mm': cell_size if apply_clustering else None,
        'energy_threshold_mev': energy_threshold if energy_threshold is not None else 0.0,
        'threshold_after_clustering': apply_threshold_after_clustering,  # NEW
        'clustered': apply_clustering,
        'geometry_type': 'SimpleBox' if is_simplebox else 'Cylindrical',
        'geometry_num_layers': geometry.num_layers,
        'geometry_layer_thickness': float(geometry.layer_thickness),
        'geometry_sampling_fraction': float(samp_fract) if samp_fract is not None else None,
        'n_events_processed': len(event_indices),
        'n_events_valid': len(processed_data_list)
    }

    # Add geometry-specific parameters
    if is_simplebox:
        metadata['geometry_z_start'] = float(geometry.z_start)
        metadata['geometry_z_end'] = float(geometry.z_end)
        metadata['geometry_box_x'] = float(geometry.box_x)
        metadata['geometry_box_y'] = float(geometry.box_y)
    else:
        metadata['geometry_rmin'] = geometry.r_min
        metadata['geometry_rmax'] = geometry.r_max
    
    # Convert metadata to HDF5-compatible types
    metadata_clean = {}
    for key, value in metadata.items():
        if value is None:
            metadata_clean[key] = 'None'
        elif isinstance(value, (bool, np.bool_)):
            metadata_clean[key] = int(value)
        elif isinstance(value, (np.integer, np.floating)):
            metadata_clean[key] = float(value)
        else:
            metadata_clean[key] = value
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to: {output_path}")
    print(f"Directory exists: {output_path.parent.exists()}")

    save_pointcloud_h5(
        pointcloud,
        output_path,
        incident_energy=incident_energies,
        incident_momentum=incident_momentum,
        sampling_fraction=samp_fract_array,
        metadata=metadata_clean
    )

    # Verify file was created
    if output_path.exists():
        file_size_mb = output_path.stat().st_size / 1e6
        print(f"✓ File saved successfully!")
        print(f"  Size: {file_size_mb:.2f} MB")
        print(f"  Path: {output_path}")
    else:
        print(f"✗ ERROR: File was not created at {output_path}")

    return pointcloud

# ==================== ANALYSIS ====================

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

