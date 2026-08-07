import numpy as np
import xml.etree.ElementTree as ET
import re
from pathlib import Path



# ==================== GEOMETRY & LAYER CONVERSION ====================

class SimpleBoxGeometry:
    """Simple box calorimeter geometry parameters (planar/Cartesian)."""

    def __init__(self, z_start=0.0, active_thickness=1.10,
                 passive_thickness=4.62, num_layers=30, num_simulated_layers=31,
                 box_x=100.0, box_y=100.0, sampling_fraction=None):
        self.z_start = z_start  # Starting z position of the box
        self.active_thickness = active_thickness
        self.passive_thickness = passive_thickness
        self.layer_thickness = active_thickness + passive_thickness
        self.num_layers = num_layers
        self.sampling_fraction = sampling_fraction
        self.num_simulated_layers = num_simulated_layers
        self.box_x = box_x  # Box width in x
        self.box_y = box_y  # Box width in y
        self.box_z = num_layers * self.layer_thickness  # Total depth
        self.z_end = z_start + self.box_z
        self.z_end_simulated = z_start + num_simulated_layers * self.layer_thickness

    @classmethod
    def from_xml(cls, xml_file):
        """
        Create SimpleBoxGeometry from DD4hep XML file.

        Parameters:
        -----------
        xml_file : str or Path
            Path to DD4hep XML geometry file

        Returns:
        --------
        SimpleBoxGeometry instance with parameters from XML (including num_layers)
        """
        params = cls._parse_xml(xml_file)
        num_layers = params['num_layers']

        print(f"\n{'='*60}")
        print(f"LOADED SIMPLEBOX GEOMETRY FROM XML")
        print(f"{'='*60}")
        print(f"File: {Path(xml_file).name}")
        print(f"Box dimensions (x,y):  {params['box_x']:.2f} x {params['box_y']:.2f} mm")
        print(f"Active thickness:      {params['active_thickness']:.4f} mm")
        print(f"Passive thickness:     {params['passive_thickness']:.4f} mm")
        print(f"Sampling fraction:     {params['sampling_fraction'] if params['sampling_fraction'] is not None else 'N/A'}")
        print(f"Layer thickness:       {params['active_thickness'] + params['passive_thickness']:.4f} mm")
        print(f"Number of layers:      {num_layers}")
        print(f"Total depth:           {(params['active_thickness'] + params['passive_thickness']) * num_layers:.2f} mm")
        print(f"{'='*60}\n")

        return cls(
            z_start=0.0,  # Box starts at z=0
            active_thickness=params['active_thickness'],
            passive_thickness=params['passive_thickness'],
            num_layers=num_layers,
            num_simulated_layers=num_layers,  # Same as num_layers for SimpleBox
            box_x=params['box_x'],
            box_y=params['box_y'],
            sampling_fraction=params['sampling_fraction']
        )

    @staticmethod
    def _parse_xml(xml_file):
        """Parse DD4hep XML file and extract SimpleBox geometry parameters."""
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Unit conversion to mm
        units = {
            'mm': 1.0,
            'cm': 10.0,
            'm': 1000.0,
            'km': 1e6
        }

        def parse_value(value_str):
            """Parse value string like '23.34*mm' or '80*cm' to float in mm."""
            value_str = value_str.strip()
            # Match: number*unit or just number
            match = re.match(r'([0-9.]+)\*?([a-zA-Z]*)', value_str)
            if match:
                number = float(match.group(1))
                unit = match.group(2).lower() if match.group(2) else 'mm'
                return number * units.get(unit, 1.0)
            return float(value_str)

        # Extract constants
        constants = {}
        for constant in root.findall(".//constant"):
            name = constant.get('name')
            value = constant.get('value')

            if 'Box' in name:
                # Parse simple values (skip expressions that reference other Box constants)
                if 'Box_' not in value or name == value:
                    try:
                        constants[name] = parse_value(value)
                    except:
                        pass

            if 'sampling_fraction' in name:
                try:
                    constants['sampling_fraction'] = float(value)
                except:
                    pass

        # Validate required parameters
        required = ['Box_activeThickness', 'Box_passiveThickness', 'Box_x', 'Box_y', 'Box_numLayers']
        for param in required:
            if param not in constants:
                raise ValueError(f"{param} not found in XML")

        return {
            'active_thickness': constants['Box_activeThickness'],
            'passive_thickness': constants['Box_passiveThickness'],
            'box_x': constants['Box_x'],
            'box_y': constants['Box_y'],
            'num_layers': int(constants['Box_numLayers']),
            'sampling_fraction': constants.get('sampling_fraction')
        }

    def z_to_layer(self, z_mm):
        """
        Convert z-coordinate to layer number (0 to num_layers-1).
        Layers are stacked along z-axis starting at z_start=0.
        """
        layer = np.floor((z_mm - self.z_start) / self.layer_thickness).astype(int)
        # Clip to valid range [0, num_layers-1]
        layer = np.clip(layer, 0, self.num_layers - 1)
        return layer

    def __repr__(self):
        return (f"SimpleBoxGeometry(z_start={self.z_start:.2f}mm, "
                f"active={self.active_thickness:.2f}mm, "
                f"passive={self.passive_thickness:.2f}mm, "
                f"layers={self.num_layers}, "
                f"box_size=({self.box_x:.0f}x{self.box_y:.0f})mm, "
                f"sampling_fraction={self.sampling_fraction})"
                )


class ALLEGROGeometry:
    """ALLEGRO ECal barrel geometry (cylindrical, accordion LAr/Pb).

    Fixed detector: no configurable SF or nlayers.
    Layers are radial shells in the barrel.

    Coordinate system:
        Global (x, y, z) where z = beam axis.
        rho = sqrt(x^2 + y^2) = radial distance from beam.
        Layer index determined by rho.

    Supported versions:
        v02: 12 equal radial layers (EMBarrel R_min=2172.8, R_max=2578.3 mm)
        v03: 11 equal radial layers (same envelope)
    """

    # Both v02 and v03 share the same detector envelope:
    #   EMBarrel_rmin = BarECal_rmin(2100) + Air(49) + CryoFront(13.8) + LArBathFront(10) = 2172.8 mm
    #   EMBarrel_rmax = BarECal_rmax(2770) - Air(49) - CryoBack(102.7) - LArBathBack(40) = 2578.3 mm
    # Difference: v02 has 12 layers, v03 has 11 layers (equal radial spacing in both cases).

    RMIN = 2172.8   # mm — same for v02 and v03
    RMAX = 2578.3   # mm — same for v02 and v03

    VERSIONS = {
        'v02': 12,
        'v03': 11,
    }

    def __init__(self, r_min=None, r_max=None, layer_thicknesses=None, version='v03'):
        self.version = version
        self.r_min = r_min if r_min is not None else self.RMIN
        self.r_max = r_max if r_max is not None else self.RMAX
        if layer_thicknesses is not None:
            self.layer_thicknesses = np.array(layer_thicknesses, dtype=np.float64)
        else:
            num = self.VERSIONS.get(version, 11)
            t = (self.r_max - self.r_min) / num
            self.layer_thicknesses = np.full(num, t, dtype=np.float64)
        self.num_layers = len(self.layer_thicknesses)

        # Build radial layer boundaries: [r_min, r_min+t0, r_min+t0+t1, ...]
        self.layer_boundaries = np.zeros(self.num_layers + 1)
        self.layer_boundaries[0] = self.r_min
        for i, t in enumerate(self.layer_thicknesses):
            self.layer_boundaries[i + 1] = self.layer_boundaries[i] + t

        # Effective layer thickness (average, for compatibility with SimpleBox pipeline)
        self.layer_thickness = float(np.mean(self.layer_thicknesses))

        # Fixed geometry — no sampling fraction or variable nlayers
        self.sampling_fraction = None

        print(f"\n{'='*60}")
        print(f"ALLEGRO ECAL BARREL GEOMETRY  ({self.version})")
        print(f"{'='*60}")
        print(f"R_min:            {self.r_min:.1f} mm")
        print(f"R_max:            {self.r_max:.1f} mm")
        print(f"Number of layers: {self.num_layers}")
        print(f"Layer boundaries: {[f'{b:.1f}' for b in self.layer_boundaries]}")
        print(f"Total depth:      {self.layer_boundaries[-1] - self.layer_boundaries[0]:.1f} mm")
        print(f"{'='*60}\n")

    def rho_to_layer(self, rho):
        """Map radial distance rho to layer index (0 to num_layers-1)."""
        layer = np.searchsorted(self.layer_boundaries, rho, side='right') - 1
        return np.clip(layer, 0, self.num_layers - 1)

    def __repr__(self):
        return (f"ALLEGROGeometry(r_min={self.r_min:.1f}mm, "
                f"r_max={self.r_max:.1f}mm, "
                f"layers={self.num_layers})")


class LEMURSBarrelGeometry:
    """Barrel calorimeter geometry for any LEMURS detector.

    All detectors use equal-width radial layers.
    rho = sqrt(x^2 + y^2) determines the layer index.

    Usage:
        geom = LEMURSBarrelGeometry("par04_siw")
        geom = LEMURSBarrelGeometry("fccee_cld")
        layers = geom.rho_to_layer(rho_array)
    """

    # layer_thickness = full layer (all slices), from XML
    DETECTORS = {
        "par04_siw": {
            "r_min": 800.0,
            "layer_thickness": 1.7,    # Si(0.3) + W(1.4) = 1.7 mm
            "num_layers": 90,
            "collection": "ECalBarrelCollection",
            "nominal_sf": 0.0263,      # mean E_dep/E_inc for this geometry
        },
        "par04_scipb": {
            "r_min": 800.0,
            "layer_thickness": 5.6,    # Polystyrene(1.2) + Pb(4.4) = 5.6 mm
            "num_layers": 45,
            "collection": "ECalBarrelCollection",
            "nominal_sf": 0.0330,      # mean E_dep/E_inc for this geometry
        },
        "odd": {
            "r_min": 1250.0,
            "layer_thickness": 5.05,   # W(1.9)+G10(0.15)+GndHV(0.1)+Si(0.5)+Air(0.1)+PCB(1.3)+Air(0.25)+G10(0.75)
            "num_layers": 48,
            "collection": "ECalBarrelCollection",
            "nominal_sf": 0.0255,      # mean E_dep/E_inc for this geometry
        },
        "fccee_cld": {
            "r_min": 2150.0,
            "layer_thickness": 5.05,   # same layer structure as ODD
            "num_layers": 40,
            "collection": "ECalBarrelCollection",
            "nominal_sf": 0.0263,      # mean E_dep/E_inc for this geometry
        },
        "fccee_allegro": {
            "r_min": 2172.8,
            "layer_thickness": (2578.3 - 2172.8) / 11,  # ~36.86 mm (v03, 11 equal layers)
            "num_layers": 11,
            "collection": "ECalBarrelModuleThetaMerged",
            "nominal_sf": 0.162,       # mean E_dep/E_inc for this geometry
        },
    }

    def __init__(self, detector):
        if detector not in self.DETECTORS:
            raise ValueError(f"Unknown detector '{detector}'. "
                             f"Choose from: {list(self.DETECTORS.keys())}")
        cfg = self.DETECTORS[detector]
        self.detector = detector
        self.r_min = cfg["r_min"]
        self.num_layers = cfg["num_layers"]
        self.layer_thickness = cfg["layer_thickness"]
        self.r_max = self.r_min + self.num_layers * self.layer_thickness
        self.collection = cfg["collection"]
        self.contrib_collection = cfg["collection"] + "Contributions"
        self.nominal_sf = cfg.get("nominal_sf")

        self.layer_boundaries = np.array(
            [self.r_min + i * self.layer_thickness for i in range(self.num_layers + 1)]
        )

    def rho_to_layer(self, rho):
        """Map radial distance rho to layer index (0 to num_layers-1)."""
        return np.clip(
            ((rho - self.r_min) / self.layer_thickness).astype(int),
            0, self.num_layers - 1
        )

    def __repr__(self):
        return (f"LEMURSBarrelGeometry('{self.detector}', r_min={self.r_min:.1f}mm, "
                f"r_max={self.r_max:.1f}mm, layers={self.num_layers}, "
                f"collection='{self.collection}')")


def add_layer_info(data, geometry, verbose=True, remove_overflow=False):
    """
    Add layer number and coordinate info to data dictionary.
    Supports both SimpleBoxGeometry (planar, z-based) and
    ALLEGROGeometry (cylindrical, rho-based).

    Parameters:
    -----------
    geometry : SimpleBoxGeometry or ALLEGROGeometry
        Geometry object defining the calorimeter structure
    remove_overflow : bool
        If True, removes hits in overflow layer (old behavior - CAUSES ENERGY LOSS!)
        If False, caps overflow hits to max layer (RECOMMENDED - preserves energy!)
    """
    is_allegro = isinstance(geometry, (ALLEGROGeometry, LEMURSBarrelGeometry))

    if is_allegro:
        # Cylindrical geometry: layers determined by rho = sqrt(x^2 + y^2)
        rho = np.sqrt(data['x']**2 + data['y']**2)
        layer = geometry.rho_to_layer(rho)
        n_total = len(layer)

        if verbose:
            n_inside = np.sum((rho >= geometry.r_min) & (rho <= geometry.r_max))
            print(f"\n{'='*60}")
            print(f"ALLEGRO LAYER ASSIGNMENT (rho-based)")
            print(f"{'='*60}")
            print(f"Total hits:              {n_total:,}")
            print(f"Hits inside barrel:      {n_inside:,} ({100*n_inside/n_total:.1f}%)")
            print(f"Rho range:               [{rho.min():.1f}, {rho.max():.1f}] mm")
            print(f"Layer range:             [{layer.min()}, {layer.max()}]")
            print(f"{'='*60}\n")

        data['layer'] = layer
        return data

    # Planar geometry: layers stacked in z-direction
    coord = data['z']
    coord_name = 'z'
    layer_raw = np.floor((coord - geometry.z_start) / geometry.layer_thickness).astype(int)

    # Cap layer at num_layers-1 instead of removing!
    if remove_overflow:
        # OLD BEHAVIOR - removes overflow (BAD for high-f configs!)
        layer = np.where(layer_raw >= geometry.num_layers, -1, layer_raw)
        valid_mask = layer >= 0
        n_overflow = np.sum(~valid_mask)
        n_total = len(layer)

        if n_overflow > 0 and verbose:
            print(f"\n{'='*60}")
            print(f"OVERFLOW LAYER FILTERING (OLD METHOD - NOT RECOMMENDED)")
            print(f"{'='*60}")
            print(f"Total hits:              {n_total:,}")
            print(f"Hits in overflow layer:  {n_overflow:,} ({100*n_overflow/n_total:.2f}%)")
            print(f"Energy in overflow:      {data['energy'][~valid_mask].sum():.3f} GeV")
            print(f"  THIS ENERGY IS BEING LOST!")
            print(f"{'='*60}\n")

        # Apply filter
        filtered_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray) and len(value) == n_total:
                filtered_data[key] = value[valid_mask]
            else:
                filtered_data[key] = value

        filtered_data[coord_name] = coord[valid_mask]
        filtered_data['layer'] = layer[valid_mask]
        return filtered_data

    else:
        # NEW BEHAVIOR - cap at max layer (GOOD - preserves energy!)
        layer = np.clip(layer_raw, 0, geometry.num_layers - 1)
        n_overflow = np.sum(layer_raw >= geometry.num_layers)
        n_total = len(layer)

        if n_overflow > 0 and verbose:
            print(f"\n{'='*60}")
            print(f"OVERFLOW LAYER HANDLING (NEW METHOD - RECOMMENDED)")
            print(f"{'='*60}")
            print(f"Total hits:              {n_total:,}")
            print(f"Hits beyond layer {geometry.num_layers-1}:  {n_overflow:,} ({100*n_overflow/n_total:.2f}%)")
            print(f"Energy in overflow:      {data['energy'][layer_raw >= geometry.num_layers].sum():.3f} GeV")
            print(f" Capped to layer {geometry.num_layers-1} - ENERGY PRESERVED!")
            print(f"{'='*60}\n")

        # NO filtering - keep all hits!
        data[coord_name] = coord
        data['layer'] = layer
        return data