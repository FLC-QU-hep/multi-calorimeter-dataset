#!/usr/bin/env python3
"""
Generate MC particles for LEMURS barrel calorimeter simulations (LCIO output).

Supports all 5 LEMURS detectors (arXiv:2509.05108):
  - par04_siw, par04_scipb, odd, fccee_cld, fccee_allegro

Angular distribution (identical for all detectors):
  cos(theta) uniform in [cos(theta_max), cos(theta_min)]
  phi uniform in [0, 2*pi]
  theta range: [0.87, 2.27] rad  (~50 to ~130 deg from z-axis)

Gun position: on the inner cylindrical surface of the ECal barrel,
at r = R_min - epsilon (epsilon = 1e-8 mm).

Position formulas:
  r_gun  = R_min - 1e-8 mm
  x_gun  = r_gun * cos(phi)
  y_gun  = r_gun * sin(phi)
  z_gun  = r_gun * cos(theta) / sin(theta)   [= r_gun * cot(theta)]

Momentum direction (radially outward into barrel):
  px = |p| * sin(theta) * cos(phi)
  py = |p| * sin(theta) * sin(phi)
  pz = |p| * cos(theta)
"""

import argparse
import math
import random
import time

import numpy as np
from pyLCIO import EVENT, IMPL, IOIMPL

# ============================================================================
# LEMURS DETECTOR CONFIGURATIONS
# r_min sources:
#   par04_siw/scipb : ddfastsim/Detector/xml/Par04_fullsim.xml (EMBarrel_rmin = 80*cm)
#   odd             : OpenDataDetectorEnvelopes.xml (ecal_b_rmin = 1250*mm)
#   fccee_cld       : k4geo FCCee_DectDimensions.xml (ECalBarrel_inner_radius = 2150*mm)
#   fccee_allegro   : ALLEGRO_o1_v03 DectDimensions.xml (EMBarrel_rmin = 2172.8 mm)
# ============================================================================

DETECTORS = {
    "par04_siw": {
        "r_min_mm": 800.0,
        "e_min_gev": 1.0, "e_max_gev": 1000.0,
        "num_layers": 90, "sf": 0.0321,
        "active": "Si(0.3mm)", "passive": "W(1.4mm)",
        "depth_mm": 153.0, "depth_x0": 36.3,
    },
    "par04_scipb": {
        "r_min_mm": 800.0,
        "e_min_gev": 1.0, "e_max_gev": 1000.0,
        "num_layers": 45, "sf": 0.0330,
        "active": "Polystyrene(1.2mm)", "passive": "Pb(4.4mm)",
        "depth_mm": 252.0, "depth_x0": 38.5,
    },
    "odd": {
        "r_min_mm": 1250.0,
        "e_min_gev": 1.0, "e_max_gev": 1000.0,
        "num_layers": 48, "sf": 0.0255,
        "active": "Si(0.5mm)", "passive": "W(1.9mm)",
        "depth_mm": 242.4, "depth_x0": 36.2,
    },
    "fccee_cld": {
        "r_min_mm": 2150.0,
        "e_min_gev": 1.0, "e_max_gev": 100.0,
        "num_layers": 40, "sf": 0.0257,
        "active": "Si(0.5mm)", "passive": "W(1.9mm)",
        "depth_mm": 202.0, "depth_x0": 21.5,
    },
    "fccee_allegro": {
        "r_min_mm": 2172.8,
        "e_min_gev": 1.0, "e_max_gev": 100.0,
        "num_layers": 12, "sf": 0.1430,
        "active": "LAr(1.2mm)", "passive": "Pb(2.0mm)",
        "depth_mm": 38.4, "depth_x0": 22.0,
    },
}

# LEMURS angular distribution (same for all detectors)
THETA_MIN_RAD = 0.87   # rad (~50 deg)
THETA_MAX_RAD = 2.27   # rad (~130 deg)
EPSILON_MM = 1e-8

PARTICLE_DATA = {
    22:  (0.0,      0.0),    # photon
    11:  (0.000511, -1.0),   # electron
    -11: (0.000511,  1.0),   # positron
}


class LEMURSParticleGenerator:
    """Single incident particles on a LEMURS barrel inner surface."""

    def __init__(self, detector: str, particle_ids: list = None,
                 e_min: float = None, e_max: float = None):
        if detector not in DETECTORS:
            raise ValueError(f"Unknown detector '{detector}'. "
                             f"Choose from: {list(DETECTORS.keys())}")
        self.detector = detector
        cfg = DETECTORS[detector]

        self.r_gun = cfg["r_min_mm"] - EPSILON_MM
        self.e_min = e_min if e_min is not None else cfg["e_min_gev"]
        self.e_max = e_max if e_max is not None else cfg["e_max_gev"]
        self.particle_ids = particle_ids or [22]

        self.cos_min = math.cos(THETA_MAX_RAD)  # cos(2.27) ~ -0.637
        self.cos_max = math.cos(THETA_MIN_RAD)  # cos(0.87) ~ +0.637

        for pid in self.particle_ids:
            if pid not in PARTICLE_DATA:
                raise ValueError(f"Unknown PDG ID {pid}")
        self.masses = {pid: PARTICLE_DATA[pid][0] for pid in self.particle_ids}
        self.charges = {pid: PARTICLE_DATA[pid][1] for pid in self.particle_ids}

    def __call__(self) -> dict:
        pid = random.choice(self.particle_ids)
        mass = self.masses[pid]
        charge = self.charges[pid]
        # Flat (uniform) energy sampling (matches LEMURS GPSflat)
        E = random.uniform(self.e_min, self.e_max)

        cos_theta = random.uniform(self.cos_min, self.cos_max)
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta ** 2))
        phi = random.uniform(0.0, 2.0 * math.pi)

        x_gun = self.r_gun * math.cos(phi)
        y_gun = self.r_gun * math.sin(phi)
        z_gun = self.r_gun * cos_theta / sin_theta  # r_gun * cot(theta)

        p_norm = math.sqrt(max(0.0, E ** 2 - mass ** 2))
        px = p_norm * sin_theta * math.cos(phi)
        py = p_norm * sin_theta * math.sin(phi)
        pz = p_norm * cos_theta

        return {
            "particle_id": pid,
            "position": (x_gun, y_gun, z_gun),
            "momentum": (px, py, pz),
            "mass": mass,
            "charge": charge,
        }

    def __str__(self) -> str:
        cfg = DETECTORS[self.detector]
        z_max = self.r_gun * abs(math.cos(THETA_MIN_RAD) / math.sin(THETA_MIN_RAD))
        return "\n".join([
            f"LEMURSParticleGenerator ({self.detector}):",
            f"  R_gun         : {self.r_gun:.4f} mm  (R_min={cfg['r_min_mm']})",
            f"  theta range   : [{math.degrees(THETA_MIN_RAD):.2f}, "
            f"{math.degrees(THETA_MAX_RAD):.2f}] deg",
            f"  cos(theta)    : [{self.cos_min:.4f}, {self.cos_max:.4f}]  (uniform)",
            f"  |z_gun| max   : {z_max:.1f} mm",
            f"  Energy        : [{self.e_min}, {self.e_max}] GeV",
            f"  Particles     : {self.particle_ids}",
            f"  Detector      : {cfg['active']} / {cfg['passive']}, "
            f"{cfg['num_layers']} layers, SF={cfg['sf']}",
        ])


# ============================================================================
# LCIO I/O
# ============================================================================

def _make_mc_particle(props: dict):
    mc = IMPL.MCParticleImpl()
    mc.setPDG(props["particle_id"])
    mc.setMomentum(np.array(props["momentum"], dtype=np.float64))
    mc.setMass(props["mass"])
    mc.setCharge(props["charge"])
    mc.setVertex(np.array(props["position"], dtype=np.float64))
    mc.setEndpoint(np.array(props["position"], dtype=np.float64))
    mc.setGeneratorStatus(1)
    return mc


def write_lcio(file_name: str, generator: LEMURSParticleGenerator,
               num_events: int) -> None:
    writer = IOIMPL.LCFactory.getInstance().createLCWriter()
    writer.open(file_name, EVENT.LCIO.WRITE_NEW)

    run = IMPL.LCRunHeaderImpl()
    run.setRunNumber(0)
    writer.writeRunHeader(run)

    t0 = int(time.time() * 1e9)
    dt = 200

    for i in range(num_events):
        props = generator()
        event = IMPL.LCEventImpl()
        event.setEventNumber(i)
        event.setTimeStamp(t0 + i * dt)
        event.setRunNumber(0)

        col = IMPL.LCCollectionVec(EVENT.LCIO.MCPARTICLE)
        col.addElement(_make_mc_particle(props))
        event.addCollection(col, EVENT.LCIO.MCPARTICLE)
        writer.writeEvent(event)

    writer.flush()
    writer.close()


# ============================================================================
# CLI
# ============================================================================

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Generate MC particles for LEMURS barrel calorimeters (LCIO output)."
    )
    parser.add_argument("--detector", required=True,
                        choices=list(DETECTORS.keys()),
                        help="LEMURS detector name.")
    parser.add_argument("--particle-ids", type=int, nargs="+", default=[22],
                        help="PDG IDs (default: 22 = photon).")
    parser.add_argument("--min-energy", type=float, default=None,
                        help="Min energy [GeV] (default: from detector config).")
    parser.add_argument("--max-energy", type=float, default=None,
                        help="Max energy [GeV] (default: from detector config).")
    parser.add_argument("--num-events", type=int, default=1000,
                        help="Number of events (default: 1000).")
    parser.add_argument("--output", type=str, default="lemurs_particles.slcio",
                        help="Output LCIO file.")
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    generator = LEMURSParticleGenerator(
        detector=parsed.detector,
        particle_ids=parsed.particle_ids,
        e_min=parsed.min_energy,
        e_max=parsed.max_energy,
    )
    print(generator)
    write_lcio(parsed.output, generator, parsed.num_events)
    print(f"Wrote {parsed.num_events} events to {parsed.output}")


if __name__ == "__main__":
    main()