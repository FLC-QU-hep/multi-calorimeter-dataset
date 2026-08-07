#!/usr/bin/env python3
"""
Generate MC particles and write them to an LCIO file for ddsim simulation.

Generates single incident particles with configurable energy and angular
distributions using SPHERICAL coordinates (Cone).

IMPORTANT ON ANGLES:
To achieve a coverage of +/- 45 degrees (from -45 to +45):
- Set theta-min = 0 (perpendicular)
- Set theta-max = 0.7854 (45 degrees opening angle)
- Set phi-min = -3.14 and phi-max = 3.14 (Full rotation)

The 'negative' angles are handled by the Phi rotation (pointing left vs right).
"""

import argparse
import math
import random
import time
from collections.abc import Callable, Iterable

import numpy as np
from pyLCIO import EVENT, IMPL, IOIMPL


PARTICLE_DATA = {
    22: (0.0, 0.0),         # photon
    11: (0.000511, -1.0),   # electron
    -11: (0.000511, 1.0),   # positron
    211: (0.13957, 1.0),    # pi+
    -211: (0.13957, -1.0),  # pi-
    321: (0.49368, 1.0),    # K+
    -321: (0.49368, -1.0),  # K-
    130: (0.49761, 0.0),    # K_L
    2212: (0.93827, 1.0),   # proton
    -2212: (0.93827, -1.0), # antiproton
    2112: (0.93957, 0.0),   # neutron
    -2112: (0.93957, 0.0),  # antineutron
}


class ParticleGenerator:
    def __init__(
        self,
        particle_ids: Iterable[int],
        position: Iterable[float] = (0.0, 0.0, -200.0),
        min_energy: float = 1.0,
        max_energy: float = 100.0,
        enable_angles: bool = False,
        theta_min: float = 0.0,
        theta_max: float = 0.7854,
        phi_min: float = -math.pi,
        phi_max: float = math.pi,
    ) -> None:
        self.particle_ids = list(particle_ids)
        position = tuple(position)
        if len(position) != 3:
            raise ValueError("Position must be a tuple of three floats (x, y, z).")
        self.position: tuple[float, float, float] = position
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.enable_angles = enable_angles
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.phi_min = phi_min
        self.phi_max = phi_max

        # Validate particle IDs
        self.masses = {}
        self.charges = {}
        for pid in self.particle_ids:
            if pid not in PARTICLE_DATA:
                raise ValueError(
                    f"Unknown PDG ID {pid}. Add it to PARTICLE_DATA dictionary."
                )
            self.masses[pid] = PARTICLE_DATA[pid][0]
            self.charges[pid] = PARTICLE_DATA[pid][1]

    def __call__(self) -> dict[str, int | float | tuple[float, float, float]]:
        pid = random.choice(self.particle_ids)
        position = self.position
        energy = random.uniform(self.min_energy, self.max_energy)
        charge = self.charges[pid]
        mass = self.masses[pid]

        momentum_norm = math.sqrt(energy**2 - mass**2)

        if self.enable_angles:
            # === SPHERICAL GEOMETRY (CONE) ===
            # We sample cos(theta) to ensure uniform distribution on the sphere cap.
            # cos(0) = 1 (center), cos(max_angle) < 1
            
            cos_min = math.cos(self.theta_max)
            cos_max = math.cos(self.theta_min)
            
            # This logic ensures strict adherence to the max angle.
            # No particle will exceed theta_max (e.g. 45 deg).
            cos_theta = random.uniform(cos_min, cos_max)
            sin_theta = math.sqrt(1.0 - cos_theta**2)
            
            # Phi rotates the particle around the Z-axis.
            # - When Phi is 0, particle goes +X
            # - When Phi is PI, particle goes -X (this creates the -45 deg effect)
            phi = random.uniform(self.phi_min, self.phi_max)

            # Convert Spherical to Cartesian (Z is beam axis)
            px = momentum_norm * sin_theta * math.cos(phi)
            py = momentum_norm * sin_theta * math.sin(phi)
            pz = momentum_norm * cos_theta
        else:
            # Perpendicular incidence: along +z axis
            px = 0.0
            py = 0.0
            pz = momentum_norm

        momentum = (px, py, pz)

        return {
            "particle_id": pid,
            "position": position,
            "momentum": momentum,
            "mass": mass,
            "charge": charge,
        }

    def __str__(self):
        result = "ParticleGenerator:\n"
        result += f"  Position: {self.position}\n"
        result += f"  Energy range: [{self.min_energy}, {self.max_energy}] GeV\n"
        result += f"  Enable Angles: {self.enable_angles}\n"
        if self.enable_angles:
            result += f"  Theta range: [{self.theta_min:.4f}, {self.theta_max:.4f}] rad "
            result += f"[{math.degrees(self.theta_min):.1f}°, {math.degrees(self.theta_max):.1f}°]\n"
            result += f"  Phi range:   [{self.phi_min:.4f}, {self.phi_max:.4f}] rad\n"
        return result[:-1]


def create_mc_particle(properties):
    mc_particle = IMPL.MCParticleImpl()
    mc_particle.setPDG(properties["particle_id"])
    mc_particle.setMomentum(np.array(properties["momentum"], dtype=np.float64))
    mc_particle.setMass(properties["mass"])
    mc_particle.setCharge(properties["charge"])
    mc_particle.setVertex(np.array(properties["position"], dtype=np.float64))
    mc_particle.setEndpoint(np.array(properties["position"], dtype=np.float64))
    mc_particle.setGeneratorStatus(1)
    return mc_particle


def write(file_name, particle_generator, num_events):
    writer = IOIMPL.LCFactory.getInstance().createLCWriter()
    writer.open(file_name, EVENT.LCIO.WRITE_NEW)

    run = IMPL.LCRunHeaderImpl()
    run.setRunNumber(0)
    writer.writeRunHeader(run)

    time_stamp = int(time.time() * 1e9)
    delta_time = 200

    for i in range(num_events):
        particle_properties = particle_generator()
        event = IMPL.LCEventImpl()
        event.setEventNumber(i)
        event.setTimeStamp(time_stamp + i * delta_time)
        event.setRunNumber(0)

        mc_collection = IMPL.LCCollectionVec(EVENT.LCIO.MCPARTICLE)
        mc_particle = create_mc_particle(particle_properties)
        mc_collection.addElement(mc_particle)

        event.addCollection(mc_collection, EVENT.LCIO.MCPARTICLE)
        writer.writeEvent(event)

    writer.flush()
    writer.close()


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Generate MC particles and write them to an LCIO file for ddsim simulation.",
    )
    parser.add_argument("--particle-ids", type=int, nargs="+", default=[22], help="List of PDG particle IDs.")
    parser.add_argument("--position", type=float, nargs=3, default=(0.0, 0.0, -200.0), help="Position (x, y, z) [mm].")
    parser.add_argument("--min-energy", type=float, default=1.0, help="Min energy [GeV].")
    parser.add_argument("--max-energy", type=float, default=100.0, help="Max energy [GeV].")
    parser.add_argument("--enable-angles", action="store_true", help="Enable angular distribution.")
    
    # Updated arguments for Spherical Geometry
    parser.add_argument("--theta-min", type=float, default=0.0, help="Min polar angle [rad] (default: 0).")
    parser.add_argument("--theta-max", type=float, default=0.7854, help="Max polar angle [rad] (default: 45deg).")
    parser.add_argument("--phi-min", type=float, default=-math.pi, help="Min azimuthal angle [rad].")
    parser.add_argument("--phi-max", type=float, default=math.pi, help="Max azimuthal angle [rad].")
    
    parser.add_argument("--output", type=str, default="mc_particles.slcio", help="Output file name.")
    parser.add_argument("--num-events", type=int, default=200, help="Number of events.")
    return parser.parse_args(args)


def main(args=None):
    parsed_args = parse_args(args)
    generator = ParticleGenerator(
        particle_ids=parsed_args.particle_ids,
        position=parsed_args.position,
        min_energy=parsed_args.min_energy,
        max_energy=parsed_args.max_energy,
        enable_angles=parsed_args.enable_angles,
        theta_min=parsed_args.theta_min,
        theta_max=parsed_args.theta_max,
        phi_min=parsed_args.phi_min,
        phi_max=parsed_args.phi_max,
    )
    print(generator)
    write(parsed_args.output, generator, parsed_args.num_events)
    print(f"Wrote {parsed_args.num_events} events to {parsed_args.output}")


if __name__ == "__main__":
    main()