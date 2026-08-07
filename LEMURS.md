# LEMURS: Step-Level G4 Point Cloud Datasets

[![arXiv](https://img.shields.io/badge/arXiv-2608.18233-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.18233)
[![DOI](https://img.shields.io/badge/DOI-10.25592%2Fuhhfdm.19103-blue)](https://doi.org/10.25592/uhhfdm.19103)
[![Python Version](https://img.shields.io/badge/Python_3.12-306998?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/FLC-QU-hep/multi-calorimeter-dataset?tab=MIT-1-ov-file)

---

## 1. Project Goal

**LEMURS** ([arXiv:2509.05108](https://arxiv.org/abs/2509.05108)) released shower datasets for 5 calorimeters, but only as processed cell-level outputs, without Geant4 step-level data. This project **resimulates all 5 LEMURS calorimeters from scratch** with full G4 step positions, converts them to point clouds, and uses them to study transfer learning.

### The Experiment

Train the same model, [**AllShowers**](https://github.com/FLC-QU-hep/AllShowers)
(paper branch [`multi-geometry`](https://github.com/FLC-QU-hep/AllShowers/tree/multi-geometry)),
with two different pre-trainings, then
fine-tune both on ALLEGRO and compare. The per-layer point-count model,
[**PointCountFM**](https://github.com/FLC-QU-hep/PointCountFM), is trained on
the same datasets. Pre-trained weights for both are on Hugging Face:
[AllShowers-multi-geometry](https://huggingface.co/FLC-QU-hep/AllShowers-multi-geometry)
and
[PointCountFM-multi-geometry](https://huggingface.co/FLC-QU-hep/PointCountFM-multi-geometry).

| Pre-training | Geometry | Dataset |
|---|---|---|
| **A** | SimpleBox (toy, variable SF/nlayers) | `4Mshowers_angles.h5`, already exists |
| **B** | LEMURS (5 realistic calorimeters) | resimulated with G4 steps (this project) |

**Research question:** Does pre-training on realistic calorimeter geometries (B) transfer better to ALLEGRO than pre-training on a toy geometry (A)?

**Pipeline:** G4 step simulation → local frame → HDF5 → OT matching → pre-train → fine-tune on ALLEGRO → compare

---

## 2. Simulation: LCIO Approach

### Why LCIO (not GPS)

GPS places the gun at the origin and cannot set an arbitrary 3D position on the barrel surface. LCIO pre-generates MC particles with full position+momentum control per event.

### Angular Distribution (same for all 5 detectors)

| Parameter | Value |
|---|---|
| Particle | Photon (γ), PDG=22 |
| cos(θ) | Uniform in [cos(2.27), cos(0.87)] → isotropic solid angle |
| θ range | [0.87, 2.27] rad (~50–130 deg from z-axis) |
| φ | Uniform [0, 2π] |

### Gun Position Formula

```
R_GUN = R_min - 1e-8 mm    (per-detector, epsilon inside vacuum)

φ      ~ Uniform[0, 2π]
cos(θ) ~ Uniform[cos(2.27), cos(0.87)]  →  θ = arccos(sample)

gun_x = R_GUN * cos(φ)
gun_y = R_GUN * sin(φ)
gun_z = R_GUN * cos(θ) / sin(θ)     (= R_GUN * cot θ)

px = E * sin(θ) * cos(φ)
py = E * sin(θ) * sin(φ)
pz = E * cos(θ)
```

### Key Scripts

| Script | Purpose |
|---|---|
| `scripts/lemurs/create_mc_particles.py` | Generalized LCIO generator for all 5 detectors. `--detector` flag. Uses `pyLCIO`, **key4hep python only, NOT conda** |
| `ddfastsim/options/Par04/Par04_ddsim_steer_steps_lcio.py` | Par04_SiW steering: HitCreationMode=2, LCIO input |
| `ddfastsim/options/Par04_SciPb/Par04_ddsim_steer_steps_lcio.py` | Par04_SciPb steering |
| `ddfastsim/options/ODD/ODD_ddsim_steer_steps_lcio.py` | ODD steering |
| `ddfastsim/options/FCCeeCLD/FCCeeCLD_ddsim_steer_steps_lcio.py` | FCCee_CLD steering |
| `ddfastsim/options/FCCeeALLEGRO/FCCeeALLEGRO_ddsim_steer_steps_lcio.py` | ALLEGRO steering |

### Running (Par04_SiW example)

```bash
# 1M = 150k + 850k @ 100 GeV (submit_inline_100gev auto-chains + postprocesses):
sbatch run/lemurs/par04_siw/simulation/submit_inline_100gev.sh
sbatch run/lemurs/par04_siw/simulation/submit_850k_100gev.sh
```

### Running (ALLEGRO)

```bash
sbatch run/lemurs/allegro/simulation/submit_final.sh
```

---

## 3. Processing Pipeline (per detector)

1. **G4 simulation** → ROOT files with step positions
2. **ROOT → HDF5** → local frame recentering, layer assignment, 1mm grid clustering
3. **HDF5 merge** → single training-ready file
4. **OT matching** → adds `target/point_clouds` for model training

### Local Frame

```python
# Cylindrical local frame (same for all barrel detectors):
dh_t = r_hit * wrap(phi_hit - phi_gun)   # tangential [mm]
dh_z = z_hit - z_gun                      # beam-axis  [mm]
layer = rho_to_layer(rho)                  # integer layer index
```

### HDF5 Schema (showerdata-compatible)

```
showers           → vlen float32: [dh_t, dh_z, layer, E_GeV, 0] per hit
energies          → (N,1) E_incident [GeV]
directions        → (N,3) unit vector (sinθcosφ, sinθsinφ, cosθ)
sampling_fraction → (N,1) E_dep/E_incident
num_layers        → (N,1) distinct layers hit
gun_position      → (N,3) (gun_x, gun_y, gun_z) [mm]
```

---

## 4. Key File Paths

```
WORK_DIR  = <workspace parent of the repo root> (i.e. dirname of REPO_ROOT; export WORK_DIR to override)
BASE_PATH = WORK_DIR/multi-calorimeter-dataset/  (== REPO_ROOT; set by `source env.sh`)

MC Generation:
  BASE_PATH/scripts/lemurs/create_mc_particles.py       (all 5 detectors)
  BASE_PATH/scripts/simplebox/create_mc_particles.py    (SimpleBox only)

Steering Files:
  WORK_DIR/ddfastsim/options/Par04/Par04_ddsim_steer_steps_lcio.py
  WORK_DIR/ddfastsim/options/Par04_SciPb/Par04_ddsim_steer_steps_lcio.py
  WORK_DIR/ddfastsim/options/ODD/ODD_ddsim_steer_steps_lcio.py
  WORK_DIR/ddfastsim/options/FCCeeCLD/FCCeeCLD_ddsim_steer_steps_lcio.py
  WORK_DIR/ddfastsim/options/FCCeeALLEGRO/FCCeeALLEGRO_ddsim_steer_steps_lcio.py

SLURM Submission:
  BASE_PATH/run/lemurs/par04_siw/simulation/submit_inline_100gev.sh
  BASE_PATH/run/lemurs/par04_scipb/simulation/submit_inline_100gev.sh
  BASE_PATH/run/lemurs/odd/simulation/submit_inline_100gev.sh
  BASE_PATH/run/lemurs/fccee_cld/simulation/submit_inline.sh
  BASE_PATH/run/lemurs/allegro/simulation/submit_final.sh

Processing:
  BASE_PATH/scripts/lemurs/process_root_to_h5_allegro.py
  BASE_PATH/utils/calo_geometry.py

Output:
  BASE_PATH/output_dataset/{Par04_SiW,Par04_SciPb,ODD,FCCee_CLD,ALLEGRO}/root/
  BASE_PATH/output_dataset/{Par04_SiW,Par04_SciPb,ODD,FCCee_CLD,ALLEGRO}/h5/

Model:
  WORK_DIR/AllShowers-AllGeometries/
  WORK_DIR/AllShowers-AllGeometries/conf/sf_nlayers_angles.yaml
```

---

## 5. Environment

```bash
# Key4hep (simulation + pyLCIO):
source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2025-05-29
source WORK_DIR/ddfastsim/install/bin/thisDDFastSim.sh

# For Par04/ODD/CLD (local XML):
source WORK_DIR/k4geo/install/bin/thisk4geo.sh

# For ALLEGRO (cvmfs XML):
# DO NOT source thisk4geo.sh — conflicts with cvmfs

# Conda (processing, OT matching, training):
conda activate calo-transfer
```

- `pyLCIO` → key4hep only, not in conda
- `uproot`, `h5py`, `torch` → conda `calo-transfer`

---

## 6. References

- Multi-geometry transfer study (SimpleBox → LEMURS / ALLEGRO): the paper this
  repository accompanies, [arXiv:2608.18233](https://arxiv.org/abs/2608.18233)
- LEMURS: [arXiv:2509.05108](https://arxiv.org/abs/2509.05108)
- AllShowers: [github.com/FLC-QU-hep/AllShowers](https://github.com/FLC-QU-hep/AllShowers),
  branch [`multi-geometry`](https://github.com/FLC-QU-hep/AllShowers/tree/multi-geometry)
  (weights: [Hugging Face](https://huggingface.co/FLC-QU-hep/AllShowers-multi-geometry))
- PointCountFM: [github.com/FLC-QU-hep/PointCountFM](https://github.com/FLC-QU-hep/PointCountFM),
  branch [`multi-geometry`](https://github.com/FLC-QU-hep/PointCountFM/tree/multi-geometry)
  (weights: [Hugging Face](https://huggingface.co/FLC-QU-hep/PointCountFM-multi-geometry))
- ddfastsim: [gitlab.cern.ch/fastsim/ddfastsim](https://gitlab.cern.ch/fastsim/ddfastsim)
- ALLEGRO detector: FCC-ee Conceptual Design Report,
  [k4geo](https://github.com/key4hep/k4geo) geometry `ALLEGRO_o1_v03`
- ODD ECal: [github.com/OpenDataDetector/OpenDataDetector](https://github.com/OpenDataDetector/OpenDataDetector)