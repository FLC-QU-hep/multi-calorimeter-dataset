# Multi-Calorimeter Dataset

[![arXiv](https://img.shields.io/badge/arXiv-2608.18233-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.18233)
[![DOI](https://img.shields.io/badge/DOI-10.25592%2Fuhhfdm.19103-blue)](https://doi.org/10.25592/uhhfdm.19103)
[![Python Version](https://img.shields.io/badge/Python_3.12-306998?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/FLC-QU-hep/multi-calorimeter-dataset?tab=MIT-1-ov-file)

Geant4 calorimeter-shower datasets (DD4hep/DDSim → ROOT → HDF5) in two contexts:
- **SimpleBox**: toy detector, variable sampling-fraction / n-layers (`4Mshowers_angles.h5`).
- **LEMURS**: 5 realistic calorimeters: **Par04_SiW**, **Par04_SciPb**, **ODD**, **FCCee_CLD** (the four pre-training detectors, merged into `lemurs_4M.h5`) + **FCCee_ALLEGRO** v03 (fine-tune target, `allegro_100k.h5`).

See [LEMURS.md](LEMURS.md) for the resimulation methodology & pipeline.

## Project Structure

```
├── calo_configs/base/          # Base detector XML templates
├── sf_nlayers_angles/          # SimpleBox SF/n-layer calibration → config generation
│                               #   (measure_sf → fit_model → generate_*_configs)
├── scripts/
│   ├── simplebox/              # SimpleBox: MC gun, ROOT→H5, batch merge
│   ├── lemurs/                 # LEMURS: MC generator, ROOT→H5, merge, create_cond
│   └── plots/                  # Plotting
├── utils/                      # Geometry, preprocessing, showerdata I/O
├── run/
│   ├── build.sh                # Geant4 env (k4geo + ddfastsim)
│   ├── simplebox/              # SimpleBox: simulation + h5_creation
│   └── lemurs/                 # 5 detectors + merge_lemurs.sh
└── output_dataset/             # Per-detector outputs (gitignored)
```

## Workflows

> Run all workflows **from the repo root**, and `source env.sh` first, since it sets
> `REPO_ROOT` and pre-creates the gitignored `./log/` tree that the relative
> `#SBATCH --output=log/…` directives write to.

> The final conversion to the training-ready HDF5 format uses the
> [showerdata](https://github.com/FLC-QU-hep/ShowerData) package (MIT):
> `pip install git+https://github.com/FLC-QU-hep/ShowerData`.

### SimpleBox (SF + Layers)

```bash
source env.sh
python sf_nlayers_angles/generate_final_configs.py
sbatch run/simplebox/simulation/submit_angles.sh
sbatch run/simplebox/h5_creation/parallel_processing_angles.sh
sbatch run/simplebox/h5_creation/create_angles.sh
```

### LEMURS Step-Level Simulation

```bash
# Par04_SiW / Par04_SciPb / ODD (1M = 150k + 850k @ 100 GeV):
sbatch run/lemurs/par04_siw/simulation/submit_inline_100gev.sh   # 150k batch (auto-chains + postprocess)
sbatch run/lemurs/par04_siw/simulation/submit_850k_100gev.sh     # 850k batch

# FCCee_CLD (1M = h5_1M + 850k):
sbatch run/lemurs/fccee_cld/simulation/submit_inline.sh
sbatch run/lemurs/fccee_cld/simulation/submit_inline_100gev.sh
sbatch run/lemurs/fccee_cld/simulation/postprocess_chain.sh

# ALLEGRO (fine-tune target):
sbatch run/lemurs/allegro/simulation/submit_final.sh
sbatch run/lemurs/allegro/h5_creation/create.sh

# Merge the 4 pretrain detectors → lemurs_4M.h5:
sbatch run/lemurs/merge_lemurs.sh
```

### Zeroshot

```bash
python sf_nlayers_angles/generate_zeroshot_config.py
sbatch run/simplebox/simulation/submit_angles_zeroshot.sh
sbatch run/simplebox/h5_creation/create_angles_zeroshot.sh
```

## Setup

**Processing / analysis** (ROOT→HDF5, merge, plots), a standard Python env:

```bash
pip install -r requirements.txt        # or: conda env create -f environment.yml
```

**Simulation** (Geant4 step-level showers) uses a separate, non-pip **Key4hep**
stack (release 2025-05-29) plus `k4geo`, `DD4hep/DDSim`, `pyLCIO`, and the
**ddfastsim** DDFastSim plugin. ddfastsim is an *external dependency* released
separately and is not included here. `run/build.sh` builds `k4geo` /
`OpenDataDetector` / `ddfastsim` assumed present under the workspace parent. Point
the scripts at your install with `KEY4HEP_SETUP`, `KEY4HEP_RELEASE`, `DDFASTSIM_DIR`.

Paths auto-derive from each script's location and honour `$REPO_ROOT`. To run from
a different checkout, `source env.sh` (or `export REPO_ROOT=…`) before submitting,
and sbatch's `--export=ALL` forwards it. **Cluster note:** the `#SBATCH --partition=allcpu`
and `module load maxwell mamba` lines are DESY-Maxwell specific, so change the partition
to your site's and the env activation to `conda activate` your `environment.yml` env.

## Data Access

The generated datasets (`4Mshowers_angles.h5`, `lemurs_4M.h5`, `allegro_100k.h5`, …)
are **not** stored in git (`output_dataset/` is gitignored). They are deposited
as a single record in the Universität Hamburg Research Data Repository,
DOI: [10.25592/uhhfdm.19103](https://doi.org/10.25592/uhhfdm.19103)
(mirrored in [CITATION.cff](CITATION.cff)). The record's own README documents
its file naming and the HDF5 schema.
Alternatively, regenerate them from scratch via the [Workflows](#workflows) above
(requires the Key4hep / ddfastsim simulation stack).

## Related Releases

Everything from the multi-geometry paper, in one place:

- **Paper**: arXiv 2608.18233 (badge above). Companion transfer study, JINST 21 (2026) P07037, [10.1088/1748-0221/21/07/P07037](https://doi.org/10.1088/1748-0221/21/07/P07037).
- **Data**: this record, DOI [10.25592/uhhfdm.19103](https://doi.org/10.25592/uhhfdm.19103) (SimpleBox 4M + mini, LEMURS 4M, ALLEGRO 100k, held-out test sets).
- **Weights**: [FLC-QU-hep/AllShowers-multi-geometry](https://huggingface.co/FLC-QU-hep/AllShowers-multi-geometry) and [FLC-QU-hep/PointCountFM-multi-geometry](https://huggingface.co/FLC-QU-hep/PointCountFM-multi-geometry) on Hugging Face (Apache-2.0).
- **Code**: [AllShowers](https://github.com/FLC-QU-hep/AllShowers/tree/multi-geometry) and [PointCountFM](https://github.com/FLC-QU-hep/PointCountFM/tree/multi-geometry), branch `multi-geometry`, plus this repository for the simulation chain.

## License

Released under the [MIT License](LICENSE).

## Citation

If you use this dataset or code, please cite it (machine-readable metadata in
[CITATION.cff](CITATION.cff)) and the accompanying multi-geometry paper (see
the arXiv badge above).