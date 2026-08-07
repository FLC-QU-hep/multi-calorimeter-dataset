#!/usr/bin/env bash
# Set/override REPO_ROOT for the dataset scripts.
#
# The run/ and scripts/ entry points auto-derive the repo root from their own
# location and honour $REPO_ROOT when set. To run from a different checkout,
# export it (or source this file) before submitting jobs:
#
#     source env.sh                     # sets REPO_ROOT to this file's directory
#     # or: export REPO_ROOT=/path/to/multi-calorimeter-dataset
#
# sbatch's default --export=ALL forwards REPO_ROOT into jobs; the self-resubmit
# chains use --export=ALL,OFFSET=... so it also reaches child jobs.
#
# NOTE: under sbatch, $0 is a spool copy, so the in-script auto-derive fallback
# is unreliable for batch jobs — export REPO_ROOT (via this file) before submitting.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
export REPO_ROOT

# Pre-create the (gitignored) SLURM log tree so relative `#SBATCH --output=log/...`
# directives resolve when jobs are submitted from the repo root. SLURM opens the
# output file before the job body runs, so the dirs must exist beforehand.
mkdir -p "$REPO_ROOT"/log/{allegro,allegro_final,fccee_cld,fccee_cld_100gev,h5_creation,odd_100gev,odd_850k,par04_scipb_100gev,par04_siw_100gev,processing_configs_angles,scipb_850k,sf_nlayers_angles,sf_nlayers_angles_zeroshot,siw_850k}
