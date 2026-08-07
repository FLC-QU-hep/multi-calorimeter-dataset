#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=allegro_h5
#SBATCH --output=log/allegro/h5_creation_%j.out
#SBATCH --error=log/allegro/h5_creation_%j.err

# ============================================================================
# Combine individual ALLEGRO HDF5 files into a unified dataset.
# ============================================================================

BASE_PATH="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../../.." && pwd)}"

mkdir -p $BASE_PATH/log/allegro

cd $BASE_PATH
source "${KEY4HEP_SETUP:-/cvmfs/sw.hsf.org/key4hep/setup.sh}" -r "${KEY4HEP_RELEASE:-2025-05-29}"

python scripts/lemurs/multi_dataset_h5_creation.py \
    --input-dir output_dataset/ALLEGRO/h5/final \
    --output output_dataset/ALLEGRO/h5/allegro_1M.h5