#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=merge_lemurs
#SBATCH --output=log/merge_lemurs_%j.out
#SBATCH --error=log/merge_lemurs_%j.err
#SBATCH --export=ALL

set -e

BASE_PATH="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
WORK_DIR="${WORK_DIR:-$(dirname "$BASE_PATH")}"
ALLSHOWERS_ROOT="${ALLSHOWERS_ROOT:-$WORK_DIR/AllShowers-AllGeometries}"

cd "$BASE_PATH"
source "$ALLSHOWERS_ROOT/.venv/bin/activate"

echo "Date: $(date) | Host: $(hostname) | Job: $SLURM_JOB_ID"

# CLD uses h5_1M for its 150k batch
H5_SUBDIRS_DEFAULT="h5_150k_100gev h5_850k_100gev"
H5_SUBDIRS_CLD="h5_1M h5_850k_100gev"

for det in odd par04_scipb par04_siw; do
    python -u scripts/lemurs/merge_h5_lemurs.py \
        --step merge-detector --detector "$det" \
        --h5-subdirs $H5_SUBDIRS_DEFAULT
done

python -u scripts/lemurs/merge_h5_lemurs.py \
    --step merge-detector --detector fccee_cld \
    --h5-subdirs $H5_SUBDIRS_CLD

python -u scripts/lemurs/merge_h5_lemurs.py --step combine-all

echo "Done: $(date)"
