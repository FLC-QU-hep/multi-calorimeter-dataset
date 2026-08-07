#!/bin/bash
#SBATCH --job-name=process_configs_angles
#SBATCH --partition=allcpu
#SBATCH --array=0-999%200
#SBATCH --time=0:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=log/processing_configs_angles/config_%a.out
#SBATCH --error=log/processing_configs_angles/config_%a.err

# ============================================================================
# Process ROOT -> HDF5 for sf_nlayers_angles (auto-chains batches 0-9999)
# To run all 10k configs: sbatch --export=ALL,OFFSET=0 run/simplebox/h5_creation/parallel_processing_angles.sh
# ============================================================================

BASE_PATH="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"
OFFSET=${OFFSET:-0}
CONFIG_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))

mkdir -p $BASE_PATH/log/processing_configs_angles

module load maxwell mamba 2>/dev/null
. mamba-init 2>/dev/null || true
conda activate calo-transfer 2>/dev/null

cd $BASE_PATH

echo "Processing config $CONFIG_ID (array=$SLURM_ARRAY_TASK_ID, offset=$OFFSET)"
python scripts/simplebox/process_root_to_h5.py --config_id $CONFIG_ID --ref-dir sf_nlayers_angles --sub-dir SiW_final --n-showers 400

# Auto-submit next batch when task 999 completes
if [ $SLURM_ARRAY_TASK_ID -eq 999 ]; then
    NEXT_OFFSET=$((OFFSET + 1000))
    if [ $NEXT_OFFSET -lt 10000 ]; then
        echo "Auto-submitting next batch with OFFSET=$NEXT_OFFSET"
        sbatch --export=ALL,OFFSET=$NEXT_OFFSET $BASE_PATH/run/simplebox/h5_creation/parallel_processing_angles.sh
    else
        echo "All 10k configs processed!"
    fi
fi