#!/bin/bash
#SBATCH --job-name=multi_h5_angles_zeroshot
#SBATCH --partition=allcpu
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=log/h5_creation/multi_dataset_angles_zeroshot_%j.out
#SBATCH --error=log/h5_creation/multi_dataset_angles_zeroshot_%j.err

# ============================================================================
# H5 creation for zero-shot interpolation dataset: sf=0.035, num_layers=35
# 1 config, 100k events
# ============================================================================

BASE_PATH="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"
WORK_DIR="${WORK_DIR:-$(dirname "$BASE_PATH")}"
ALLSHOWERS_ROOT="${ALLSHOWERS_ROOT:-$WORK_DIR/AllShowers-AllGeometries}"

echo "=========================================="
echo "Starting H5 Creation (zero-shot: sf=0.035, nl=35)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p "$BASE_PATH/log/h5_creation"

module load maxwell mamba 2>/dev/null
. mamba-init 2>/dev/null || true
conda activate calo-transfer 2>/dev/null

cd "$BASE_PATH"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ""
echo ">>> [1/2] ROOT -> intermediate H5..."
python scripts/simplebox/multi_dataset_h5_creation.py \
    --ref-dir sf_nlayers_angles \
    --sub-dir SiW_zeroshot \
    --n-digits 3 \
    --n-showers 100000 \
    --metadata-file zeroshot_metadata.json \
    --output-name 1simplebox_100k_1-100GeV_sf0035_nl35_zeroshot.h5

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "✗ FAILED: Step 1+2 failed (exit $EXIT_CODE)"
    exit $EXIT_CODE
fi
echo "✓ Steps 1+2 done"

echo ""
echo ">>> [2/2] intermediate H5 -> showerdata format..."
conda deactivate

ALLSHOWERS=$ALLSHOWERS_ROOT
source ${ALLSHOWERS}/.venv/bin/activate

TEMP="$BASE_PATH/output_dataset/SimpleBox/h5/sf_nlayers_angles/SiW_zeroshot/temp"
DST="$BASE_PATH/output_dataset/SimpleBox/showerdata/sf_nlayers_angles/SiW_zeroshot/1simplebox_100k_1-100GeV_sf0035_nl35_zeroshot.h5"

REPO_ROOT="$BASE_PATH" TEMP="$TEMP" DST="$DST" python3 - <<'PYEOF'
import os
import sys
sys.path.insert(0, os.environ["REPO_ROOT"])
from utils.showerdata_utils import combine_temp_to_showerdata

combine_temp_to_showerdata(os.environ["TEMP"], os.environ["DST"])
PYEOF

EXIT_CODE=$?
echo ""
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: zero-shot dataset ready"
    echo "  -> output_dataset/SimpleBox/showerdata/sf_nlayers_angles/SiW_zeroshot/"
else
    echo "✗ FAILED: Step 3 failed"
fi
echo "=========================================="

exit $EXIT_CODE