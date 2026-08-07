#!/bin/bash
#SBATCH --job-name=multi_h5_creation_angles
#SBATCH --partition=allcpu
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=64
#SBATCH --output=log/h5_creation/multi_dataset_angles_%j.out
#SBATCH --error=log/h5_creation/multi_dataset_angles_%j.err

# ============================================================================
# Multi-Dataset H5 Creation Script for sf_nlayers_angles
# ============================================================================

BASE_PATH="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"
WORK_DIR="${WORK_DIR:-$(dirname "$BASE_PATH")}"
ALLSHOWERS_ROOT="${ALLSHOWERS_ROOT:-$WORK_DIR/AllShowers-AllGeometries}"

echo "=========================================="
echo "Starting Multi-Dataset H5 Creation (angles)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
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
echo "Environment:"
echo "  Working directory: $(pwd)"
echo "  Python: $(which python)"
echo "  Conda environment: $CONDA_DEFAULT_ENV"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "=========================================="
echo ""

echo ">>> [1/2] Steps 1+2: ROOT -> intermediate H5 (calo-transfer env)..."
python scripts/simplebox/multi_dataset_h5_creation.py \
    --ref-dir sf_nlayers_angles \
    --n-showers 400 \
    --output-name 4Mshowers_angles.h5

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "✗ FAILED: Steps 1+2 failed (exit $EXIT_CODE)"
    exit $EXIT_CODE
fi
echo "✓ Steps 1+2 done"

# --- Phase 2: convert to showerdata format (AllShowers venv has showerdata) ---
echo ""
echo ">>> [2/2] Step 3: intermediate H5 -> showerdata format (AllShowers venv)..."
conda deactivate

ALLSHOWERS=$ALLSHOWERS_ROOT
source ${ALLSHOWERS}/.venv/bin/activate

TEMP="$BASE_PATH/output_dataset/SimpleBox/h5/sf_nlayers_angles/SiW_final/temp"
DST="$BASE_PATH/output_dataset/SimpleBox/showerdata/sf_nlayers_angles/SiW_final/4Mshowers_angles.h5"

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
echo "Multi-Dataset H5 Creation Completed"
echo "=========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: training-ready dataset at:"
    echo "  $DST"
else
    echo "✗ FAILED: Step 3 failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE