#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=siw_postproc
#SBATCH --output=log/par04_siw_100gev/postprocess_%j.out
#SBATCH --error=log/par04_siw_100gev/postprocess_%j.err

# ============================================================================
# Par04_SiW 150k post-processing: merge → AllShowers finetune → PointCountFM
# ============================================================================

BASE_PATH="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../../.." && pwd)}"
WORK_DIR="$(dirname "$BASE_PATH")"
ALLSHOWERS=$WORK_DIR/AllShowers-AllGeometries/data
PCFM=$WORK_DIR/PointCountFM/data

NUM_LAYERS=90
DET=par04_siw
DET_DIR=Par04_SiW

module load maxwell mamba 2>/dev/null
. mamba-init 2>/dev/null
conda activate calo-transfer 2>/dev/null

cd $BASE_PATH

# --- Step 1: Merge 1500 h5 files ---
echo "=== Step 1: Merging h5 files ==="
python3 scripts/lemurs/merge_h5_lemurs.py \
    --step merge-detector \
    --detector $DET \
    --h5-subdir h5_150k_100gev \
    --output-name ${DET}_150k.h5

MERGED=$BASE_PATH/output_dataset/$DET_DIR/${DET}_150k.h5
if [ ! -f "$MERGED" ]; then
    echo "FAILED: merge did not produce $MERGED"
    exit 1
fi
echo "Merged: $(ls -lh $MERGED | awk '{print $5}')"

# --- Step 2: Copy to AllShowers as finetune file ---
echo "=== Step 2: AllShowers finetune files ==="
mkdir -p $ALLSHOWERS/finetune_par04_siw
cp $MERGED $ALLSHOWERS/finetune_par04_siw/${DET}_110k.h5
echo "Copied to $ALLSHOWERS/finetune_par04_siw/${DET}_110k.h5"

# --- Step 3: Create conditioning file (10k) ---
echo "=== Step 3: Creating ${DET}_10k_cond.h5 ==="
python3 scripts/lemurs/create_cond.py \
    --detector $DET \
    --src $MERGED \
    --n 10000 \
    --output finetune_par04_siw/${DET}_10k_cond.h5

if [ ! -f "$ALLSHOWERS/finetune_par04_siw/${DET}_10k_cond.h5" ]; then
    echo "FAILED: conditioning file not created"
    exit 2
fi
echo "Created: $(ls -lh $ALLSHOWERS/finetune_par04_siw/${DET}_10k_cond.h5 | awk '{print $5}')"

# --- Step 4: Create PointCountFM training file ---
echo "=== Step 4: Creating PointCountFM ${DET}_finetune_150k.h5 ==="
python3 -c "
import h5py, numpy as np
from tqdm import tqdm

NUM_LAYERS = $NUM_LAYERS
src = '$MERGED'
dst = '$PCFM/${DET}_finetune_150k.h5'

with h5py.File(src, 'r') as f:
    n = f['energies'].shape[0]
    print(f'Processing {n} showers...')
    energy = f['energies'][:].astype(np.float32)
    sf = f['sampling_fraction'][:].astype(np.float32)
    nl = f['num_layers'][:].astype(np.int32)
    dirs = f['directions'][:].astype(np.float32)
    n_cols = int(f['shape'][2])

    num_points = np.zeros((n, NUM_LAYERS), dtype=np.int32)
    for i in tqdm(range(n), desc='Computing num_points_per_layer'):
        raw = f['showers'][i]
        if len(raw) == 0:
            continue
        nhits = len(raw) // n_cols
        layers = np.clip(raw.reshape(nhits, n_cols)[:, 2].astype(int), 0, NUM_LAYERS - 1)
        for l in layers:
            num_points[i, l] += 1

with h5py.File(dst, 'w') as f:
    f.create_dataset('energy', data=energy, compression='gzip')
    f.create_dataset('num_points', data=num_points, compression='gzip')
    f.create_dataset('sampling_fraction', data=sf, compression='gzip')
    f.create_dataset('n_layers', data=nl, compression='gzip')
    f.create_dataset('directions', data=dirs, compression='gzip')
    f.attrs['n_showers'] = n
    f.attrs['max_layers'] = NUM_LAYERS

print(f'Saved: {dst}')
print(f'  energy: {energy.shape}, num_points: {num_points.shape}')
print(f'  E range: [{energy.min():.2f}, {energy.max():.2f}] GeV')
"

if [ ! -f "$PCFM/${DET}_finetune_150k.h5" ]; then
    echo "FAILED: PointCountFM training file not created"
    exit 3
fi
echo "Created: $(ls -lh $PCFM/${DET}_finetune_150k.h5 | awk '{print $5}')"

echo "=========================================="
echo "All post-processing complete!"
echo "  Merged:       $MERGED"
echo "  AllShowers:   $ALLSHOWERS/finetune_par04_siw/${DET}_110k.h5"
echo "  Cond:         $ALLSHOWERS/finetune_par04_siw/${DET}_10k_cond.h5"
echo "  PointCountFM: $PCFM/${DET}_finetune_150k.h5"
echo "=========================================="
