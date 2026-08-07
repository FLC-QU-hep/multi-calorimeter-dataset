#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=cld_post
#SBATCH --output=log/fccee_cld/postprocess_%j.out
#SBATCH --error=log/fccee_cld/postprocess_%j.err

# Chain: validate → (retry if needed) → merge → symlink → cond → OT → 8 training
# Usage: sbatch --dependency=afterany:<sim_jobid> postprocess_chain.sh

set -e

SIM_BASE="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../../.." && pwd)}"
TRAIN_BASE="${ALLSHOWERS_ROOT:-$(dirname "$SIM_BASE")/AllShowers-AllGeometries}"
DET=fccee_cld
OUT=$SIM_BASE/output_dataset/FCCee_CLD
H5_DIR=$OUT/h5_1M
N_JOBS=150
N_EVENTS=1000
SELF=$SIM_BASE/run/lemurs/$DET/simulation/postprocess_chain.sh
SIM_SCRIPT=$SIM_BASE/run/lemurs/$DET/simulation/submit_inline.sh

unset PYTHONPATH PYTHONHOME
export PATH=/usr/local/bin:/usr/bin:/bin
module load maxwell mamba 2>/dev/null
. mamba-init 2>/dev/null
conda activate calo-transfer 2>/dev/null

# ============================================================
# Step 0: Validate all H5 files (exist + correct event count)
# ============================================================
echo "=== Validation ==="
MISSING=$(python3 -c "
import h5py, os
missing = []
for i in range($N_JOBS):
    path = f'$H5_DIR/${DET}_{i:05d}.h5'
    if not os.path.exists(path):
        missing.append(i); continue
    try:
        with h5py.File(path, 'r') as f:
            n = f['energies'].shape[0]
            if n < $N_EVENTS * 0.95:  # accept >=950 (some showers have no hits)
                missing.append(i)
    except:
        missing.append(i)
print(','.join(map(str, missing)))
")

if [ -n "$MISSING" ]; then
    N_MISS=$(echo "$MISSING" | tr ',' '\n' | wc -l)
    echo "MISSING/BAD: $N_MISS tasks → [$MISSING]"

    # Resubmit only missing tasks
    RETRY_JOB=$(sbatch --parsable --array=$MISSING $SIM_SCRIPT)
    echo "Retry job: $RETRY_JOB"

    # Resubmit self after retry
    SELF_JOB=$(sbatch --parsable --dependency=afterany:$RETRY_JOB $SELF)
    echo "Re-queued postprocess: $SELF_JOB"
    exit 0
fi

echo "All $N_JOBS files OK ($((N_JOBS * N_EVENTS)) events)"

# ============================================================
# Step 1: Merge H5
# ============================================================
echo "=== Merge H5 ==="
cd $SIM_BASE
python3 scripts/lemurs/merge_h5_lemurs.py --step merge-detector --detector $DET

echo "=== Rename → 150k ==="
mv $OUT/${DET}_1M.h5 $OUT/${DET}_150k.h5
echo "Merged: $(ls -lh $OUT/${DET}_150k.h5)"

# ============================================================
# Step 2: Symlinks
# ============================================================
echo "=== Symlinks ==="
cd $TRAIN_BASE/data/finetune_cld
rm -f cld_1M.h5 cld_110k.h5
ln -s $OUT/${DET}_150k.h5 cld_1M.h5
ln -s cld_1M.h5 cld_110k.h5
ls -la cld_1M.h5 cld_110k.h5

# ============================================================
# Step 3: Conditioning file (10k validation events)
# ============================================================
echo "=== cld_10k_cond.h5 ==="
python3 -c "
import h5py, numpy as np

src = '$TRAIN_BASE/data/finetune_cld/cld_1M.h5'
out = '$TRAIN_BASE/data/finetune_cld/cld_10k_cond.h5'
N, NL = 10000, 45

with h5py.File(src, 'r') as f:
    total = f['energies'].shape[0]
    START = total - N
    energies = f['energies'][START:START+N]
    directions = f['directions'][START:START+N]
    sf = f['sampling_fraction'][START:START+N]
    nl = f['num_layers'][START:START+N]
    pdg = f['pdg'][START:START+N]
    lzp = f['layer_z_pos'][START:START+N]

    npl = np.zeros((N, NL), dtype=np.int32)
    for i in range(N):
        pts = f['showers'][START + i].reshape(-1, 5)
        li = (pts[:, 2] + 0.1).astype(np.int32)
        np.add.at(npl[i], li[li < NL], 1)

with h5py.File(out, 'w') as f:
    f.create_dataset('energies', data=energies)
    f.create_dataset('directions', data=directions)
    f.create_dataset('sampling_fraction', data=sf)
    f.create_dataset('num_layers', data=nl)
    f.create_dataset('pdg', data=pdg)
    f.create_dataset('layer_z_pos', data=lzp[:, :NL])
    f.create_dataset('num_points_per_layer', data=npl)

print(f'OK: {N} events, npl sum range [{npl.sum(1).min()}, {npl.sum(1).max()}]')
"

# ============================================================
# Step 4: Submit OT (auto-chains to 8 training jobs)
# ============================================================
echo "=== Submit OT ==="
cd $TRAIN_BASE
OT_JOB=$(sbatch --parsable run/training/ot_match_cld.sh)
echo "OT job: $OT_JOB (auto-chains to 8 training)"
echo "=== Pipeline complete ==="
