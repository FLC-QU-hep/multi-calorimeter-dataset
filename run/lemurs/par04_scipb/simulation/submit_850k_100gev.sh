#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-999
#SBATCH --job-name=scipb_850k
#SBATCH --output=log/scipb_850k/inline_%A_%a.out
#SBATCH --error=log/scipb_850k/inline_%A_%a.err

BASE_PATH="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../../.." && pwd)}"
WORK_DIR="$(dirname "$BASE_PATH")"

OFFSET=${OFFSET:-0}
BATCH_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))
N_EVENTS=100
DET=par04_scipb
DET_DIR=Par04_SciPb

H5_OUTPUT_DIR=$BASE_PATH/output_dataset/$DET_DIR/h5_850k_100gev
TMP_DIR=$BASE_PATH/output_dataset/$DET_DIR/tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $H5_OUTPUT_DIR $TMP_DIR

PADDED=$(printf "%05d" $BATCH_ID)

echo "=========================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID, OFFSET: $OFFSET, BATCH_ID: $BATCH_ID"
echo "$DET inline sim+H5 | N_EVENTS=$N_EVENTS | Energy: 1-100 GeV"
echo "=========================================="

cd $WORK_DIR
source "${KEY4HEP_SETUP:-/cvmfs/sw.hsf.org/key4hep/setup.sh}" -r "${KEY4HEP_RELEASE:-2025-05-29}"
source k4geo/install/bin/thisk4geo.sh
source ddfastsim/install/bin/thisDDFastSim.sh

DDFASTSIM="$WORK_DIR/ddfastsim/build/libDDFastSim.so $WORK_DIR/ddfastsim/build/libDDFastSimdetector.so"
export DDFASTSIM_INSTALL=$WORK_DIR/ddfastsim/install
export PAR04_COMPACT=$WORK_DIR/ddfastsim/install/Detector/xml/Par04_SciPb_fullsim.xml

STEERING=$WORK_DIR/ddfastsim/options/Par04_SciPb/Par04_ddsim_steer_steps_lcio.py
SLCIO=$TMP_DIR/${DET}_${PADDED}.slcio
ROOT_FILE=$TMP_DIR/${DET}_${PADDED}.root

echo "Step 1: Generating MC particles (1-100 GeV)..."
python3 $BASE_PATH/scripts/lemurs/create_mc_particles.py \
    --detector $DET \
    --min-energy 1 \
    --max-energy 100 \
    --num-events $N_EVENTS \
    --output $SLCIO

if [ $? -ne 0 ]; then
    echo "FAILED: MC particle generation"
    rm -rf $TMP_DIR
    exit 1
fi

echo "Step 2: Running ddsim..."
LD_PRELOAD=$DDFASTSIM ddsim \
    --steeringFile $STEERING \
    --inputFiles   $SLCIO \
    --outputFile   $ROOT_FILE \
    -N             $N_EVENTS

if [ $? -ne 0 ]; then
    echo "FAILED: ddsim"
    rm -rf $TMP_DIR
    exit 2
fi

rm -f $SLCIO
echo "ROOT file: $(ls -lh $ROOT_FILE | awk '{print $5}')"

echo "Step 3: Converting ROOT → H5..."
unset PYTHONPATH PYTHONHOME
export PATH=/usr/local/bin:/usr/bin:/bin
module load maxwell mamba 2>/dev/null
. mamba-init 2>/dev/null
conda activate calo-transfer 2>/dev/null

cd $BASE_PATH
python3 scripts/lemurs/process_root_to_h5.py \
    --detector $DET \
    --root-file $ROOT_FILE \
    --output-dir $H5_OUTPUT_DIR

H5_EXIT=$?

if [ $H5_EXIT -eq 0 ]; then
    rm -rf $TMP_DIR
    H5_FILE=$H5_OUTPUT_DIR/${DET}_${PADDED}.h5
    echo "OK: $(ls -lh $H5_FILE 2>/dev/null | awk '{print $5}')"
else
    echo "H5 FAILED — keeping ROOT: $ROOT_FILE"
fi

echo "Batch $BATCH_ID done (exit: $H5_EXIT)"
exit $H5_EXIT
