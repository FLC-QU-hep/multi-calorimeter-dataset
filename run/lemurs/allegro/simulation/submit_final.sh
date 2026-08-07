#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-999
#SBATCH --job-name=allegro_final_sim
#SBATCH --output=log/allegro_final/sim_%A_%a.out
#SBATCH --error=log/allegro_final/sim_%A_%a.err

# ============================================================================
# ALLEGRO ECal — LEMURS resimulation, large-scale final run
# 1000 batches x 1000 events = 1M showers
# ROOT only — no HDF5 (preprocessing comes later)
# ============================================================================

BASE_PATH="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../../.." && pwd)}"
WORK_DIR="$(dirname "$BASE_PATH")"

ROOT_OUTPUT_DIR=$BASE_PATH/output_dataset/ALLEGRO/root/final
LOG_DIR=$BASE_PATH/log/allegro_final

mkdir -p $ROOT_OUTPUT_DIR $LOG_DIR

BATCH_ID=$SLURM_ARRAY_TASK_ID
N_EVENTS=${N_EVENTS:-1000}

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID, Array: $BATCH_ID"
echo "ALLEGRO ECal — LEMURS resimulation (ROOT only)"
echo "N_EVENTS: $N_EVENTS"
echo "=========================================="

# --- Environment ---
cd $WORK_DIR
source "${KEY4HEP_SETUP:-/cvmfs/sw.hsf.org/key4hep/setup.sh}" -r "${KEY4HEP_RELEASE:-2025-05-29}"
source ddfastsim/install/bin/thisDDFastSim.sh
# NOTE: do NOT source thisk4geo.sh — local k4geo plugins conflict with cvmfs XML

DDFASTSIM="$WORK_DIR/ddfastsim/build/libDDFastSim.so $WORK_DIR/ddfastsim/build/libDDFastSimdetector.so"
export DDFASTSIM_OPTIONS=$WORK_DIR/ddfastsim/options
export ALLEGRO_COMPACT=/cvmfs/sw.hsf.org/key4hep/releases/2025-05-29/x86_64-almalinux9-gcc14.2.0-opt/k4geo/00-22-ubhvqv/share/k4geo/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml

STEERING=$WORK_DIR/ddfastsim/options/FCCeeALLEGRO/FCCeeALLEGRO_ddsim_steer_steps_lcio.py
SLCIO=$ROOT_OUTPUT_DIR/allegro_final_$(printf "%04d" $BATCH_ID).slcio
OUTPUT=$ROOT_OUTPUT_DIR/allegro_final_$(printf "%04d" $BATCH_ID).root

echo "Steering : $STEERING"
echo "LCIO     : $SLCIO"
echo "Output   : $OUTPUT"
echo "=========================================="

# --- Step 1: Generate LCIO input ---
echo "Generating MC particles (ALLEGRO barrel surface, LEMURS angles)..."
python3 $BASE_PATH/scripts/lemurs/create_mc_particles.py \
    --detector fccee_allegro \
    --num-events $N_EVENTS \
    --output $SLCIO

if [ $? -ne 0 ]; then
    echo "FAILED: MC particle generation"
    exit 1
fi

# --- Step 2: Run ddsim ---
echo "Running ddsim..."

LD_PRELOAD=$DDFASTSIM ddsim \
    --steeringFile $STEERING \
    --inputFiles   $SLCIO \
    --outputFile   $OUTPUT \
    -N             $N_EVENTS

EXIT_CODE=$?
rm -f $SLCIO

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Batch $BATCH_ID completed successfully → $OUTPUT"
else
    echo "Batch $BATCH_ID FAILED (exit code $EXIT_CODE)"
fi
echo "=========================================="

exit $EXIT_CODE