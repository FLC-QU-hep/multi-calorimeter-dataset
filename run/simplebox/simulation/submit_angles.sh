#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-999
#SBATCH --job-name=sf_nlayers_angles_sim
#SBATCH --output=log/sf_nlayers_angles/sim_%A_%a.out
#SBATCH --error=log/sf_nlayers_angles/sim_%A_%a.err

# ============================================================================
# Simulation for sf_nlayers_angles configs (auto-chains batches 0-9999)
#
# Uses LCIO input with customizable incident angles.
# Updated to use SPHERICAL distribution (Cone) to strictly limit Theta to 45 deg.
#
# NOTE: The range [-45, +45] degrees is achieved by:
#       Theta (opening angle) = [0, 45] deg
#       Phi (rotation) = [-180, 180] deg
# ============================================================================

BASE_PATH="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"
WORK_DIR="${WORK_DIR:-$(dirname "$BASE_PATH")}"
DDFASTSIM_DIR="${DDFASTSIM_DIR:-$WORK_DIR/ddfastsim}"
CONFIG_DIR=sf_nlayers_angles/SiW_final
OUTPUT_DIR=sf_nlayers_angles/SiW_final
N_EVENTS=400
OFFSET=${OFFSET:-0}

# ============================================================================
# ANGLE CONFIGURATION -- Spherical (Cone)
# ============================================================================
# Theta: Polar angle from Z-axis (0 = perpendicular)
# 0.7854 rad = 45 degrees
# This ensures total deviation is NEVER more than 45 degrees.
THETA_MIN=0.0
THETA_MAX=0.7854

# Phi: Azimuthal angle.
# Rotating 360 degrees covers both "left" (-45) and "right" (+45) directions.
PHI_MIN=-3.14159
PHI_MAX=3.14159

MIN_ENERGY=1
MAX_ENERGY=100
PARTICLE_IDS="22"
POSITION="0 0 -0.00000001"
# ============================================================================

mkdir -p $BASE_PATH/log/sf_nlayers_angles
mkdir -p $BASE_PATH/output_dataset/SimpleBox/root/$OUTPUT_DIR

CONFIG_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID, Array: $SLURM_ARRAY_TASK_ID, OFFSET: $OFFSET, CONFIG_ID: $CONFIG_ID"
echo "Angles: theta=[${THETA_MIN}, ${THETA_MAX}] rad (Cone opening)"
echo "        phi=[${PHI_MIN}, ${PHI_MAX}] rad (Rotation)"
echo "=========================================="

cd "$WORK_DIR"
source "${KEY4HEP_SETUP:-/cvmfs/sw.hsf.org/key4hep/setup.sh}" -r "${KEY4HEP_RELEASE:-2025-05-29}"
source k4geo/install/bin/thisk4geo.sh
source ddfastsim/install/bin/thisDDFastSim.sh

DDFASTSIM="$DDFASTSIM_DIR/build/libDDFastSim.so $DDFASTSIM_DIR/build/libDDFastSimdetector.so"

compactFile=$BASE_PATH/calo_configs/par04/SimpleBox/$CONFIG_DIR/SimpleBox_config_$(printf "%04d" $CONFIG_ID).xml
outputFile=$BASE_PATH/output_dataset/SimpleBox/root/$OUTPUT_DIR/${N_EVENTS}_1-100GeV_SiW_xml_$(printf "%04d" $CONFIG_ID).root
slcioFile=$BASE_PATH/output_dataset/SimpleBox/root/$OUTPUT_DIR/mc_particles_$(printf "%04d" $CONFIG_ID).slcio
steering_file="$DDFASTSIM_DIR/example_scripts/SimpleBox_ddsim_steer_lcio.py"

echo "Config file: $compactFile"
echo "Output file: $outputFile"

# Step 1: Generate LCIO input with Spherical angular distribution
echo "Generating MC particles..."
python $BASE_PATH/scripts/simplebox/create_mc_particles.py \
    --particle-ids $PARTICLE_IDS \
    --position $POSITION \
    --min-energy $MIN_ENERGY \
    --max-energy $MAX_ENERGY \
    --enable-angles \
    --theta-min $THETA_MIN \
    --theta-max $THETA_MAX \
    --phi-min $PHI_MIN \
    --phi-max $PHI_MAX \
    --num-events $N_EVENTS \
    --output $slcioFile

if [ $? -ne 0 ]; then
    echo "✗ Config $CONFIG_ID: Failed to generate MC particles"
    exit 1
fi

# Step 2: Run ddsim with LCIO input
echo "Running ddsim..."
LD_PRELOAD=$DDFASTSIM ddsim \
    --steeringFile $steering_file \
    --inputFiles $slcioFile \
    --enableDetailedShowerMode \
    --outputFile $outputFile \
    --compactFile $compactFile \
    -N $N_EVENTS

EXIT_CODE=$?

# Clean up intermediate LCIO file
rm -f $slcioFile

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Config $CONFIG_ID completed"
else
    echo "✗ Config $CONFIG_ID failed"
fi

# Auto-submit next batch when task 999 completes
if [ $SLURM_ARRAY_TASK_ID -eq 999 ]; then
    NEXT_OFFSET=$((OFFSET + 1000))
    if [ $NEXT_OFFSET -lt 10000 ]; then
        echo "Auto-submitting next batch with OFFSET=$NEXT_OFFSET"
        sbatch --export=ALL,OFFSET=$NEXT_OFFSET $BASE_PATH/run/simplebox/simulation/submit_angles.sh
    else
        echo "All 10k configs completed!"
    fi
fi

exit $EXIT_CODE