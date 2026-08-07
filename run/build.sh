WORK_DIR="${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
cd "$WORK_DIR"

source "${KEY4HEP_SETUP:-/cvmfs/sw.hsf.org/key4hep/setup.sh}" -r "${KEY4HEP_RELEASE:-2025-05-29}"
cd k4geo
mkdir build
cd build
cmake -DBoost_NO_BOOST_CMAKE=ON -S .. -B . -DCMAKE_INSTALL_PREFIX=../install
make -j 16 install
source ../install/bin/thisk4geo.sh
cd ../../ddfastsim
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
make -j 4 install
source ../install/bin/thisDDFastSim.sh
cd ../..