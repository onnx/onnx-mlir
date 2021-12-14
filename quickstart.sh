#!/bin/bash 
# This script uses scripts in util to quickly build mlir and onnx-mlir in the workding directory.
# the workding directory is always the parent directory of onnx-mlir.
# This script is designed to be used interactively.
#
# Common issues include:
#   - protobuf compiler not found: run `sudo apt install -y protobuf-compiler`
#   - curses not found: run `sudo apt install -y libncurses-dev`
#   - xxx killed: this may be caused by insufficient memory; run `CMAKE_BUILD_PARALLEL_LEVEL=2 bash onnx-mlir/quickstart.sh` instead to limit threads to 2 and reduce memory use
set -e # exit on error.

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
ONNX_MLIR_REPO=$(dirname "$SCRIPT")
echo $ONNX_MLIR_REPO

##### Install MLIR and ONNX-MLIR
pushd $ONNX_MLIR_REPO/..
source $ONNX_MLIR_REPO/utils/install-mlir.sh
popd

pushd $ONNX_MLIR_REPO/..
source $ONNX_MLIR_REPO/utils/install-onnx-mlir.sh
popd

read -p "onnx-mlir has been built. Press enter to continue to demos, ctrl+C to quit"

#### run demos in original working directory

#### download mnist from onnx model zoo
wget https://github.com/onnx/models/raw/master/vision/classification/mnist/model/mnist-8.tar.gz
tar xvf mnist-8.tar.gz
cd mnist

# build model
$ONNX_MLIR_REPO/build/Debug/bin/onnx-mlir --EmitLib ./model.onnx

# run with run
pushd $ONNX_MLIR_REPO/utils
source build-run-onnx-lib.sh
popd
$ONNX_MLIR_REPO/build/Debug/bin/run-onnx-lib ./model.so






