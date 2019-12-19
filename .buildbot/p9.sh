#!/bin/bash

# Check for required env variables ONNF_DEP_DIR, LLVM_PROJECT_ROOT
if [[ -z "${ONNF_DEP_DIR}" ]]; then
  echo "ONNF_DEP_DIR env var is missing."
  exit 1
fi
if [[ -z "${LLVM_PROJECT_ROOT}" ]]; then
  echo "LLVM_PROJECT_ROOT env var is missing."
  exit 1
fi

# Set up env variables to expose onnf dependencies:
export PATH=$ONNF_DEP_DIR/bin:$PATH
export LD_LIBRARY_PATH=$ONNF_DEP_DIR/lib:$ONNF_DEP_DIR/lib64:
export CPATH=$ONNF_DEP_DIR/include:$CPATH
export PATH=$ONNF_DEP_DIR/bin:$PATH

# Set up mock installation path within current workspace:
export INSTALL_PATH=$WORKSPACE/INSTALL_PATH
mkdir -p "$INSTALL_PATH"
export PATH=$INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$INSTALL_PATH/lib:$INSTALL_PATH/lib64:$LD_LIBRARY_PATH
export CPATH=$INSTALL_PATH/include:$CPATH

# Create virtual environment specific to the current build instance:
conda create -n onnf_conda_workspace_"${BUILD_NUMBER}" python=3.7 numpy
source activate onnf_conda_workspace_"${BUILD_NUMBER}"

# Create build directory and generate make files:
mkdir build && cd build
CC=$ONNF_DEP_DIR/bin/gcc                       \
CXX=$ONNF_DEP_DIR/bin/g++                      \
BOOST_ROOT=$ONNF_DEP_DIR                       \
LLVM_SRC=$LLVM_PROJECT_ROOT/llvm               \
LLVM_BUILD=$LLVM_PROJECT_ROOT/build            \
cmake3 -DONNF_ENABLE_MODEL_TEST_CPP=ON         \
       -DONNF_ENABLE_BENCHMARK=ON              \
       -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH"  \
       ..

# Build and test:
make -j "$(nproc)" install
OMP_NUM_THREADS=20 OMP_THREAD_LIMIT=20 ctest3 -j "$(nproc)"
