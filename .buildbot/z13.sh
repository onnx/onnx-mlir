#!/bin/bash

# Exit on error:
set -e

# Check for required env variables JAVA_HOME
if [[ -z "${JAVA_HOME}" ]]; then
  echo "JAVA_HOME env var is missing."
  exit 1
fi

if [[ -z "${LLVM_PROJECT_ROOT}" ]]; then
  echo "LLVM_PROJECT_ROOT env var is missing."
  exit 1
fi

# Set up mock installation path:
export INSTALL_PATH=${WORKSPACE}/INSTALL_PATH
mkdir -p ${INSTALL_PATH}
export PATH=${INSTALL_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${INSTALL_PATH}/lib:${INSTALL_PATH}/lib64:${LD_LIBRARY_PATH}
export CPATH=${INSTALL_PATH}/include:${CPATH}

# Set up project specific environment variables:
export PATH=${JAVA_HOME}/bin:${PATH}
export CLASSPATH=.:${JAVA_HOME}/lib:${JAVA_HOME}/lib/tools.jar:${CLASSPATH}

export BUILD_PATH=build-against-$(basename ${LLVM_PROJECT_ROOT})
mkdir ${BUILD_PATH} && cd ${BUILD_PATH}

LLVM_PROJ_SRC=${LLVM_PROJECT_ROOT}              \
LLVM_PROJ_BUILD=${LLVM_PROJECT_ROOT}/build      \
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} .. \

make -j$(nproc) onnx-mlir
make -j$(nproc) check-onnx-lit
make -j$(nproc) check-onnx-backend
