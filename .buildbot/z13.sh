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

# If LLVM_PROJECT_ROOT does not exist, or llvm-project commit hash changed,
# (re)build llvm-project
if ! [ -d ${LLVM_PROJECT_ROOT} ]; then
    git clone https://github.com/llvm/llvm-project.git ${LLVM_PROJECT_ROOT}
fi
PREBUILT_LLVM_PROJECT_SHA1=$(git -C ${LLVM_PROJECT_ROOT} rev-parse HEAD)
EXPECTED_LLVM_PROJECT_SHA1=$(cat utils/clone-mlir.sh|grep -Po '(?<=git checkout )[0-9a-f]+')

echo "${LLVM_PROJECT_ROOT} sha1 prebuilt ${PREBUILT_LLVM_PROJECT_SHA1} expected ${EXPECTED_LLVM_PROJECT_SHA1}"

if [ "$1" = "shared" ]; then
    LLVM_BUILD_SHARED_LIBS=ON
else
    LLVM_BUILD_SHARED_LIBS=OFF
fi

if ! [ -d ${LLVM_PROJECT_ROOT}/build ] ||
   [ "${PREBUILT_LLVM_PROJECT_SHA1}" != "${EXPECTED_LLVM_PROJECT_SHA1}" ]; then
    echo "Rebuild llvm-project with sha1 ${EXPECTED_LLVM_PROJECT_SHA1}"
    pushd ${LLVM_PROJECT_ROOT}
    git fetch && git checkout ${EXPECTED_LLVM_PROJECT_SHA1}
    rm -rf build && mkdir -p build && cd build
    cmake -G Ninja ../llvm \
	  -DLLVM_ENABLE_PROJECTS=mlir \
	  -DLLVM_BUILD_EXAMPLES=ON \
	  -DLLVM_TARGETS_TO_BUILD="host" \
	  -DCMAKE_BUILD_TYPE=Release \
	  -DLLVM_ENABLE_ASSERTIONS=ON \
	  -DLLVM_ENABLE_RTTI=ON \
	  -DBUILD_SHARED_LIBS=${LLVM_BUILD_SHARED_LIBS}
    cmake --build . --target -- ${MAKEFLAGS}
    popd
else
    echo "Rebuild llvm-project not needed"
fi

# Build ONNX MLIR against specified llvm-project
export BUILD_PATH=build-against-$(basename ${LLVM_PROJECT_ROOT})
mkdir ${BUILD_PATH} && cd ${BUILD_PATH}

LLVM_PROJ_SRC=${LLVM_PROJECT_ROOT}              \
LLVM_PROJ_BUILD=${LLVM_PROJECT_ROOT}/build      \
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} .. \

make -j$(nproc) onnx-mlir
make -j$(nproc) check-onnx-lit
RUNTIME_DIR=$(pwd)/lib make -j$(nproc) check-onnx-backend
