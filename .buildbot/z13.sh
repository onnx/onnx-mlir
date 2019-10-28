#!/bin/bash

# Check for required env variables JAVA_HOME
if [[ -z "${JAVA_HOME}" ]]; then
  echo "JAVA_HOME env var is missing."
  exit 1
fi

# Set up mock installation path:
export INSTALL_PATH=$WORKSPACE/INSTALL_PATH
mkdir -p $INSTALL_PATH
export PATH=$INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$INSTALL_PATH/lib:$INSTALL_PATH/lib64:$LD_LIBRARY_PATH
export CPATH=$INSTALL_PATH/include:$CPATH

# Set up project specific environment variables:
export PATH=$JAVA_HOME/bin:$PATH
export CLASSPATH=.:$JAVA_HOME/lib:$JAVA_HOME/lib/tools.jar:$CLASSPATH
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export BUILD_PATH=$(pwd)
export CPATH=$(pwd)/../runtime/

mkdir build && cd build
cmake -DCMAKE_C_COMPILER=$CC \
      -DCMAKE_CXX_COMPILER=$CXX \
      -DCMAKE_VERBOSE_MAKEFILE=ON \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH \
      -DDLC_ENABLE_NODE_TEST_JAVA=ON \
      -DDLC_ENABLE_NODE_TEST_JNI=ON \
      -DDLC_ENABLE_NODE_TEST_CPP=OFF \
      -DDLC_TARGET_ARCH=z13 ..

make -j "$(nproc)" install
ctest -j "$(nproc)"
