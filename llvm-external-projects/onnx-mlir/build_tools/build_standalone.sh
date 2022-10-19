#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Simple script that does a CMake configure of this project as an external
# LLVM project so it can be tested in isolation to larger assemblies.
# This is meant for CI's and project maintainers.

set -eu -o errtrace

build_dir=$1
llvm_project_dir=$2
project_dir="$(cd $(dirname $0)/.. && pwd)"


cmake -GNinja -B"$build_dir" -S"$llvm_project_dir/llvm" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=onnx-mlir-dialects \
  -DLLVM_EXTERNAL_ONNX_MLIR_DIALECTS_SOURCE_DIR="$project_dir" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_CCACHE_BUILD=ON \
	-DCMAKE_C_COMPILER=clang-15 -DCMAKE_CXX_COMPILER=clang++-15 -DLLVM_ENABLE_LLD=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cd "$build_dir"
ninja tools/onnx-mlir-dialects/all
