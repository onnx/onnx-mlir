/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------------- OMOptions.cpp ----------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Options that provide fine control on passes.
//
//===----------------------------------------------------------------------===//

#include "src/Support/OMOptions.hpp"

namespace onnx_mlir {
llvm::cl::OptionCategory OMPassOptions("ONNX-MLIR Pass Options",
    "These are options to provide fine control on passes");
}
