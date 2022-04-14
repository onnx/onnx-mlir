/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------------- OMOptions.hpp ----------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Options that provide fine control on passes.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Support/CommandLine.h"

namespace onnx_mlir {
extern llvm::cl::OptionCategory OMPassOptions;
}
