/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- Version.cpp -------------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file defines several version-related utility functions.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Support/raw_ostream.h"
#include <string>

namespace onnx_mlir {
std::string getOnnxMlirFullVersion();
void getVersionPrinter(llvm::raw_ostream &os);
}
