/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ONNXTypes.cpp - ONNX Operations -------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of utility functions for  ONNX Types.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/Dialect/ONNX/ONNXTypes.hpp"

using namespace mlir;

// ONNXTyps.cpp.inc is NOT included here, but in ONNXOps.cpp.
// The reason is that the functions for type print/parse is used
// in dialect initialization. If included here, compilation warning
// for unsed symbol will occur.

// This file is for  utility functions for type definition if there is any.
