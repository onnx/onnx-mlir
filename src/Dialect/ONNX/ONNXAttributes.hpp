/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- ONNXAttributes.hpp -----------------------------===//
//
// This file defines attributes in the ONNX Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ONNX_ATTRIBUTES_H
#define ONNX_MLIR_ONNX_ATTRIBUTES_H

#include "mlir/IR/Attributes.h"

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/ONNX/ONNXAttributes.hpp.inc"
#endif
