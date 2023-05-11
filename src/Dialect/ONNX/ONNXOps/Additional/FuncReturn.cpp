/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- FuncReturn.cpp - ONNX Operations -------------------===//
//
// This file provides definition of ONNX dialect FuncReturn operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXFuncReturnOp::verify() {
  // TODO: Implement this.
  return success();
}
