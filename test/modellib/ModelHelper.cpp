/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===========-- ModelHelper.cpp - Helper function for building models -=======//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for all the models that can be built.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"

#include "include/OnnxMlirRuntime.h"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "test/modellib/ModelLib.hpp"

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

/// Build an ONNXConstantOp from an OMTensor.
ONNXConstantOp buildONNXConstantOp(MLIRContext *ctx, OpBuilder builder,
    OMTensor *omt, RankedTensorType resultType) {
  int64_t numElems = omTensorGetNumElems(omt);
  auto bufferPtr = omTensorGetDataPtr(omt);
  float *arrayPtr = reinterpret_cast<float *>(bufferPtr);
  auto array = std::vector<float>(arrayPtr, arrayPtr + numElems);
  auto denseAttr =
      DenseElementsAttr::get(resultType, llvm::makeArrayRef(array));
  ONNXConstantOp constantTensor = builder.create<ONNXConstantOp>(
      UnknownLoc::get(ctx), resultType, Attribute(), denseAttr, FloatAttr(),
      ArrayAttr(), IntegerAttr(), ArrayAttr(), StringAttr(), ArrayAttr());
  return constantTensor;
}
