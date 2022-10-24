/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ONNXOps.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include "src/Dialect/ONNX/ONNXLayoutHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/Diagnostic.hpp"
#include "src/Support/TypeUtilities.hpp"

#include <algorithm>
#include <string>

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Unsupported Operations
//===---------------------------------------------------------------------===//

// Operations for which shape inference has not been implemented yet
// If you add the implementation for one op, move it out of this section
// Also please add test case in test/mlir/onnx/onnx_shape_inference.mlir
// Followed by the implementation of lowering to Krnl and
// Enable the corresponding node test in check-onnx-backend

#define NOT_IMPLEMENTED_INFER_SHAPE(T)                                         \
  LogicalResult T::inferShapes(                                                \
      std::function<void(mlir::Region &)> doShapeInference) {                  \
    return emitError(NOT_IMPLEMENTED_MESSAGE);                                 \
  }

// Listed alphabetically.
NOT_IMPLEMENTED_INFER_SHAPE(ONNXAdagradOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXAdamOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXArrayFeatureExtractorOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXBatchNormalizationOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXBinarizerOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXCastMapOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXClipV11Op)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXClipV12Op)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXClipV6Op)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXDetOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXDictVectorizerOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXFeatureVectorizerOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXGradientOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXGridSampleOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXImputerOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXIsInfOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXLabelEncoderOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXLinearClassifierOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXLinearRegressorOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXLpPoolOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXMaxPoolOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXMaxUnpoolOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXMomentumOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXMultinomialOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXNegativeLogLikelihoodLossOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXNormalizerOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXPadV11Op)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXPadV2Op)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXRandomUniformLikeOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXRandomUniformOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXResizeV10Op)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXResizeV11Op)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXSVMClassifierOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXSVMRegressorOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXSoftmaxCrossEntropyLossOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXStringNormalizerOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXTfIdfVectorizerOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXTreeEnsembleClassifierOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXTreeEnsembleRegressorOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXUpsampleV7Op)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXUpsampleV9Op)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXZipMapOp)

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/Dialect/ONNX/ONNXOps.cpp.inc"
