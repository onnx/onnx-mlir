/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------ONNXShapeHelper.cpp - help for shapes----------------=== //
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file has the computations to compute the shapes using the new index expr
// approach.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

#include <algorithm>

#define DEBUG_TYPE "shape-helper"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper
//===----------------------------------------------------------------------===//

/// Refine `inferredDims` using the output's shape if possible. For example,
/// replacing a dynamic dim in `inferredDims` by a static dim in the output's
/// shape.
static void refineDims(DimsExpr &inferredDims, Value output) {
  // Nothing to do if the output is unranked.
  if (!isRankedShapedType(output.getType()))
    return;

  llvm::ArrayRef<int64_t> existingDims = getShape(output.getType());
  // Do not handle the case of scalar tensor whose type can be tensor<f32>
  // or tensor<1xf32>. Just use the inferredShape in this case.
  if (existingDims.size() < 1 || inferredDims.size() < 1)
    return;

  assert((existingDims.size() == inferredDims.size()) &&
         "Inferred shape and existing shape are inconsistent in the number "
         "of elements");

  // Try to update inferredDim if existingDim is static.
  for (unsigned i = 0; i < existingDims.size(); ++i) {
    // existingDim is dynamic, nothing to do.
    if (ShapedType::isDynamic(existingDims[i]))
      continue;

    // inferredDim is unknown at shape inference: update it.
    if (inferredDims[i].isQuestionmark()) {
      inferredDims[i] = LiteralIndexExpr(existingDims[i]);
      continue;
    }
    // inferredDim is unknown at lowering: use existing dim for efficiency.
    if (!inferredDims[i].isLiteral()) {
      inferredDims[i] = LiteralIndexExpr(existingDims[i]);
      continue;
    }
    // inferredDim is different from existingDim. Believe in existingDim.
    if (inferredDims[i].isLiteral() &&
        (existingDims[i] != inferredDims[i].getLiteral())) {
      // Warning for users.
      llvm::outs() << "Warning: [Shape inference] the inferred dim ("
                   << inferredDims[i].getLiteral()
                   << ") is different from the existing dim ("
                   << existingDims[i] << "). Use the existing dim instead.\n";
      inferredDims[i] = LiteralIndexExpr(existingDims[i]);
    }
  }
}

// Reuse scope if given, otherwise create one now and free in destructor.
template <class OP>
ONNXOpShapeHelper<OP>::ONNXOpShapeHelper(
    OP *newOp, int numResults, IndexExprScope *inScope)
    : op(newOp), fGetDenseVal(getDenseElementAttributeFromONNXValue),
      fLoadVal(nullptr), outputsDims(), ownScope(inScope == nullptr) {
  assert(op && "Expecting a valid pointer");
  if (ownScope)
    scope = new IndexExprScope(nullptr, newOp->getLoc());
  setNumberOfOutputs(numResults);
}

template <class OP>
ONNXOpShapeHelper<OP>::ONNXOpShapeHelper(OP *newOp, int numResults,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseValInput,
    ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope)
    : op(newOp), fLoadVal(fLoadVal), outputsDims(),
      ownScope(inScope == nullptr) {
  assert(op && "Expecting a valid pointer");
  if (ownScope)
    scope = new IndexExprScope(rewriter, newOp->getLoc());
  setNumberOfOutputs(numResults);
  // Get the dense value by combining provided function (if any) with the
  // default one.
  fGetDenseVal = [=](Value array) {
    DenseElementsAttr res = nullptr;
    // Try with the provided method, if any.
    if (fGetDenseValInput)
      res = fGetDenseValInput(array);
    // If provided method was not provided or failed, try default ONNX method.
    if (!res)
      res = getDenseElementAttributeFromONNXValue(array);
    return res;
  };
}

template <>
Value ONNXOpShapeHelper<Operation>::getOutput(int n) {
  return op->getResult(n);
}

// Set output dims for the N-th output.
template <class OP>
void ONNXOpShapeHelper<OP>::setOutputDims(DimsExpr inferredDims, int n) {
  outputsDims[n] = inferredDims;
  // Try to refine outputsDims[n] using the output's shape if possible. For
  // example, replacing a dynamic dim in outputsDims[n] by a static dim in the
  // output's shape.
  Value output = getOutput(n);
  refineDims(outputsDims[n], output);
}

/// Update a tensor type by using the given shape, elementType and encoding.
void updateType(Value val, ArrayRef<int64_t> shape, Type elementType,
    Attribute encoding, bool refineShape) {
  // Try to combine the given shape and the output's shape if possible.
  IndexExprScope scope(nullptr, val.getLoc());
  DimsExpr inferredDims;
  for (int64_t d : shape) {
    if (ShapedType::isDynamic(d))
      inferredDims.emplace_back(QuestionmarkIndexExpr());
    else
      inferredDims.emplace_back(LiteralIndexExpr(d));
  }
  if (refineShape)
    refineDims(inferredDims, val);
  SmallVector<int64_t, 4> inferredShape;
  IndexExpr::getShape(inferredDims, inferredShape);

  // Get element type.
  if (!elementType)
    elementType = getElementType(val.getType());

  // Get encoding.
  if (auto valType = val.getType().dyn_cast<RankedTensorType>())
    if (!encoding)
      encoding = valType.getEncoding();

  // Build result type.
  RankedTensorType resType;
  if (encoding)
    resType = RankedTensorType::get(inferredShape, elementType, encoding);
  else
    resType = RankedTensorType::get(inferredShape, elementType);

  val.setType(resType);
}

//===----------------------------------------------------------------------===//
// ONNX Shape Helper template instantiation
// Keep template instantiation at the end of the file.
//===----------------------------------------------------------------------===//

template struct ONNXOpShapeHelper<ONNXArgMaxOp>;
template struct ONNXOpShapeHelper<ONNXArgMinOp>;
template struct ONNXOpShapeHelper<ONNXAveragePoolOp>;
template struct ONNXOpShapeHelper<ONNXCategoryMapperOp>;
template struct ONNXOpShapeHelper<ONNXClipOp>;
template struct ONNXOpShapeHelper<ONNXCompressOp>;
template struct ONNXOpShapeHelper<ONNXConcatOp>;
template struct ONNXOpShapeHelper<ONNXConcatShapeTransposeOp>;
template struct ONNXOpShapeHelper<ONNXConvOp>;
template struct ONNXOpShapeHelper<ONNXDFTOp>;
template struct ONNXOpShapeHelper<ONNXDepthToSpaceOp>;
template struct ONNXOpShapeHelper<ONNXExpandOp>;
template struct ONNXOpShapeHelper<ONNXFlattenOp>;
template struct ONNXOpShapeHelper<ONNXGatherOp>;
template struct ONNXOpShapeHelper<ONNXGatherElementsOp>;
template struct ONNXOpShapeHelper<ONNXGatherNDOp>;
template struct ONNXOpShapeHelper<ONNXGemmOp>;
template struct ONNXOpShapeHelper<ONNXQLinearMatMulOp>;
template struct ONNXOpShapeHelper<ONNXMatMulIntegerOp>;
template struct ONNXOpShapeHelper<ONNXMatMulOp>;
template struct ONNXOpShapeHelper<ONNXMaxPoolSingleOutOp>;
template struct ONNXOpShapeHelper<ONNXOneHotOp>;
template struct ONNXOpShapeHelper<ONNXPadOp>;
template struct ONNXOpShapeHelper<ONNXReduceSumOp>;
template struct ONNXOpShapeHelper<ONNXReshapeOp>;
template struct ONNXOpShapeHelper<ONNXLRNOp>;
template struct ONNXOpShapeHelper<ONNXReverseSequenceOp>;
template struct ONNXOpShapeHelper<ONNXRoiAlignOp>;
template struct ONNXOpShapeHelper<ONNXShapeOp>;
template struct ONNXOpShapeHelper<ONNXSliceOp>;
template struct ONNXOpShapeHelper<ONNXSpaceToDepthOp>;
template struct ONNXOpShapeHelper<ONNXSplitOp>;
template struct ONNXOpShapeHelper<ONNXSplitV11Op>;
template struct ONNXOpShapeHelper<ONNXSqueezeOp>;
template struct ONNXOpShapeHelper<ONNXSqueezeV11Op>;
template struct ONNXOpShapeHelper<ONNXTileOp>;
template struct ONNXOpShapeHelper<ONNXTopKOp>;
template struct ONNXOpShapeHelper<ONNXTransposeOp>;
template struct ONNXOpShapeHelper<ONNXUnsqueezeOp>;
template struct ONNXOpShapeHelper<ONNXUnsqueezeV11Op>;

#if DEPRECATED
template struct ONNXOpBroadcastedShapeHelper<Operation>;
template struct ONNXOpBroadcastedShapeHelper<ONNXExpandOp>;

template struct ONNXGenericPoolShapeHelper<ONNXAveragePoolOp,
    ONNXAveragePoolOpAdaptor>;
template struct ONNXGenericPoolShapeHelper<ONNXConvOp, ONNXConvOpAdaptor>;
template struct ONNXGenericPoolShapeHelper<ONNXMaxPoolSingleOutOp,
    ONNXMaxPoolSingleOutOpAdaptor>;
#endif

// Keep template instantiation at the end of the file.

} // namespace onnx_mlir
