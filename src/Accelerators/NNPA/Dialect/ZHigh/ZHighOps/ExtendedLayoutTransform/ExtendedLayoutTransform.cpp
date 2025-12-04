/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ExtendedLayoutTransform.cpp - ZHigh Operations
//------------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ShapeHelper.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// Verifier: encapsulate all the current constraints/limitations.
//===----------------------------------------------------------------------===//

LogicalResult ZHighExtendedLayoutTransformOp::verify() {
  ZHighExtendedLayoutTransformOpAdaptor operandAdaptor =
      ZHighExtendedLayoutTransformOpAdaptor(*this);
  Value source = operandAdaptor.getSource();
  if (!hasShapeAndRank(source))
    return success();
  ShapedType sourceType = mlir::cast<ShapedType>(source.getType());
  auto sourceShape = sourceType.getShape();
  int64_t sourceRank = sourceType.getRank();

  // First constraint; innermost dim is a literal and mod 64. This constraint
  // can be lifted but is kept at this time because it results in simple and
  // efficient code; pattern is seen in key benchmarks.
  if (sourceShape[sourceRank - 1] == ShapedType::kDynamic)
    return emitOpError("Support only compiler constant innermost dim");
  if (sourceShape[sourceRank - 1] % 64 != 0)
    return emitOpError("Support only innermost dim that are multiple of 64");

  // Second constraints: we should be able to handle z layout in software.
  if (isZTensor(sourceType) && !supportedLayoutForCompilerGeneratedStickUnstick(
                                   source, /*nhwc*/ false)) {
    return emitOpError("Support only a subset of zLayouts for source value");
  }
  auto outputLayout = getTargetLayoutAttr();
  if (outputLayout) {
    if (getDlf16ToF32())
      return emitOpError("Cannot have a zLayout and request f32 output");
    if (!supportedLayoutForCompilerGeneratedStickUnstick(
            outputLayout, /*nhwc*/ false)) {
      return emitOpError("Support only a subset of zLayouts");
    }
  }

  // Check validity of reshape split.
  int64_t splitAxis = getReshapeSplitAxis();
  int64_t splitRank = sourceRank;
  if (splitAxis != -1) {
    // We have a split, make sure the axis is in the proper range.
    if (splitAxis < 0 || splitAxis >= splitRank)
      return emitOpError("out of bound split Axis");
    if (splitAxis == sourceRank - 1) {
      // Split the innermost one, ensure the split factor is a multiple of 64.
      int64_t splitFactor = getReshapeSplitFactor();
      int64_t splitDim = sourceShape[sourceRank - 1];
      if (splitDim % splitFactor != 0)
        return emitOpError(
            "Expected the split to break into even sub-partitions");
      if (splitFactor % 64 != 0)
        return emitOpError("Innermost dim expected to remain a multiple of 64");
    }
    // Because of the split, rank increases by one.
    splitRank++;
  }

  // Check the transpose.
  auto transposePattern = getTransposePattern();
  if (transposePattern.has_value()) {
    // Check rank and that we have an actual permutation.
    int64_t transposeRank = ArrayAttrSize(transposePattern);
    if (transposeRank != splitRank)
      return emitOpError(
          "Rank of transpose pattern does not match the (splitted) source");
    for (int64_t permVal = 0; permVal < transposeRank; ++permVal) {
      // Ensure we have permVal in the pattern.
      bool hasVal = false;
      for (int d = 0; d < transposeRank; ++d) {
        if (ArrayAttrIntVal(transposePattern, d) == hasVal) {
          // Got it.
          hasVal = true;
          break;
        }
      }
      if (!hasVal)
        return emitOpError("Transpose pattern is not a permutation");
    }
    // Second constraint: we don't want to permute the innermost dim. Could
    // potentially be relaxed.
    if (ArrayAttrIntVal(transposePattern, splitRank - 1) != splitRank - 1)
      return emitOpError("Support transpose of all but innermost dim");
  }

  // Check the merge
  int64_t mergeAxis = getReshapeMergeAxis();
  if (mergeAxis != -1) {
    if (!(mergeAxis >= 0 && mergeAxis < splitRank - 1))
      return emitOpError("Reshape merge axis out of bound");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Custom builders
//===----------------------------------------------------------------------===//

void ZHighExtendedLayoutTransformOp::build(OpBuilder &builder,
    OperationState &state, Type resType, Value source, int64_t reshapeSplitAxis,
    int64_t reshapeSplitFactor, std::optional<ArrayAttr> transposePattern,
    int64_t reshapeMergeAxis, bool dlf16ToF32,
    std::optional<StringAttr> layout) {

  auto si64Ty = builder.getIntegerType(64, /*isSigned=*/true);
  build(builder, state, resType, source,
      builder.getIntegerAttr(si64Ty, reshapeSplitAxis),
      builder.getIntegerAttr(si64Ty, reshapeSplitFactor),
      transposePattern.value_or(nullptr),
      builder.getIntegerAttr(si64Ty, reshapeMergeAxis),
      builder.getBoolAttr(dlf16ToF32), layout.value_or(nullptr));
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighExtendedLayoutTransformOpShapeHelper::computeShape() {
  ZHighExtendedLayoutTransformOp eLTOp =
      llvm::cast<ZHighExtendedLayoutTransformOp>(op);
  ZHighExtendedLayoutTransformOpAdaptor operandAdaptor(operands);
  DimsExpr outputDims;

  createIE->getShapeAsDims(operandAdaptor.getSource(), sourceDims);
  int64_t sourceRank = sourceDims.size();
  // Handle first step: reshape that split one dim into two.
  int64_t splitAxis = eLTOp.getReshapeSplitAxis();
  IndexExpr splitFactor = LitIE(eLTOp.getReshapeSplitFactor());
  reshapeSplitDims.clear();
  for (int64_t d = 0; d < sourceRank; ++d) {
    if (d != splitAxis) {
      // If splitAxis is -1, we will always come here.
      reshapeSplitDims.emplace_back(sourceDims[d]);
    } else {
      // Split: add sourceDim[d]/splitFactor. splitfactor, splitFactor.
      reshapeSplitDims.emplace_back(sourceDims[d].ceilDiv(splitFactor));
      reshapeSplitDims.emplace_back(splitFactor);
    }
  }
  int64_t splitRank = reshapeSplitDims.size();

  // Handle second step: transpose; no change in rank.
  transposeDims = reshapeSplitDims;
  auto transposePattern = eLTOp.getTransposePattern();
  if (transposePattern.has_value()) {
    for (int64_t d = 0; d < splitRank; ++d) {
      int64_t permuteIndex = ArrayAttrIntVal(transposePattern, d);
      transposeDims[d] = reshapeSplitDims[permuteIndex];
    }
  }

  // Handle third steps.
  int64_t mergeAxis = eLTOp.getReshapeMergeAxis();
  reshapeMergeDims.clear();
  for (int64_t d = 0; d < splitRank; ++d) {
    if (d != mergeAxis) {
      // If merge axis == -1, always come here.
      reshapeMergeDims.emplace_back(transposeDims[d]);
    } else {
      // Merge this and the next dim together.
      reshapeMergeDims.emplace_back(transposeDims[d] * transposeDims[d + 1]);
      // skip next d
      ++d;
    }
  }
  setOutputDims(reshapeMergeDims);
  return success();
}

LogicalResult ZHighExtendedLayoutTransformOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getSource()))
    return success();

  Type elementType = getElementType(getResult().getType());
  ZHighExtendedLayoutTransformOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Canonicalization patterns
//===----------------------------------------------------------------------===//

} // namespace zhigh
} // namespace onnx_mlir
