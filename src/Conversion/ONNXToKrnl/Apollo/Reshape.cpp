/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Reshape.cpp - Reshape Op ---------------------------===//
//
// Copyright 2021 Microsoft
//
// =============================================================================
//
// This file lowers ONNX reshape operator to a function call
// that will be lowered to Apollo-specific code.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

namespace {
struct ONNXReshapeOpApolloLowering : public ConversionPattern {

  ONNXReshapeOpApolloLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXReshapeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXReshapeOpAdaptor operandAdaptor(operands);
    ONNXReshapeOp reshapeOp = dyn_cast_or_null<ONNXReshapeOp>(op);

    auto loc = op->getLoc();
    Value data = operandAdaptor.data();
    Value shape = operandAdaptor.shape();
    auto dataShape = data.getType().cast<MemRefType>().getShape();

    if (!hasAllConstantDimensions(data.getType().cast<MemRefType>())) {
      op->emitOpError("Only static sized reshape supported");
      return failure();
    }

    // If shape input was promoted to attribute, get its values from the
    // attribute.
    SmallVector<int64_t, 4> shapeAttrValues;
    if (auto constOp = getONNXConstantOp(reshapeOp.shape())) {
      if (auto shapeAttr =
              constOp.valueAttr().dyn_cast_or_null<DenseElementsAttr>()) {
        size_t i = 0;
        for (auto e : shapeAttr.getValues<IntegerAttr>()) {
          auto index = i++;
          auto value = e.getInt();
          if (value == 0) {
            // ISSUE-TODO: opset-14 changes this behavior if `allowzero` is set
            // default behavior for 0 is keep same as original
            value = dataShape[index];
          } else if (value < 0) {
            assert(value == -1);
            // ISSUE-TODO: validate only a single dimension is -1
            // ISSUE-TODO: auto-compute the size of the -1 dimension
            op->emitOpError("Reshape with -1 not handled");
            return failure();
          }
          shapeAttrValues.emplace_back(value);
        }
      }
    }

    if (shapeAttrValues.size() == 0) {
      op->emitOpError("Dynamic reshape not supported");
      return failure();
    }

    // Code to generate maps for linalg.reshape is copied from
    //   `mlir\lib\Conversion\TosaToLinalg\TosaToLinalg.cpp`

    // Compute the reassociation maps for the linalg operation.
    ArrayRef<int64_t> expandedShape =
        dataShape.size() > shapeAttrValues.size() ? dataShape : shapeAttrValues;
    ArrayRef<int64_t> collapsedShape =
        dataShape.size() > shapeAttrValues.size() ? shapeAttrValues : dataShape;
    size_t currSrcDim = 0, currDstDim = 0;
    SmallVector<linalg::ReassociationExprs, 4> reassociationMap(
        collapsedShape.size());

    // First scan all dimensions in the source shapes to see whether we have a
    // perfect case where consecutive dimensions in source are collapsed. For
    // such case we can just generate one single linalg.reshape.
    bool canHandleThisReshape = true;
    while (currSrcDim < expandedShape.size() &&
           currDstDim < collapsedShape.size()) {
      int64_t dstSize = collapsedShape[currDstDim];
      int64_t srcSize = expandedShape[currSrcDim];
      while (srcSize < dstSize && currSrcDim < expandedShape.size()) {
        reassociationMap[currDstDim].push_back(
            rewriter.getAffineDimExpr(currSrcDim++));
        srcSize *= expandedShape[currSrcDim];
      }
      if (srcSize == dstSize) {
        reassociationMap[currDstDim].push_back(
            rewriter.getAffineDimExpr(currSrcDim++));
        // If the next dim in collapsedShape is not 1, treat subsequent dims
        // in expandedShape which are 1 to be collapsed.
        if (currDstDim == collapsedShape.size() - 1 ||
            collapsedShape[currDstDim + 1] != 1) {
          while (currSrcDim < expandedShape.size() &&
                 expandedShape[currSrcDim] == 1) {
            reassociationMap[currDstDim].push_back(
                rewriter.getAffineDimExpr(currSrcDim++));
          }
        }
      } else {
        canHandleThisReshape = false;
        break;
      }
      currDstDim++;
    }

    if (currSrcDim != expandedShape.size() ||
        currDstDim != collapsedShape.size())
      canHandleThisReshape = false;

    // original code had a special case for this which i haven't copied over
    // see comment above on where to grab the implementation if needed
    if (!canHandleThisReshape) {
      op->emitOpError("This operation has an unsupported reshape");
      return failure();
    }

    auto resultType = MemRefType::get(
        shapeAttrValues, data.getType().cast<MemRefType>().getElementType());
    auto reshape = rewriter.create<linalg::ReshapeOp>(
        loc, resultType, data, reassociationMap);

    rewriter.replaceOp(op, {reshape});

    return success();
  }
};

} // namespace

void populateLoweringONNXReshapeOpApolloPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXReshapeOpApolloLowering>(ctx);
}
