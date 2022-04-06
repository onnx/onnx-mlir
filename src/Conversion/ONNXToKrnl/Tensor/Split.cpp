/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Split.cpp - Lowering Split Op -----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Split Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

template <typename Adaptor, typename Op, typename ShapeHelper>
LogicalResult ONNXSplitOpLoweringCommon(Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) {
  // Gather info.
  Location loc = op->getLoc();
  Adaptor operandAdaptor(operands);
  Op splitOp = cast<Op>(op);
  uint64_t rank =
      splitOp.input().getType().template cast<ShapedType>().getRank();
  unsigned outputNum = splitOp.getNumResults();
  unsigned axis = splitOp.axis();

  // Get a shape helper.
  ShapeHelper shapeHelper(&splitOp, &rewriter,
      krnl::getDenseElementAttributeFromKrnlValue,
      krnl::loadDenseElementArrayValueAtIndex);
  auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
  assert(succeeded(shapecomputed) && "Could not compute output shape");

  // Alloc and dealloc.
  SmallVector<Value, 4> allocs;
  for (unsigned i = 0; i < outputNum; ++i) {
    checkInsertDealloc(op, i);
    auto memRefType = convertToMemRefType(splitOp.outputs()[i].getType());
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, memRefType, loc, shapeHelper.dimsForOutput(i));
    allocs.emplace_back(alloc);
  }

  // Creates loops, one for each output.
  for (unsigned i = 0; i < outputNum; ++i) {
    OpBuilder::InsertionGuard insertGuard(rewriter);

    // Scope for krnl ops
    IndexExprScope childScope(&rewriter, shapeHelper.scope);

    KrnlBuilder createKrnl(rewriter, loc);
    ValueRange loopDef = createKrnl.defineLoops(rank);
    SmallVector<IndexExpr, 4> lbs(rank, LiteralIndexExpr(0));

    MemRefBoundsIndexCapture allocsBounds(allocs[i]);
    SmallVector<IndexExpr, 4> ubs;
    allocsBounds.getDimList(ubs);

    createKrnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange indices) {
          SmallVector<IndexExpr, 4> readIndices;
          for (uint64_t r = 0; r < rank; ++r) {
            DimIndexExpr readIndex(indices[r]);
            // Compute read index for the split axis.
            if (r == axis)
              for (unsigned k = 0; k < i; ++k) {
                SymbolIndexExpr splitDim(shapeHelper.dimsForOutput(k)[r]);
                readIndex = readIndex + splitDim;
              }

            readIndices.emplace_back(readIndex);
          }

          // Insert copy.
          Value loadData =
              createKrnl.loadIE(operandAdaptor.input(), readIndices);
          createKrnl.store(loadData, allocs[i], indices);
        });
  }

  rewriter.replaceOp(op, allocs);

  return success();
}

struct ONNXSplitOpLowering : public ConversionPattern {
  ONNXSplitOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXSplitOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    return ONNXSplitOpLoweringCommon<ONNXSplitOpAdaptor, ONNXSplitOp,
        ONNXSplitOpShapeHelper>(op, operands, rewriter);
  }
};

struct ONNXSplitV11OpLowering : public ConversionPattern {
  ONNXSplitV11OpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXSplitV11Op::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    return ONNXSplitOpLoweringCommon<ONNXSplitV11OpAdaptor, ONNXSplitV11Op,
        ONNXSplitV11OpShapeHelper>(op, operands, rewriter);
  }
};

void populateLoweringONNXSplitOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSplitOpLowering>(typeConverter, ctx);
}

void populateLoweringONNXSplitV11OpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSplitV11OpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
