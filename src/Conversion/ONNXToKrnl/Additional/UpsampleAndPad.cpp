/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- UpsampleAndPad.cpp - Lowering UpsampleAndPad Op ---------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX UpsampleAndPad Operator to Krnl dialect.
//
// This is a reference implementation that:
// 1. Allocates and zeros the output tensor
// 2. Loops over the input tensor and writes each element to its corresponding
//    position in the output (accounting for upsampling strides and padding)
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXUpsampleAndPadOpLowering
    : public OpConversionPattern<ONNXUpsampleAndPadOp> {
  ONNXUpsampleAndPadOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel)
      : OpConversionPattern(typeConverter, ctx) {
    this->enableParallel = enableParallel;
  }

  bool enableParallel = false;

  LogicalResult matchAndRewrite(ONNXUpsampleAndPadOp upsampleAndPadOp,
      ONNXUpsampleAndPadOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    // Gather info.
    Operation *op = upsampleAndPadOp.getOperation();
    Location loc = ONNXLoc<ONNXUpsampleAndPadOp>(op);
    ValueRange operands = adaptor.getOperands();
    Value X = adaptor.getX();

    // Builder helper.
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);

    // Shape helper.
    ONNXUpsampleAndPadOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    Type elementType = outputMemRefType.getElementType();

    // Get input dimensions.
    uint64_t rank = create.krnlIE.getShapedTypeRank(X);

    // Get strides and pads attributes.
    std::optional<ArrayAttr> stridesAttrOpt = adaptor.getStrides();
    std::optional<ArrayAttr> padsAttrOpt = adaptor.getPads();

    // Determine k and get stride/pad values.
    uint64_t k = 0;
    SmallVector<int64_t, 4> stridesVec;
    SmallVector<int64_t, 8> padsVec;

    if (stridesAttrOpt.has_value()) {
      ArrayAttr stridesAttr = stridesAttrOpt.value();
      k = stridesAttr.size();
      for (auto attr : stridesAttr)
        stridesVec.push_back(mlir::cast<IntegerAttr>(attr).getInt());
    }

    if (padsAttrOpt.has_value()) {
      ArrayAttr padsAttr = padsAttrOpt.value();
      if (k == 0)
        k = padsAttr.size() / 2;
      for (auto attr : padsAttr)
        padsVec.push_back(mlir::cast<IntegerAttr>(attr).getInt());
    }

    // Default values if not specified.
    if (k == 0)
      k = rank;
    if (stridesVec.empty())
      stridesVec.assign(k, 1);
    if (padsVec.empty())
      padsVec.assign(2 * k, 0);

    // Allocate output tensor.
    Value outputMemRef =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    // Initialize output with zeros.
    Value zeroValue = create.math.constant(elementType, 0);
    create.krnl.memset(outputMemRef, zeroValue);

    // Loop over the input elements that land in the output. Input index `j`
    // goes to `j * stride + pad_begin`, and a negative pad removes elements.
    LitIE zero(0);
    SmallVector<IndexExpr, 4> lbs(rank, zero);
    SmallVector<IndexExpr, 4> ubs;
    create.krnlIE.getShapeAsDims(X, ubs);
    for (uint64_t i = rank - k; i < rank; ++i) {
      uint64_t dimIdx = i - (rank - k);
      int64_t stride = stridesVec[dimIdx];
      int64_t padBegin = padsVec[dimIdx];
      int64_t padEnd = padsVec[dimIdx + k];
      if (padBegin < 0)
        lbs[i] = LitIE(llvm::divideCeil(-padBegin, stride));
      if (padEnd < 0)
        ubs[i] = ubs[i] - LitIE(llvm::divideCeil(-padEnd, stride));
    }

    ValueRange loopDef = create.krnl.defineLoops(rank);

    // Parallelize the outermost 2 loops if enabled and rank >= 2. Plan holds
    // the whole nest: the window stops at 2, so the same ref is chosen either
    // way.
    auto plan = KrnlParallelPlan::noCollapse(
        loopDef, /*first*/ 0, /*last excl*/ 2, /*cost*/ {8});
    if (enableParallel && rank >= 2) {
      plan.tryCreateParallel(create.krnl, op, "upsample and pad", lbs, ubs);
    }

    create.krnl.iterateIE(loopDef, plan.optimizedLoopDef(), lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange inputLoopInd) {
          // Compute output indices.
          SmallVector<IndexExpr, 4> outputLoopInd;
          for (uint64_t i = 0; i < rank; ++i) {
            if (i < rank - k) {
              // First (rank - k) dimensions: direct mapping.
              outputLoopInd.emplace_back(DimIE(inputLoopInd[i]));
            } else {
              // Last k dimensions: apply stride and padding.
              uint64_t dimIdx = i - (rank - k);
              LitIE stride(stridesVec[dimIdx]);
              LitIE padBegin(padsVec[dimIdx]);
              // output_idx = input_idx * stride + pad_begin
              outputLoopInd.emplace_back(
                  DimIE(inputLoopInd[i]) * stride + padBegin);
            }
          }
          // Load from input and store to output.
          Value inputValue = createKrnl.load(X, inputLoopInd);
          createKrnl.storeIE(inputValue, outputMemRef, outputLoopInd);
        });

    // Replace the original op with the generated code.
    rewriter.replaceOp(op, outputMemRef);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXUpsampleAndPadOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
  patterns.insert<ONNXUpsampleAndPadOpLowering>(
      typeConverter, ctx, enableParallel);
}

} // namespace onnx_mlir
