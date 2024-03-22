/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- DFT.cpp - Lowering DFT Op --------------------------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX DFT Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

// Lowering Computation for DFT is incomplete !!

using namespace mlir;

namespace onnx_mlir {

template <typename OP>
constexpr bool isAxisInput = std::is_same_v<OP, ONNXDFTOp>;

template <typename OP_TYPE, typename OP_ADAPTOR>
LogicalResult ONNXDFTOpLoweringCommon(OP_TYPE dftOp, OP_ADAPTOR adaptor,
    ConversionPatternRewriter &rewriter, const TypeConverter *typeConverter) {
  Operation *op = dftOp.getOperation();
  return op->emitError("The lowering is not supported for DFT at this time.");
}

struct ONNXDFTOpLowering : public OpConversionPattern<ONNXDFTOp> {
  ONNXDFTOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXDFTOp dftOp, ONNXDFTOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    return ONNXDFTOpLoweringCommon<ONNXDFTOp, ONNXDFTOpAdaptor>(
        dftOp, adaptor, rewriter, typeConverter);
  }
};

void populateLoweringONNXDFTOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXDFTOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
