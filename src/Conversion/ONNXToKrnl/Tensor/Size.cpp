/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- Size.cpp - Lowering Size Op --------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Size Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXSizeOpLowering : public OpConversionPattern<ONNXSizeOp> {
  ONNXSizeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXSizeOp sizeOp, ONNXSizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    // Gather info.
    Operation *op = sizeOp.getOperation();
    Location loc = ONNXLoc<ONNXSizeOp>(op);

    MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder> create(
        rewriter, loc);

    Value data = adaptor.getData();
    ArrayRef<int64_t> dataShape =
        mlir::cast<MemRefType>(data.getType()).getShape();
    Value resultOperand = sizeOp.getSize();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = mlir::cast<MemRefType>(convertedType);

    Value alloc = create.mem.alignedAlloc(resultOperand, memRefType);

    // Accumulate static dimensions first.
    int64_t staticNumElement = 1;
    bool allStaticDimensions = true;
    for (unsigned i = 0; i < dataShape.size(); i++) {
      if (!ShapedType::isDynamic(dataShape[i]))
        staticNumElement *= dataShape[i];
      else
        allStaticDimensions = false;
    }
    // Accumulate the remaining dimensions that are unknown.
    Value noElements =
        create.math.constant(memRefType.getElementType(), staticNumElement);
    if (!allStaticDimensions) {
      for (size_t i = 0; i < dataShape.size(); i++) {
        if (ShapedType::isDynamic(dataShape[i])) {
          Value index = create.mem.dim(data, i);
          Value dim = create.math.cast(memRefType.getElementType(), index);
          noElements = create.math.mul(noElements, dim);
        }
      }
    }

    create.krnl.store(noElements, alloc, std::nullopt);
    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXSizeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSizeOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
