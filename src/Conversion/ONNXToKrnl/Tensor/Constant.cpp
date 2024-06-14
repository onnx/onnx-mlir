/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Constant.cpp - Lowering Constant Op -----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Constant Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXConstantOpLowering : public OpConversionPattern<ONNXConstantOp> {
  ONNXConstantOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXConstantOp constantOp,
      ONNXConstantOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = constantOp.getOperation();
    Location loc = ONNXLoc<ONNXConstantOp>(op);

    if (constantOp.getSparseValue().has_value())
      return emitError(loc, "Only support dense values at this time");

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = mlir::cast<MemRefType>(convertedType);

    // Emit the constant global in Krnl dialect.
    MultiDialectBuilder<KrnlBuilder> create(rewriter, loc);
    mlir::Attribute constValAttr = constantOp.getValue().value();
    if (mlir::isa<krnl::StringType>(memRefType.getElementType())) {
      // If the onnx.ConstantOp has string type value attribute,
      // The element type of the value attribute of krnl.global op should be
      // "!krnl.string" instead of "!onnx.String".
      ShapedType constStrType = RankedTensorType::get(
          memRefType.getShape(), krnl::StringType::get(rewriter.getContext()));
      SmallVector<StringRef> constStrVector(
          mlir::dyn_cast<DenseElementsAttr>(constValAttr)
              .getValues<StringAttr>());
      ArrayRef<StringRef> constStrValues(constStrVector);
      constValAttr = mlir::DenseElementsAttr::get(constStrType, constStrValues);
    }
    Value constantGlobal =
        create.krnl.constant(memRefType, "constant_", constValAttr);

    // Replace this operation with the generated krnl.global.
    rewriter.replaceOp(op, constantGlobal);

    return success();
  }
};

void populateLoweringONNXConstantOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXConstantOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
