/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- DFT.cpp - Lowering DFT Op ----------------===//
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
  Location loc = ONNXLoc<OP_TYPE>(op);
  ValueRange operands = adaptor.getOperands();

  //   using MDBuilder =
  //       MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
  //       MathBuilder,
  //           MemRefBuilder, VectorBuilder, AffineBuilderKrnlMem, SCFBuilder>;

  Value input = adaptor.getInput();

  //   int64_t oneSided = adaptor.getOnesided();

  // Convert the output type to MemRefType.
  Type convertedType = typeConverter->convertType(*op->result_type_begin());
  assert(convertedType && convertedType.isa<MemRefType>() &&
         "Failed to convert type to MemRefType");

  // Get shape.
  ONNXGenericDFTOpShapeHelper<OP_TYPE> shapeHelper(op, operands);
  shapeHelper.computeShapeAndAssertOnFailure();

  // Extract raw axis from operation.

  //   IndexExpr rawAxisIE;
  //   MDBuilder create(rewriter, loc);
  //   Value axisVal = nullptr;
  //   bool hasNoAxis = false; // Axis is a None (i.e. optional value not
  //   given). bool dynamicAxis = false; if constexpr (isAxisInput<OP_TYPE>) {
  //     // Here axis would be the input value
  //     axisVal = adaptor.getAxis();
  //     if (isNoneValue(axisVal)) {
  //       // Default value of having no axiss.
  //       hasNoAxis = true;
  //     } else {
  //         // We have a shape, try to get the array integers
  //         // rawAxisIE = LiteralIndexExpr(axisVal);
  //     }
  //   } else {
  //     // Deal with operations where we find axis in the attributes.
  //     int64_t axisAttrs = adaptor.getAxis();
  //     if (!axisAttrs) {
  //       // Default value of having no axis.
  //       hasNoAxis = true;
  //     } else {
  //     //   rawAxisIE = create.krnlIE.getIntAsSymbol(axisAttrs);
  //     }
  //   }
  onnxToKrnlSimdReport(op);
  return success();
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

struct ONNXDFTV17OpLowering : public OpConversionPattern<ONNXDFTV17Op> {
  ONNXDFTV17OpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXDFTV17Op dftOp, ONNXDFTV17OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    return ONNXDFTOpLoweringCommon<ONNXDFTV17Op, ONNXDFTV17OpAdaptor>(
        dftOp, adaptor, rewriter, typeConverter);
  }
};

void populateLoweringONNXDFTOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXDFTOpLowering>(typeConverter, ctx);
}

void populateLoweringONNXDFTV17OpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXDFTV17OpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
