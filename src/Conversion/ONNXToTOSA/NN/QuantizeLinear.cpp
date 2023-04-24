/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- ONNXQuantizeLinearOp.cpp - ONNXQuantizeLinearOp --------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNXQuantizeLinearOp operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "torch-mlir/Conversion/TorchToTosa/TosaLegalizeUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include <src/Dialect/Mlir/IndexExpr.hpp>

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXQuantizeLinearOpLoweringToTOSA : public ConversionPattern {
public:
  ONNXQuantizeLinearOpLoweringToTOSA(MLIRContext *ctx)
      : ConversionPattern(ONNXQuantizeLinearOp::getOperationName(), 1, ctx) {}
  using OpAdaptor = typename ONNXQuantizeLinearOp::Adaptor;
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Quantization formula is ((x / y_scale) + y_zero_point)
    TosaBuilder tosaBuilder(rewriter, op->getLoc());
    OpAdaptor adaptor(operands, op->getAttrDictionary());
    auto qLinearOp = llvm::cast<ONNXQuantizeLinearOp>(op);
    // Axis attribute is ignored for per-tensor quantization, which is the only one handled
    // for the moment.
    if (adaptor.axis() != 1) {
      return rewriter.notifyMatchFailure(
          qLinearOp, "Only per-tensor quantization is handled.");
    }

    mlir::Value x = qLinearOp.x();
    auto y_scale = qLinearOp.y_scale();
    mlir::Value y_zero_point = qLinearOp.y_zero_point();
    auto loc = op->getLoc();

    // Replace the division by a reciprocal followed by a mul
    auto recOp = tosa::CreateOpAndInfer<mlir::tosa::ReciprocalOp>(rewriter, loc, x.getType(), x).getResult();
    auto mulOp = tosa::CreateOpAndInfer<mlir::tosa::MulOp>(rewriter, loc, x.getType(), recOp, y_scale, 0).getResult();
    auto addOp = tosa::CreateOpAndInfer<mlir::tosa::AddOp>(rewriter, loc, x.getType(), mulOp, y_zero_point).getResult();
    // Cast into the result type
    auto castOp = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(rewriter, loc, qLinearOp.getResult().getType(), addOp).getResult();

    rewriter.replaceOp(op, castOp);
    return success();
  }
};

} // namespace

void populateLoweringONNXQuantizeLinearOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXQuantizeLinearOpLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir