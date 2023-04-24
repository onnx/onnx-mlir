/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- ONNXDequantizeLinearOp.cpp - ONNXDequantizeLinearOp --------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNXDequantizeLinearOp operator to TOSA dialect.
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

class ONNXDequantizeLinearOpLoweringToTOSA : public ConversionPattern {
public:
  ONNXDequantizeLinearOpLoweringToTOSA(MLIRContext *ctx)
      : ConversionPattern(ONNXDequantizeLinearOp::getOperationName(), 1, ctx) {}
  using OpAdaptor = typename ONNXDequantizeLinearOp::Adaptor;
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Quantization formula is (x - zero_point) * scale
    // TODO : Axis attribute
    TosaBuilder tosaBuilder(rewriter, op->getLoc());
    OpAdaptor adaptor(operands, op->getAttrDictionary());
    auto deqLinearOp = llvm::cast<ONNXDequantizeLinearOp>(op);
    auto result = deqLinearOp.getResult().getType();

    mlir::Value x = deqLinearOp.x();
    auto x_scale = deqLinearOp.x_scale();
    mlir::Value x_zero_point = deqLinearOp.x_zero_point();
    auto loc = op->getLoc();

    // Cast into the destination type first
    auto castOp = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(rewriter, loc, result, x).getResult();
    auto subOp = tosa::CreateOpAndInfer<mlir::tosa::SubOp>(rewriter, loc, result, castOp, x_zero_point).getResult();
    auto mulOp = tosa::CreateOpAndInfer<mlir::tosa::MulOp>(rewriter, loc, result, subOp, x_scale, 0).getResult();

    rewriter.replaceOp(op, mulOp);
    return success();
  }
};

} // namespace

void populateLoweringONNXDequantizeLinearOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXDequantizeLinearOpLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir