/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- ONNXAveragePoolOp.cpp - ONNXAveragePoolOp Op------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNXAveragePoolOp operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "llvm/ADT/ArrayRef.h"
#include <src/Dialect/Mlir/IndexExpr.hpp>

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXAveragePoolOpLoweringToTOSA : public ConversionPattern {
public:
  ONNXAveragePoolOpLoweringToTOSA(MLIRContext *ctx)
      : ConversionPattern(ONNXAveragePoolOp::getOperationName(), 1, ctx) {}
  using OpAdaptor = typename ONNXAveragePoolOp::Adaptor;
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto averagePoolOp = llvm::cast<ONNXAveragePoolOp>(op);
    OpAdaptor adaptor(operands, op->getAttrDictionary());

    Value input = adaptor.X();
    // The attributes storage_order and dilations are unsupported
    auto inputType = input.getType().cast<TensorType>();
    const int64_t includePad = adaptor.count_include_pad();

    if (inputType.getShape().size() != 4) {
      return rewriter.notifyMatchFailure(
          averagePoolOp, "TOSA only supports maxpool 2d");
    }
    if (includePad != 0) {
      return rewriter.notifyMatchFailure(
          averagePoolOp, "count_include_pad must be 0");
    }

    llvm::Optional<Value> newAveragePoolOp =
        tosa::convertPoolOp<ONNXAveragePoolOp, mlir::tosa::AvgPool2dOp>(
            rewriter, op);

    if (!newAveragePoolOp) {
      return rewriter.notifyMatchFailure(
          averagePoolOp, "Could not create maxpool op.");
    }

    rewriter.replaceOp(op, newAveragePoolOp.value());
    return success();
  }
};

} // namespace

void populateLoweringONNXAveragePoolOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXAveragePoolOpLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir