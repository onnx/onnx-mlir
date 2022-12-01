/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- BinaryOps.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2022, Helprack LLC.
//
// =============================================================================
//
// This file lowers most binary operators from torch to onnx using a template
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

/* Matrix product
 *
 * Operands:
 *  A    tensor of 16-bit float values or tensor of 32-bit float values or
 * tensor of 64-bit float values or tensor of 32-bit unsigned integer values or
 * tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer
 * values or tensor of 64-bit signless integer values or tensor of bfloat16 type
 * values or memref of any type values
 *  B    tensor of 16-bit float values or tensor of 32-bit float values or
 * tensor of 64-bit float values or tensor of 32-bit unsigned integer values or
 * tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer
 * values or tensor of 64-bit signless integer values or tensor of bfloat16 type
 * values or memref of any type values
 *
 * Result:
 *  Y    tensor of 16-bit float values or tensor of 32-bit float values or
 * tensor of 64-bit float values or tensor of 32-bit unsigned integer values or
 * tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer
 * values or tensor of 64-bit signless integer values or tensor of bfloat16 type
 * values or memref of any type values
 */

template <typename ONNXBinaryOp, typename TorchBinaryOp>
struct ONNXToTorchElementwiseBinaryOpLowering : public ConversionPattern {
  ONNXToTorchElementwiseBinaryOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ONNXBinaryOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    ONNXBinaryOp binaryOp = llvm::dyn_cast_or_null<ONNXBinaryOp>(op);
    Location loc = binaryOp.getLoc();
    auto resultType =
        getTypeConverter()->convertType(op->getResult(0).getType());
    Value result = rewriter.create<TorchBinaryOp>(
        loc, resultType, operands[0], operands[1]);
    setLayerNameAttr(op, result.getDefiningOp());
    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultType, result);
    return success();
  }
};

void populateLoweringONNXToTorchBinaryOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<
      ONNXToTorchElementwiseBinaryOpLowering<mlir::ONNXMatMulOp, AtenMatmulOp>>(
      typeConverter, ctx);
}
