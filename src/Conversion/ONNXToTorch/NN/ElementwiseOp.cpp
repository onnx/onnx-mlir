/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ElementwiseOp.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2022, Helprack LLC.
//
// =============================================================================
//
// This file lowers most unary operators from onnx to torch using a template
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

/**
 * Lowers most unary operators from onnx to torch using a template. It accepts
 * two parameters, the first one is ONNX operator and the second paramenter is
 * the equivalent Torch ATen operator.
 *
 * Operands:
 *   X/input    tensor, type depends on the operator.
 *
 * Results:
 *   Y/output   tensor, type depends on the operator.
 */

template <typename ONNXUnaryOp, typename TorchUnaryOp>
struct ONNXToTorchElementwiseUnaryOpLowering : public ConversionPattern {
  ONNXToTorchElementwiseUnaryOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ONNXUnaryOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXUnaryOp unaryOp = llvm::dyn_cast_or_null<ONNXUnaryOp>(op);

    assert(unaryOp && "Expecting op to have a strong type");

    Location loc = unaryOp.getLoc();
    mlir::MLIRContext *context = unaryOp.getContext();

    auto resultType = toTorchType(context, unaryOp.getResult().getType());
    auto operandTensor =
        getTorchTensor(unaryOp.getOperand(), rewriter, context, loc);

    Value result =
        rewriter.create<TorchUnaryOp>(loc, resultType, operandTensor);

    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultType, result);

    return success();
  }
};

void populateLoweringONNXToTorchElementwiseOpPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAtanOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXCosOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXCoshOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXErfOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAcosOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAcoshOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAsinOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAsinhOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAtanhOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXExpOp, AtenExpOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXLogOp, AtenLogOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXSignOp>,
      ONNXToTorchElementwiseUnaryOpLowering<ONNXSinOp, AtenSinOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXSinhOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXSoftsignOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXTanhOp, AtenTanhOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXTanOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAbsOp, AtenAbsOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXCeilOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXFloorOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXNegOp, AtenNegOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXNotOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXReciprocalOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXReluOp, AtenReluOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXRoundOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXSigmoidOp, AtenSigmoidOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXSoftplusOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXSqrtOp, AtenSqrtOp>>(
      typeConverter, ctx);
}
