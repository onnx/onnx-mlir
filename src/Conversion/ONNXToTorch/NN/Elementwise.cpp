/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- Elementwise.cpp - ONNX Op Transform ----------------------===//
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

// Lowers most unary operators from onnx to torch using a template. It accepts
// two parameters, the first one is ONNX operator and the second paramenter is
// the equivalent Torch ATen operator.
//
// Operands:
//    X/input    tensor, type depends on the operator.
//
// Results:
//    Y/output   tensor, type depends on the operator.
//
template <typename ONNXUnaryOp, typename TorchUnaryOp>
struct ONNXToTorchElementwiseUnaryOpLowering : public ConversionPattern {
  ONNXToTorchElementwiseUnaryOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ONNXUnaryOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    ONNXUnaryOp unaryOp = llvm::dyn_cast_or_null<ONNXUnaryOp>(op);
    if (!unaryOp)
      return op->emitError("expecting op to have a strong type");

    Location loc = unaryOp.getLoc();
    mlir::MLIRContext *context = unaryOp.getContext();

    Torch::ValueTensorType resultType =
        toTorchType(context, unaryOp.getResult().getType());
    mlir::Value result =
        rewriter.create<TorchUnaryOp>(loc, resultType, operands[0]);
    setLayerNameAttr(op, result.getDefiningOp());

    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultType, result);
    return success();
  }
};

void populateLoweringONNXToTorchElementwiseOpPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXCosOp, AtenCosOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXSinOp, AtenSinOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXTanhOp, AtenTanhOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXExpOp, AtenExpOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXLogOp, AtenLogOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAbsOp, AtenAbsOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXCeilOp, AtenCeilOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXFloorOp, AtenFloorOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXNegOp, AtenNegOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXReluOp, AtenReluOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXSigmoidOp, AtenSigmoidOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXSqrtOp, AtenSqrtOp>>(
      typeConverter, ctx);
}
