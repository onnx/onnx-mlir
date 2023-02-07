/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- MaxPoolSingleOut.cpp - MaxPoolSingleOut Op-----------===//
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX MaxpoolSingleOut operator to TOSA dialect.
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
#include <cstdint>
#include <src/Dialect/Mlir/IndexExpr.hpp>

using namespace mlir;

namespace onnx_mlir {

namespace {

// This calculates the values that need to be added to the padding in order to
// simulate the ceil mode
llvm::SmallVector<int64_t> getCeilConstants(llvm::ArrayRef<int64_t> inputShape,
    ONNXMaxPoolSingleOutOpShapeHelper &shapeHelper, int64_t ceilMode) {
  // This avoids having more if statements when creating the padding const.
  if (ceilMode == 0)
    return llvm::SmallVector<int64_t>{0, 0};

  SmallVector<int64_t, 4> kernelShapeVec;
  IndexExpr::getLiteral(shapeHelper.kernelShape, kernelShapeVec);

  // Get stride and pad vectors.
  SmallVector<int64_t, 2> stridesVec = shapeHelper.strides;
  SmallVector<int64_t, 4> padsVec;
  IndexExpr::getLiteral(shapeHelper.pads, padsVec);

  // Check if the idiv_check for the output dimentions in
  // https://www.mlplatform.org/tosa/tosa_spec.html#_max_pool2d has no
  // remainder. If it has a remainder, we add size(stride) to the end of the
  // padding dimension to get one dimension up. Height and width need to have
  // seperate values.
  int64_t xAxis = 0;
  if ((inputShape[2] + padsVec[0] + padsVec[2] - kernelShapeVec[0]) %
      stridesVec[0])
    xAxis = stridesVec[0];

  int64_t yAxis = 0;
  if ((inputShape[3] + padsVec[1] + padsVec[3] - kernelShapeVec[1]) %
      stridesVec[1])
    yAxis = stridesVec[1];

  return llvm::SmallVector<int64_t>{xAxis, yAxis};
}

class ONNXMaxPoolSingleOutOpLoweringToTOSA : public ConversionPattern {
public:
  ONNXMaxPoolSingleOutOpLoweringToTOSA(MLIRContext *ctx)
      : ConversionPattern(ONNXMaxPoolSingleOutOp::getOperationName(), 1, ctx) {}

  using OpAdaptor = typename ONNXMaxPoolSingleOutOp::Adaptor;
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto maxpoolOp = llvm::cast<ONNXMaxPoolSingleOutOp>(op);
    Location loc = op->getLoc();
    OpAdaptor adaptor(operands, op->getAttrDictionary());
    // Get shape.
    IndexExprBuilderForTosa createTosaIE(rewriter, loc);
    ONNXMaxPoolSingleOutOpShapeHelper shapeHelper(op, operands, &createTosaIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    TosaBuilder tosaBuilder(rewriter, loc);

    Value input = adaptor.getX();

    // The attributes storage_order and dilations are unsupported
    mlir::IntegerAttr storageOrder = adaptor.getStorageOrderAttr();
    mlir::ArrayAttr dilations = adaptor.getDilationsAttr();
    const int64_t ceilMode = adaptor.getCeilMode();

    if (input.getType().isa<MemRefType>()) {
      return rewriter.notifyMatchFailure(
          op, "memrefs as inputs are unsupported by TOSA");
    }
    auto inputType = input.getType().cast<TensorType>();

    if (inputType.getShape().size() != 4) {
      return rewriter.notifyMatchFailure(
          maxpoolOp, "TOSA only supports maxpool 2d");
    }

    // Construct the transposed type for the new MaxPool OP
    Type newResultType = RankedTensorType::get(
        llvm::SmallVector<int64_t, 4>(
            inputType.getShape().size(), ShapedType::kDynamic),
        inputType.getElementType());

    if (dilations) {
      return rewriter.notifyMatchFailure(
          maxpoolOp, "dilations attribute is unsupported by TOSA");
    }
    if (storageOrder && storageOrder.getSInt() != 0) {
      return rewriter.notifyMatchFailure(
          maxpoolOp, "storage_order attribute is unsupported by TOSA");
    }

    // ONNX Mlir uses NCHW as an input while TOSA expects NHWC. Insert a
    // transpose to change the format
    Value newMaxpoolInput = tosaBuilder.transpose(input, {0, 2, 3, 1});

    if (!IndexExpr::isLiteral(shapeHelper.pads)) {
      return rewriter.notifyMatchFailure(
          maxpoolOp, "could not determine pad values");
    }
    if (!IndexExpr::isLiteral(shapeHelper.kernelShape)) {
      return rewriter.notifyMatchFailure(
          maxpoolOp, "could not determine kernel_shape values");
    }
    // When ceil mode is 1, we need to add values to the padding
    const llvm::SmallVector<int64_t, 4> ceilConstants =
        getCeilConstants(inputType.getShape(), shapeHelper, ceilMode);
    llvm::SmallVector<int64_t, 4> pads;
    IndexExpr::getLiteral(shapeHelper.pads, pads);

    // reorder padding values
    auto newPads = rewriter.getDenseI64ArrayAttr({pads[0],
        pads[2] + ceilConstants[0], pads[1], pads[3] + ceilConstants[1]});

    auto strides = rewriter.getDenseI64ArrayAttr(shapeHelper.strides);
    llvm::SmallVector<int64_t, 4> kernelShapeVec;
    IndexExpr::getLiteral(shapeHelper.kernelShape, kernelShapeVec);
    auto kernelShape = rewriter.getDenseI64ArrayAttr(kernelShapeVec);

    Value newMaxpool = tosa::CreateOpAndInfer<mlir::tosa::MaxPool2dOp>(rewriter,
        loc, newResultType, newMaxpoolInput, kernelShape, strides, newPads);

    // Revert to original shape (NCHW)
    // Construct the old result shape out of the new one
    Value transpose = tosaBuilder.transpose(newMaxpool, {0, 3, 1, 2});

    rewriter.replaceOp(op, transpose);
    return success();
  }
};

} // namespace

void populateLoweringONNXMaxPoolSingleOutOpToTOSAPattern(
    ConversionTarget &target, RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXMaxPoolSingleOutOpLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir