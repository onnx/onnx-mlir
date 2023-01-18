/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- MaxPoolSingleOut.cpp - MaxPoolSingleOut Op-----------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#include <src/Dialect/Mlir/IndexExpr.hpp>

using namespace mlir;

namespace onnx_mlir {

namespace {

// This calculates the values that need to be added to the padding in order to
// simulate the ceil mode
llvm::SmallVector<int64_t> getCeilConstants(llvm::ArrayRef<int64_t> inputShape,
    const ONNXMaxPoolSingleOutOpShapeHelper &shapeHelper, int64_t ceilMode) {
  // This avoids having more if statements when creating the padding const.
  if (ceilMode == 0)
    return llvm::SmallVector<int64_t>{0, 0};

  SmallVector<int64_t, 2> kernelShapeVec =
      tosa::createInt64VectorFromIndexExpr(shapeHelper.kernelShape);

  // Get stride and pad vectors.
  SmallVector<int64_t, 2> stridesVec = shapeHelper.strides;
  SmallVector<int64_t, 4> padsVec =
      tosa::createInt64VectorFromIndexExpr(shapeHelper.pads);

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
  // using ConversionPattern<ONNXMaxPoolSingleOutOp>::ConversionPattern;
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

    Value input = adaptor.X();

    // The attributes storage_order and dilations are unsupported
    mlir::IntegerAttr storageOrder = adaptor.storage_orderAttr();
    mlir::ArrayAttr dilations = adaptor.dilationsAttr();
    mlir::ArrayAttr kernelShape = adaptor.kernel_shapeAttr();
    const int64_t ceilMode = adaptor.ceil_mode();

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
        llvm::SmallVector<int64_t, 4>(inputType.getShape().size(), -1),
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
    input = tosa::createTosaTransposedTensor(
        rewriter, maxpoolOp, input, {0, 2, 3, 1});

    // When ceil mode is 1, we need to add values to the padding
    const llvm::SmallVector<int64_t, 4> ceilConstants =
        getCeilConstants(inputType.getShape(), shapeHelper, ceilMode);
    llvm::SmallVector<int64_t, 4> pads =
        tosa::createInt64VectorFromIndexExpr(shapeHelper.pads);

    // reorder padding values
    mlir::ArrayAttr newPads = rewriter.getI64ArrayAttr({pads[0],
        pads[2] + ceilConstants[0], pads[1], pads[3] + ceilConstants[1]});

    mlir::ArrayAttr strides = rewriter.getI64ArrayAttr(shapeHelper.strides);

    input = tosa::CreateOpAndInfer<mlir::tosa::MaxPool2dOp>(
        rewriter, loc, newResultType, input, kernelShape, strides, newPads);

    // Revert to original shape (NCHW)
    // Construct the old result shape out of the new one
    auto newInputType = input.getType().cast<RankedTensorType>().getShape();
    Value sourceTensor =
        tosa::getConstTensor<int32_t>(rewriter, maxpoolOp, {0, 3, 1, 2}, {4})
            .value();

    Type transposedResultType = RankedTensorType::get(
        llvm::SmallVector<int64_t, 4>(newInputType.size(), -1),
        inputType.getElementType());
    tosa::CreateReplaceOpAndInfer<mlir::tosa::TransposeOp>(
        rewriter, maxpoolOp, transposedResultType, input, sourceTensor);
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