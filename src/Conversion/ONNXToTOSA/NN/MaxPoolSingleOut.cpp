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
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXMaxPoolSingleOutOpLoweringToTOSA
    : public OpConversionPattern<ONNXMaxPoolSingleOutOp> {
public:
  using OpConversionPattern<ONNXMaxPoolSingleOutOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXMaxPoolSingleOutOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXMaxPoolSingleOutOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();

    Value Input = adaptor.X();
    // - The attribute Ceil_mode is ignored because its effect is already
    // impacting the return type of the operator
    // - The attribute auto_pad is ignored because it is marked as deprecated
    // for ONNX
    // - The attributes storage_order and dilations is also unsupported
    mlir::StringAttr autoPad = adaptor.auto_padAttr();
    mlir::IntegerAttr storageOrder = adaptor.storage_orderAttr();
    mlir::ArrayAttr dilations = adaptor.dilationsAttr();
    mlir::ArrayAttr kernelShape = adaptor.kernel_shapeAttr();
    mlir::ArrayAttr pads = adaptor.padsAttr();
    mlir::ArrayAttr strides = adaptor.stridesAttr();

    if (Input.getType().isa<MemRefType>()) {
      return rewriter.notifyMatchFailure(
          op, "memrefs as inputs are unsupported by TOSA");
    }
    auto InputType = Input.getType().cast<TensorType>();
    auto resultType = op.getResult().getType();
    auto resultShape = resultType.cast<TensorType>().getShape();
    // Construct the transposed type for the new MaxPool OP
    Type newResultType = RankedTensorType::get(
        {resultShape[0], resultShape[2], resultShape[3], resultShape[1]},
        InputType.getElementType());

    if (autoPad != "NOTSET") {
      op.emitWarning(
          "auto_pad attribute is deprecated and its value will be ignored");
    }
    if (dilations) {
      op.emitWarning("dilations attribute is unsupported by TOSA and its value "
                     "will be ignored");
    }
    if (storageOrder && storageOrder.getSInt() != 0) {
      op.emitWarning("storage_order attribute is unsupported by TOSA and its "
                     "value will be ignored");
    }

    // ONNX Mlir uses NCHW as an input while TOSA expects NHWC. Insert a
    // transpose to change the format
    Input = tosa::createTosaTransposedTensor(rewriter, op, Input, {0, 2, 3, 1});

    // Pads and Strides are optionals for ONNX but mandatory for TOSA
    if (!pads) {
      pads = rewriter.getI64ArrayAttr({0, 0, 0, 0});
    } else {
      llvm::SmallVector<int64_t, 4> transposedPads =
          extractFromI64ArrayAttr(pads);
      pads = rewriter.getI64ArrayAttr({transposedPads[0], transposedPads[2],
          transposedPads[1], transposedPads[3]});
    }
    if (!strides) {
      strides = rewriter.getI64ArrayAttr({1, 1});
    }

    Input = rewriter
                .create<mlir::tosa::MaxPool2dOp>(
                    loc, newResultType, Input, kernelShape, strides, pads)
                .getResult();
    // Revert to original shape (NCHW)
    // Construct the old result shape out of the new one
    auto newInputType = Input.getType().cast<RankedTensorType>().getShape();
    Value sourceTensor =
        tosa::getConstTensor<int32_t>(rewriter, op, {0, 3, 1, 2}, {4}).value();
    Type transposedResultType = RankedTensorType::get(
        {newInputType[0], newInputType[3], newInputType[1], newInputType[2]},
        InputType.getElementType());
    tosa::CreateReplaceOpAndInfer<mlir::tosa::TransposeOp>(
        rewriter, op, transposedResultType, Input, sourceTensor);
    return success();
  }
};

} // namespace

void populateLoweringONNXMaxPoolSingleOutOpToTOSAPattern(
    ConversionTarget &target, RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXMaxPoolSingleOutOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir