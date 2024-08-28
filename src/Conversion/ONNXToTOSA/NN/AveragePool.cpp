/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- ONNXAveragePoolOp.cpp - ONNXAveragePoolOp --------------===//
//
// Copyright (c) 2023 Advanced Micro Devices, Inc.
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

void handleIncludePadAttr(
    ConversionPatternRewriter &rewriter, Operation *op, Value input) {
  Location loc = op->getLoc();

  // Get shape.
  IndexExprBuilderForTosa createTosaIE(rewriter, loc);
  ONNXGenericPoolOpShapeHelper<ONNXAveragePoolOp> shapeHelper(
      op, {}, &createTosaIE);
  shapeHelper.computeShapeAndAssertOnFailure();

  // Build an array with padding.
  llvm::SmallVector<int64_t, 4> intValues;
  IndexExpr::getLiteral(shapeHelper.pads, intValues);

  // Create Padding and ConstPad tosa::ConstOp's
  TosaBuilder tosaBuilder(rewriter, loc);
  Value padding = tosa::buildOnnxToTosaPaddingConstOp(
      rewriter, intValues, loc, {0, 0, 0, 0}, {});
  auto constTosaTensor = tosaBuilder.getSplattedConst(0.0);

  auto inputType = mlir::cast<mlir::TensorType>(input.getType());
  auto padOp = tosa::CreateOpAndInfer<mlir::tosa::PadOp>(rewriter, loc,
      mlir::RankedTensorType::get(
          llvm::SmallVector<int64_t, 4>(
              inputType.getShape().size(), ShapedType::kDynamic),
          inputType.getElementType()),
      input, padding, constTosaTensor);

  // In-place update of AveragePool by setting operand to PadOp
  // and pads attribute to {0, 0, 0, 0}.
  rewriter.modifyOpInPlace(op, [&]() { op->setOperand(0, padOp); });
  rewriter.modifyOpInPlace(op, [&]() {
    op->setAttr("pads", rewriter.getI32ArrayAttr({0, 0, 0, 0}));
  });
}

class ONNXAveragePoolOpLoweringToTOSA
    : public OpConversionPattern<ONNXAveragePoolOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXAveragePoolOp averagePoolOp,
      ONNXAveragePoolOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {

    // With 'countIncludePad' and 'ceilMode', there are 4 cases of attributes
    // that have an impact on the padding of the TOSA AveragePoolOp. The only
    // configuration that is natively supported by TOSA is if both are 0. In
    // that case we will have a 1-1 conversion. If only 'countIncludePad' is
    // set, we pull the padding out of the averagepool op into its own TOSA
    // padOp wih the padding value 0. That way the averagepool treats the padded
    // values as native input values. The padding attribute is then set to 0. If
    // 'ceilMode' is set, we add padding in the pad attribute of the TOSA
    // AveragePoolOp if necessary. Looking into
    // https://github.com/pytorch/pytorch/blob/efc7c366f4fbccf649454726a96df291c8a9df43/aten/src/ATen/native/cuda/AveragePool2d.cu#L243
    // 'ceilMode' does not seem to add implicit padding but just changes the
    // output size. If both are set, we first pull out the padding into its own
    // op and then check if we need to add padding for the ceilMode.

    const int64_t includePad = adaptor.getCountIncludePad();

    if (includePad != 0) {
      // When attribute include_pad is set, create a tosa::PadOp before lowering
      // AveragePool to TOSA. We use ONNX format for it so that the AveragePool
      // lowering still generates transposes between ONNX and TOSA formats, and
      // implementation doesn't diverge much. This will modify the original onnx
      // op.
      handleIncludePadAttr(rewriter, averagePoolOp, adaptor.getX());
    }

    FailureOr<Value> newAveragePoolOp =
        tosa::convertPoolOp<ONNXAveragePoolOp, mlir::tosa::AvgPool2dOp>(
            rewriter, averagePoolOp);

    if (failed(newAveragePoolOp)) {
      return rewriter.notifyMatchFailure(
          averagePoolOp, "Could not create averagepool op.");
    }
    rewriter.replaceOp(averagePoolOp, *newAveragePoolOp);
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
