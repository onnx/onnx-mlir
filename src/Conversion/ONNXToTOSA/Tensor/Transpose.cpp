/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Tanspose.cpp - Transpose Op --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX TransposeOp to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXTransposeLoweringToTOSA
    : public OpConversionPattern<ONNXTransposeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXTransposeOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXTransposeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    IndexExprBuilderForTosa createTosaIE(rewriter, loc);
    ONNXTransposeOpShapeHelper shapeHelper(op, {}, &createTosaIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    TosaBuilder tosaBuilder(rewriter, loc);

    Value input = adaptor.getData();

    auto inputType = input.getType().dyn_cast<TensorType>();

    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input not a ranked tensor");

    Type inputElementType = inputType.getElementType();

    if (!isTOSAFloat(inputElementType) && !isTOSASignedInt(inputElementType) &&
        !inputElementType.isInteger(1)) {
      return rewriter.notifyMatchFailure(
          op, "input element type not supported");
    }

    auto outputType = op.getResult().getType().dyn_cast<TensorType>();

    if (!outputType)
      return rewriter.notifyMatchFailure(op, "output not a ranked tensor");

    auto permVector = extractFromIntegerArrayAttr<int64_t>(op.getPermAttr());
    // TOSA needs a I32 array
    llvm::SmallVector<int32_t, 4> permVectorI32;
    permVectorI32.clear();
    llvm::transform(permVector, std::back_inserter(permVectorI32),
        [](const auto &valueI64) { return (int32_t)valueI64; });

    Value transposeOp = tosaBuilder.transpose(input, permVectorI32);

    rewriter.replaceOp(op, transposeOp);

    return success();
  }
};

} // namespace

void populateLoweringONNXTransposeOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXTransposeLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir
