/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ Tile.cpp - Tile Op --------------------------===//
//
// Copyright (c) 2024 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX TileOp to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>

#include <cstdint>

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXTileLoweringToTOSA : public OpConversionPattern<ONNXTileOp> {

public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXTileOp::Adaptor;

  LogicalResult matchAndRewrite(ONNXTileOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto inputType = adaptor.getInput().getType();
    if (!onnx_mlir::isRankedShapedType(inputType)) {
      return rewriter.notifyMatchFailure(
          op, "input is not a ranked shaped tensor");
    }

    auto resultElementType = cast<ShapedType>(inputType).getElementType();
    auto newResultElementType =
        getTypeConverter()->convertType(resultElementType);

    if (!isSupportedElementType(newResultElementType)) {
      return rewriter.notifyMatchFailure(
          op, "input/output type is invalid for tosa.tile");
    }

    int64_t inputRank = onnx_mlir::getRank(inputType);
    Type newOutputType = RankedTensorType::get(
        llvm::SmallVector<int64_t>(inputRank, ShapedType::kDynamic),
        newResultElementType);

    // Create the attribute for the repetitions
    DenseIntElementsAttr denseReps;
    if (!matchPattern(op.getRepeats(), m_Constant(&denseReps))) {
      return rewriter.notifyMatchFailure(
          op, "onnx.tile can only be lowered with constant repetitions");
    }
    auto newReps = rewriter.getDenseI64ArrayAttr(
        llvm::to_vector(denseReps.getValues<int64_t>()));

    onnx_mlir::tosa::CreateReplaceOpAndInfer<mlir::tosa::TileOp>(
        rewriter, op, newOutputType, adaptor.getInput(), newReps);
    return success();
  }

private:
  static bool isSupportedElementType(Type type) {
    if (auto intTy = dyn_cast_or_null<IntegerType>(type)) {
      // Supported integer bit widths
      std::set<unsigned> intWidth({8, 16, 32});
      return isTOSABool(type) ||
             (intTy.isSignless() &&
                 (intWidth.find(intTy.getWidth()) != intWidth.end()));
    }
    return type.isBF16() || type.isF16() || type.isF32();
  }
};

} // namespace

void populateLoweringONNXTileOpToTOSAPattern(ConversionTarget & /*target*/,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXTileLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
