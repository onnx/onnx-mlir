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

    int64_t inputRank = onnx_mlir::getRank(inputType);
    auto newResultElementType = cast<ShapedType>(inputType).getElementType();
    Type newOutputType = RankedTensorType::get(
        llvm::SmallVector<int64_t>(inputRank, ShapedType::kDynamic),
        newResultElementType);

    // Create the attribute for the repetitions
    Value reps = adaptor.getRepeats();
    auto repsConstant =
        dyn_cast_or_null<mlir::tosa::ConstOp>(reps.getDefiningOp());
    if (!repsConstant) {
      return rewriter.notifyMatchFailure(
          op, "onnx.tile can only be lowered with constant repetitions");
    }
    auto denseReps = repsConstant->getAttrOfType<DenseElementsAttr>("value");
    llvm::SmallVector<int64_t> vals;
    for (auto val : denseReps.getValues<int64_t>()) {
      vals.push_back(val);
    }
    auto newReps = rewriter.getDenseI64ArrayAttr(vals);

    tosa::CreateReplaceOpAndInfer<mlir::tosa::TileOp>(
        rewriter, op, newOutputType, adaptor.getInput(), newReps);
    return success();
  }
};

} // namespace

void populateLoweringONNXTileOpToTOSAPattern(ConversionTarget & /*target*/,
    RewritePatternSet &patterns, TypeConverter & /*typeConverter*/,
    MLIRContext *ctx) {
  patterns.insert<ONNXTileLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir
