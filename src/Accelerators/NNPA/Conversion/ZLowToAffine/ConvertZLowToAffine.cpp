/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertZLowToAffine.cpp - ZLow Dialect Lowering --------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of ZLow operations to the affine dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/KrnlToAffine/ConvertKrnlToAffine.hpp"

#include "src/Accelerators/NNPA/Conversion/ZLowToAffine/ConvertZLowToAffine.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "zlow_to_affine"

using namespace mlir;

namespace onnx_mlir {
namespace zlow {

//===----------------------------------------------------------------------===//
// ConvertZLowToAffinePass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the krnl dialect operations.
/// At this stage the dialect will contain standard operations as well like
/// add and multiply, this pass will leave these operations intact.
struct ConvertZLowToAffinePass
    : public PassWrapper<ConvertZLowToAffinePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertZLowToAffinePass);

  StringRef getArgument() const override { return "convert-zlow-to-affine"; }

  StringRef getDescription() const override { return "Lower ZLow dialect."; }

  void runOnOperation() final;
};

void ConvertZLowToAffinePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (funcOp.getBody().empty()) // external function: nothing to do
    return;

  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);
  onnx_mlir::krnl::AffineTypeConverter typeConverter;

  ConversionTarget target(*ctx);
  target.addLegalDialect<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
      mlir::memref::MemRefDialect, mlir::func::FuncDialect,
      mlir::vector::VectorDialect, onnx_mlir::zlow::ZLowDialect,
      mlir::KrnlDialect>();

  // These ops will be lowered to affine.
  target.addIllegalOp<ZLowConvertDLF16Op>();

  // Patterns.
  RewritePatternSet patterns(ctx);
  populateZLowToAffineConversion(typeConverter, patterns, ctx);

  if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
    signalPassFailure();
    return;
  }
}

std::unique_ptr<Pass> createConvertZLowToAffinePass() {
  return std::make_unique<ConvertZLowToAffinePass>();
}

void populateZLowToAffineConversion(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  populateLoweringZLowConvertDLF16OpPattern(typeConverter, patterns, ctx);
}

} // namespace zlow
} // namespace onnx_mlir
