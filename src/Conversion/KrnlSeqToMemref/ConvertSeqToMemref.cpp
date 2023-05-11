/*
 * SPDX-License-Identifier: Apache-2.0
 */
//====------ ConvertSeqToMemref.cpp - ONNX dialects to Krnl lowering
//-------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of frontend operations to a combination of
// Krnl IR and standard operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Pass/Pass.h"

#include "src/Conversion/KrnlSeqToMemref/ConvertSeqToMemref.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

struct ConvertSeqToMemrefPass
    : public PassWrapper<ConvertSeqToMemrefPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertSeqToMemrefPass)

  StringRef getArgument() const override { return "convert-seq-to-memref"; }

  StringRef getDescription() const override {
    return "Lower Krnl Seq ops to memref dialect.";
  }

  void runOnOperation() final;
};

void ConvertSeqToMemrefPass::runOnOperation() {
  mlir::func::FuncOp funcOp = getOperation();
  if (funcOp.getBody().empty()) // external function: nothing to do
    return;
  MLIRContext *ctx = &getContext();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  target.addIllegalOp<KrnlSeqAllocOp>();
  target.addIllegalOp<KrnlSeqDeallocOp>();
  target.addIllegalOp<KrnlSeqExtractOp>();
  target.addIllegalOp<KrnlSeqStoreOp>();
  target.addLegalDialect<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
      mlir::memref::MemRefDialect, mlir::func::FuncDialect,
      mlir::vector::VectorDialect, mlir::scf::SCFDialect>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the frontend operations.
  RewritePatternSet patterns(ctx);

  // Define patterns.
  KrnlTypeConverter typeConverter;
  populateLoweringKrnlSeqAllocOpPattern(typeConverter, patterns, ctx);
  populateLoweringKrnlSeqDeallocOpPattern(typeConverter, patterns, ctx);
  populateLoweringKrnlSeqExtractOpPattern(typeConverter, patterns, ctx);
  populateLoweringKrnlSeqStoreOpPattern(typeConverter, patterns, ctx);

  if (failed(applyPartialConversion(
          getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createConvertSeqToMemrefPass() {
  return std::make_unique<ConvertSeqToMemrefPass>();
}

} // namespace krnl
} // namespace onnx_mlir
