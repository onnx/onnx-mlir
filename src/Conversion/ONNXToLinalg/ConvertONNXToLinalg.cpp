/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertONNXToLinalg.cpp - ONNX dialects to Krnl lowering -----===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of frontend operations to a combination of
// Krnl IR and standard operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "src/Compiler/CompilerOptions.hpp"

#include "src/Accelerators/Accelerator.hpp"
#include "src/Conversion/ONNXToLinalg/ONNXToLinalgCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

void populateONNXToLinalgConversionPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {

  // Math
  populateLoweringONNXMatMulOpLinalgPattern(patterns, typeConverter, ctx);
}

//===----------------------------------------------------------------------===//
// ONNX to Krnl Dialect lowering pass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to Krnl loops of the ONNX operations.
struct ONNXToLinalgLoweringPass
    : public PassWrapper<ONNXToLinalgLoweringPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXToLinalgLoweringPass)

  StringRef getArgument() const override { return "convert-onnx-to-linalg"; }

  StringRef getDescription() const override {
    return "Lower ONNX ops to Linalg dialect.";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  ONNXToLinalgLoweringPass() = default;
  ONNXToLinalgLoweringPass(const ONNXToLinalgLoweringPass &pass)
      : PassWrapper<ONNXToLinalgLoweringPass, OperationPass<ModuleOp>>() {}

  void runOnOperation() final;
};

void ONNXToLinalgLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target.addLegalDialect<KrnlDialect, AffineDialect, arith::ArithDialect,
      func::FuncDialect, linalg::LinalgDialect, math::MathDialect,
      memref::MemRefDialect, shape::ShapeDialect, scf::SCFDialect,
      tensor::TensorDialect>();
  // Needed to support unsigned int computations. To be removed if we use a
  // scheme that does not rely on the UnrealizedConversionCastOp.
  target.addLegalOp<::mlir::UnrealizedConversionCastOp>();
  // Make ONNXNoneOp legal so that other ONNX ops can use it during the
  // lowering. ONNXNoneOp will be dangling and removed by calling
  // canonicalization after the lowering.
  target.addLegalOp<::mlir::ONNXNoneOp>();
  target.addLegalOp<linalg::MatmulOp>();
  target.addLegalOp<tensor::EmptyOp>();

  // The following requirements are from Krnl and they are kept if ONNXToKrnl
  // is after this pass.
  // If the Linalg is on tensor instead of memref, this lowering will not
  // generate memref or Affine load/store. However, these requiremnts will may
  // be an issue if Ops are lowered other than Krnl Use krnl.load/store instead
  // of std.load/store and affine.load/store. krnl.load/store will be lowered to
  // std.load/store and affine.load/store by `convert-krnl-to-affine` pass.
  target.addIllegalOp<mlir::memref::LoadOp>();
  target.addIllegalOp<mlir::AffineLoadOp>();
  target.addIllegalOp<mlir::memref::StoreOp>();
  target.addIllegalOp<mlir::AffineStoreOp>();

  target.addIllegalOp<ONNXMatMulOp>();

  // TODO: add any other ops which are considered legal.
  // Some operations can be marked as being still legal.
  // Example: target.addLegalOp<mlir::OpName>();

  // For future: Handle the accelerator target.
  // for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators())
  // accel->conversionTargetONNXToLinalg(target);

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the frontend operations.
  RewritePatternSet patterns(&getContext());

  // Convert types to legal types for the Krnl dialect.
  LinalgTypeConverter linalgTypeConverter;

  // Define patterns.
  populateONNXToLinalgConversionPattern(
      patterns, linalgTypeConverter, &getContext());

  // For future: Rewrite patterns for accelerators.
  // for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators())
  //  accel->rewritePatternONNXToLinalg(patterns, krnlTypeConverter,
  //  &getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createLowerONNXToLinalgPass() {
  return std::make_unique<ONNXToLinalgLoweringPass>();
}

} // namespace onnx_mlir
