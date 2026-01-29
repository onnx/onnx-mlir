//===- ConvertONNXToLinalg.cpp - ONNX dialects to Linalg lowering --------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of frontend operations to Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/CommandLine.h"

#include "src/Conversion/ONNXToLinalg/ONNXToLinalgCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// ONNX to Linalg Lowering Pass
//===----------------------------------------------------------------------===//

namespace {

#define GEN_PASS_DECL_CONVERTONNXTOLINALG
#define GEN_PASS_DEF_CONVERTONNXTOLINALG
#include "src/Conversion/ONNXToLinalg/Passes.h.inc"

struct ConvertONNXToLinalgPass
    : public impl::ConvertONNXToLinalgBase<ConvertONNXToLinalgPass> {
  using ConvertONNXToLinalgBase::ConvertONNXToLinalgBase;

  void runOnOperation() override {
    auto function = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    TypeConverter typeConverter;

    // Populate lowering patterns with pass options
    // TableGen-generated options can be used directly as their types
    populateLoweringONNXMatMulOpToLinalgPattern(
        patterns, typeConverter, context, linalgOps, useLinalgPath);
    populateLoweringONNXConvOpToLinalgPattern(
        patterns, typeConverter, context, linalgOps, useLinalgPath);

    // Apply patterns greedily
    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(function, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createConvertONNXToLinalg() {
  return std::make_unique<ConvertONNXToLinalgPass>();
}

std::unique_ptr<Pass> createConvertONNXToLinalg(
    const std::string &linalgOps, bool useLinalgPath) {
  // Use TableGen-generated options structure
  ConvertONNXToLinalgOptions options;
  options.linalgOps = linalgOps;
  options.useLinalgPath = useLinalgPath;
  return std::make_unique<ConvertONNXToLinalgPass>(options);
}

} // namespace onnx_mlir
