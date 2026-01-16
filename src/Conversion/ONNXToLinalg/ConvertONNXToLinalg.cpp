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

    // Get the linalg-ops option value
    // Pass::Option<std::string> can be used directly as std::string
    std::string opsList = linalgOps;

    // If linalg-ops option is specified, check if MatMul is included
    // For now, we only support MatMul, so if the option is empty or contains
    // "onnx.MatMul", we add the pattern
    bool shouldLowerMatMul =
        opsList.empty() || opsList.find("onnx.MatMul") != std::string::npos;

    RewritePatternSet patterns(context);
    TypeConverter typeConverter;

    // Populate lowering patterns based on options
    if (shouldLowerMatMul) {
      populateLoweringONNXMatMulOpToLinalgPattern(
          patterns, typeConverter, context);
    }

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

} // namespace onnx_mlir
