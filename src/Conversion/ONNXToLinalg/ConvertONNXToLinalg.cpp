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

struct ConvertONNXToLinalgPass
    : public PassWrapper<ConvertONNXToLinalgPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertONNXToLinalgPass)

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  ConvertONNXToLinalgPass() = default;
  ConvertONNXToLinalgPass(const ConvertONNXToLinalgPass &pass)
      : PassWrapper<ConvertONNXToLinalgPass, OperationPass<func::FuncOp>>() {}
  ConvertONNXToLinalgPass(const std::string &linalgOps, bool useLinalgPath) {
    this->linalgOps = linalgOps;
    this->useLinalgPath = useLinalgPath;
  }

  StringRef getArgument() const override { return "convert-onnx-to-linalg"; }

  StringRef getDescription() const override {
    return "Lower ONNX operations to Linalg dialect";
  }

  void runOnOperation() override {
    auto function = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    TypeConverter typeConverter;

    // Populate lowering patterns with pass options
    populateLoweringONNXMatMulOpToLinalgPattern(patterns, typeConverter,
        context, linalgOps.getValue(), useLinalgPath.getValue());

    // Apply patterns greedily
    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(function, std::move(patterns), config))) {
      signalPassFailure();
    }
  }

public:
  // Option to specify which ONNX operations should be lowered to Linalg
  // dialect. Operations are specified as a comma-separated list or regex
  // patterns. Example: --convert-onnx-to-linalg='linalg-ops=MatMul,Conv' or
  // --convert-onnx-to-linalg='linalg-ops="MatMul.*"'
  // Special values: ALL (all operations), NONE (no operations).
  // If not specified, defaults to converting all operations.
  Option<std::string> linalgOps{*this, "linalg-ops",
      llvm::cl::desc(
          "Specify which ONNX operations should be lowered to Linalg dialect.\n"
          "Operations are specified as a comma-separated list or regex "
          "patterns.\n"
          "Example: --convert-onnx-to-linalg='linalg-ops=MatMul,Conv' or\n"
          "--convert-onnx-to-linalg='linalg-ops=\"MatMul.*\"'\n"
          "Special values: ALL (all operations), NONE (no operations).\n"
          "If not specified, defaults to converting all operations."),
      llvm::cl::init("")};

  // Option to enable Linalg path for all operations.
  // This is used when --use-linalg-path is set in the main compiler.
  Option<bool> useLinalgPath{*this, "use-linalg-path",
      llvm::cl::desc("Enable Linalg path for all operations (equivalent to "
                     "--use-linalg-path "
                     "in onnx-mlir)."),
      llvm::cl::init(false)};
};

} // namespace

std::unique_ptr<Pass> createConvertONNXToLinalg() {
  return std::make_unique<ConvertONNXToLinalgPass>();
}

std::unique_ptr<Pass> createConvertONNXToLinalg(
    const std::string &linalgOps, bool useLinalgPath) {
  return std::make_unique<ConvertONNXToLinalgPass>(linalgOps, useLinalgPath);
}

} // namespace onnx_mlir
