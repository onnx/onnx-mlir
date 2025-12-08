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

struct ConvertONNXToLinalgPass
    : public PassWrapper<ConvertONNXToLinalgPass, OperationPass<func::FuncOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertONNXToLinalgPass)
  
  StringRef getArgument() const override { return "convert-onnx-to-linalg"; }
  
  StringRef getDescription() const override {
    return "Lower ONNX operations to Linalg dialect";
  }

  void runOnOperation() override {
    auto function = getOperation();
    MLIRContext *context = &getContext();
    
    llvm::errs() << "[DEBUG] ConvertONNXToLinalgPass: Running on function " 
                 << function.getName() << "\n";
    
    // Debug: Check if there are any ONNX ops to convert
    bool hasONNXOps = false;
    int onnxOpCount = 0;
    function.walk([&](Operation *op) {
      if (isa<ONNXMatMulOp>(op)) {
        hasONNXOps = true;
        onnxOpCount++;
        llvm::errs() << "[DEBUG] ConvertONNXToLinalgPass: Found ONNXMatMulOp at " 
                     << op->getLoc() << "\n";
      }
      return WalkResult::advance();
    });
    
    llvm::errs() << "[DEBUG] ConvertONNXToLinalgPass: Found " << onnxOpCount 
                 << " ONNXMatMulOp(s)\n";
    
    if (!hasONNXOps) {
      // No ONNX ops to convert, skip this pass
      llvm::errs() << "[DEBUG] ConvertONNXToLinalgPass: No ONNX ops found, skipping\n";
      return;
    }
    
    RewritePatternSet patterns(context);
    TypeConverter typeConverter;
    
    // Populate lowering patterns
    populateLoweringONNXMatMulOpToLinalgPattern(patterns, typeConverter, context);
    
    llvm::errs() << "[DEBUG] ConvertONNXToLinalgPass: Applying " 
                 << patterns.getNativePatterns().size() << " patterns\n";
    
    // Apply patterns greedily
    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(function, std::move(patterns), config))) {
      llvm::errs() << "[DEBUG] ConvertONNXToLinalgPass: Pattern application failed\n";
      signalPassFailure();
      return;
    }
    
    // Debug: Check if conversion was successful
    int linalgOpCount = 0;
    function.walk([&](Operation *op) {
      if (isa<linalg::MatmulOp>(op)) {
        linalgOpCount++;
      }
      return WalkResult::advance();
    });
    
    llvm::errs() << "[DEBUG] ConvertONNXToLinalgPass: After conversion, found " 
                 << linalgOpCount << " linalg.matmul op(s)\n";
  }
};

} // namespace

std::unique_ptr<Pass> createConvertONNXToLinalgPass() {
  return std::make_unique<ConvertONNXToLinalgPass>();
}

} // namespace onnx_mlir

