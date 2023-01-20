/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ ScfParallelEmpty.cpp - Lower ScfParallelOpEmpty -----------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ScfParallelOpEmpty operator.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

/*struct ScfParallelOpEmptyLowering : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

   LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
// LogicalResult matchAndRewrite(scf::ParallelOp parallelOp,
//                                PatternRewriter &rewriter) const override {

    fprintf(stderr,"in empty parallel pattern\n");
    //dump(parallelOp);    
 
    return success();
  }
};*/


class ScfParallelOpEmptyLowering : public ConversionPattern {
public:
  explicit ScfParallelOpEmptyLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, scf::ParallelOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    fprintf(stderr,"in empty parallel pattern\n");
    op->dump();    
    return success();
  }
};

namespace onnx_mlir {
namespace krnl {
void populateLoweringScfParallelOpEmptyPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ScfParallelOpEmptyLowering>(typeConverter, ctx);
}
}
}
