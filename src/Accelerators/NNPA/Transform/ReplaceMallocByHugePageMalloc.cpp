//===-- ReplaceMallocByHugePageMalloc.cpp - Replace malloc ---------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This pass replaces llvm.call @malloc with llvm.call @OMHugePageMalloc.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Pass/NNPAPasses.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {

/// This pass replaces llvm.call @malloc with llvm.call @OMHugePageMalloc.
///
/// Example:
/// The following code:
///   %0 = llvm.call @malloc(%size) : (i64) -> !llvm.ptr
///
/// will be replaced by:
///   %0 = llvm.call @OMHugePageMalloc(%size) : (i64) -> !llvm.ptr

/// Replace malloc with OMHugePageMalloc.
class ReplaceMallocPattern : public OpRewritePattern<LLVM::CallOp> {
public:
  using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      LLVM::CallOp callOp, PatternRewriter &rewriter) const override {
    Location loc = callOp.getLoc();

    // 1. Match: Check if this is a call to malloc.
    FlatSymbolRefAttr calleeAttr = callOp.getCalleeAttr();
    if (!calleeAttr)
      return failure();

    StringRef calleeName = calleeAttr.getValue();
    if (calleeName != "malloc")
      return failure();

    // Verify the signature: (i64) -> !llvm.ptr
    if (callOp.getNumOperands() != 1)
      return failure();

    if (callOp.getNumResults() != 1)
      return failure();

    Value result = callOp.getResult();
    Type resultType = result.getType();
    if (!mlir::isa<LLVM::LLVMPointerType>(resultType))
      return failure();

    // 2. Rewrite: Replace with call to OMHugePageMalloc.
    LLVMBuilder createLLVM(rewriter, loc);
    llvm::SmallVector<mlir::Value, 1> operands(
        callOp.getOperands().begin(), callOp.getOperands().end());
    Value newCall = createLLVM.call({resultType}, "OMHugePageMalloc", operands);

    rewriter.replaceOp(callOp, newCall);

    return success();
  }
};

/*!
 *  Module pass that replaces malloc with OMHugePageMalloc.
 */
class ReplaceMallocByHugePageMallocPass
    : public PassWrapper<ReplaceMallocByHugePageMallocPass,
          OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const override {
    return "replace-malloc-by-hugepage-malloc";
  }

  StringRef getDescription() const override {
    return "Replace malloc with OMHugePageMalloc.";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Declare OMHugePageMalloc function if not already present.
    OpBuilder builder(&getContext());
    builder.setInsertionPointToStart(moduleOp.getBody());
    LLVMBuilder createLLVM(builder, moduleOp.getLoc());

    // Get or insert OMHugePageMalloc function declaration.
    // Signature: (i64) -> !llvm.ptr
    IntegerType i64Type = IntegerType::get(&getContext(), 64);
    LLVM::LLVMPointerType ptrType = LLVM::LLVMPointerType::get(&getContext());
    createLLVM.getOrInsertSymbolRef(
        moduleOp, "OMHugePageMalloc", ptrType, {i64Type});

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<ReplaceMallocPattern>(&getContext());

    static_cast<void>(applyPatternsGreedily(moduleOp, std::move(patterns)));
  }
};

std::unique_ptr<Pass> createReplaceMallocByHugePageMallocPass() {
  return std::make_unique<ReplaceMallocByHugePageMallocPass>();
}

} // namespace onnx_mlir
