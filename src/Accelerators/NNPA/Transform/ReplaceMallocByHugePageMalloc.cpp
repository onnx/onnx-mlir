//===-- ReplaceMallocByHugePageMalloc.cpp - Replace malloc ---------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This pass replaces llvm.call @malloc with llvm.call @OMHugePageMalloc.
// Buffers that the model frees itself are routed to the runtime's memory
// cache instead (OMCacheAlloc / OMCacheFreeSized).
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
///
/// When the memory cache is enabled, a malloc whose result is passed directly
/// to free, and otherwise only converted to an integer (to compute the aligned
/// pointer), is a temporary of the model: it is not returned, stored, or
/// passed to another call. The pair is then rewritten to use the cache:
///   %0 = llvm.call @OMCacheAlloc(%size) : (i64) -> !llvm.ptr
///   ...
///   llvm.call @OMCacheFreeSized(%0, %size) : (!llvm.ptr, i64) -> ()
/// The free also receives the allocation size (%size dominates the free, since
/// it is defined before the malloc whose result the free uses), so that the
/// runtime frees sizes the cache never serves without taking its lock.
/// With the cache enabled, the pass also delimits each inference for the
/// cache's epoch policy: in every entry point (run_*), the call to the model's
/// body (_mlir_ciface_*) is wrapped as
///   %e = llvm.call @OMCacheEpoch(%zero) : (i64) -> i64
///   llvm.call @_mlir_ciface_main_graph(...)
///   llvm.call @OMCacheEpoch(%e) : (i64) -> i64
/// Model outputs keep using OMHugePageMalloc, so that the host can free them
/// with free(). Allocations inside OpenMP regions are left alone: they are
/// small per-thread buffers that glibc already reuses well.

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
  ReplaceMallocByHugePageMallocPass() = default;
  ReplaceMallocByHugePageMallocPass(
      const ReplaceMallocByHugePageMallocPass &pass)
      : PassWrapper<ReplaceMallocByHugePageMallocPass,
            OperationPass<ModuleOp>>() {}
  ReplaceMallocByHugePageMallocPass(bool enableMemoryCache) {
    this->enableMemoryCache = enableMemoryCache;
  }

  Option<bool> enableMemoryCache{*this, "enable-memory-cache",
      llvm::cl::desc("Use OMCacheAlloc/OMCacheFreeSized for buffers that the "
                     "model frees itself"),
      llvm::cl::init(true)};

  Statistic numCacheAllocs{
      this, "cache-allocs", "Number of malloc calls rewritten to OMCacheAlloc"};
  Statistic numEpochWraps{this, "cache-epoch-wraps",
      "Number of entry-point calls wrapped with OMCacheEpoch"};
  Statistic numCacheFrees{this, "cache-frees",
      "Number of free calls rewritten to OMCacheFreeSized"};

  StringRef getArgument() const override {
    return "replace-malloc-by-hugepage-malloc";
  }

  StringRef getDescription() const override {
    return "Replace malloc with OMHugePageMalloc.";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Route the buffers that the model frees itself to the memory cache. The
    // remaining malloc calls are rewritten to OMHugePageMalloc below.
    if (enableMemoryCache) {
      replaceTemporaryMallocByCacheAlloc(moduleOp);
      wrapEntryPointsWithCacheEpoch(moduleOp);
    }

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

private:
  static bool isCallTo(LLVM::CallOp callOp, StringRef name) {
    FlatSymbolRefAttr calleeAttr = callOp.getCalleeAttr();
    return calleeAttr && calleeAttr.getValue() == name;
  }

  static bool isInsideOpenMP(Operation *op) {
    for (Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp()) {
      Dialect *dialect = parent->getDialect();
      if (dialect && dialect->getNamespace() == "omp")
        return true;
    }
    return false;
  }

  // Return the free calls of a malloc result if the buffer is a temporary of
  // the model, that is, if the result is passed directly to free and its only
  // other uses are conversions to an integer. Return an empty list otherwise.
  static SmallVector<LLVM::CallOp, 2> getFreesOfTemporary(
      LLVM::CallOp mallocOp) {
    SmallVector<LLVM::CallOp, 2> frees;
    for (Operation *user : mallocOp.getResult().getUsers()) {
      auto userCall = mlir::dyn_cast<LLVM::CallOp>(user);
      if (userCall && isCallTo(userCall, "free") &&
          userCall.getNumOperands() == 1) {
        frees.emplace_back(userCall);
        continue;
      }
      if (mlir::isa<LLVM::PtrToIntOp>(user))
        continue;
      // Any other use (descriptor, store, return, call) may let it escape.
      return {};
    }
    return frees;
  }

  void replaceTemporaryMallocByCacheAlloc(ModuleOp moduleOp) {
    SmallVector<std::pair<LLVM::CallOp, SmallVector<LLVM::CallOp, 2>>> pairs;
    moduleOp.walk([&](LLVM::CallOp callOp) {
      if (!isCallTo(callOp, "malloc") || callOp.getNumOperands() != 1 ||
          callOp.getNumResults() != 1 ||
          !mlir::isa<LLVM::LLVMPointerType>(callOp.getResult().getType()))
        return;
      if (isInsideOpenMP(callOp))
        return;
      SmallVector<LLVM::CallOp, 2> frees = getFreesOfTemporary(callOp);
      if (!frees.empty())
        pairs.emplace_back(callOp, std::move(frees));
    });
    if (pairs.empty())
      return;

    // Declare OMCacheAlloc and OMCacheFreeSized.
    MLIRContext *context = &getContext();
    OpBuilder builder(context);
    builder.setInsertionPointToStart(moduleOp.getBody());
    LLVMBuilder createLLVM(builder, moduleOp.getLoc());
    IntegerType i64Type = IntegerType::get(context, 64);
    LLVM::LLVMPointerType ptrType = LLVM::LLVMPointerType::get(context);
    FlatSymbolRefAttr cacheAllocRef = createLLVM.getOrInsertSymbolRef(
        moduleOp, "OMCacheAlloc", ptrType, {i64Type});
    FlatSymbolRefAttr cacheFreeRef =
        createLLVM.getOrInsertSymbolRef(moduleOp, "OMCacheFreeSized",
            LLVM::LLVMVoidType::get(context), {ptrType, i64Type});

    for (auto &[mallocOp, frees] : pairs) {
      // OMCacheAlloc has the signature of malloc: only the callee changes.
      mallocOp.setCalleeAttr(cacheAllocRef);
      numCacheAllocs++;
      // Each free becomes OMCacheFreeSized(ptr, size), with the malloc's size.
      Value size = mallocOp.getOperand(0);
      for (LLVM::CallOp freeOp : frees) {
        OpBuilder freeBuilder(freeOp);
        LLVM::CallOp::create(freeBuilder, freeOp.getLoc(), TypeRange{},
            cacheFreeRef, ValueRange{freeOp.getOperand(0), size});
        freeOp.erase();
        numCacheFrees++;
      }
    }
  }

  // Delimit each inference for the cache's epoch policy: in every entry point
  // (run_*), wrap the call to the model's body (_mlir_ciface_*) with
  // OMCacheEpoch(0) before (returns the epoch id) and OMCacheEpoch(id) after.
  // Done whenever the cache is enabled, since zdnnx also uses it.
  void wrapEntryPointsWithCacheEpoch(ModuleOp moduleOp) {
    SmallVector<LLVM::CallOp, 2> bodyCalls;
    moduleOp.walk([&](LLVM::CallOp callOp) {
      FlatSymbolRefAttr callee = callOp.getCalleeAttr();
      if (!callee || !callee.getValue().starts_with("_mlir_ciface_"))
        return;
      auto func = callOp->getParentOfType<LLVM::LLVMFuncOp>();
      if (func && func.getName().starts_with("run_"))
        bodyCalls.emplace_back(callOp);
    });
    if (bodyCalls.empty())
      return;

    MLIRContext *context = &getContext();
    OpBuilder builder(context);
    builder.setInsertionPointToStart(moduleOp.getBody());
    LLVMBuilder createLLVM(builder, moduleOp.getLoc());
    IntegerType i64Type = IntegerType::get(context, 64);
    FlatSymbolRefAttr epochRef = createLLVM.getOrInsertSymbolRef(
        moduleOp, "OMCacheEpoch", i64Type, {i64Type});

    for (LLVM::CallOp bodyCall : bodyCalls) {
      OpBuilder callBuilder(bodyCall);
      LLVMBuilder createCall(callBuilder, bodyCall.getLoc());
      Value zero = createCall.constant(i64Type, static_cast<int64_t>(0));
      Value epoch = createCall.call({i64Type}, epochRef, {zero});
      callBuilder.setInsertionPointAfter(bodyCall);
      createCall.call({i64Type}, epochRef, {epoch});
      numEpochWraps++;
    }
  }
};

std::unique_ptr<Pass> createReplaceMallocByHugePageMallocPass(
    bool enableMemoryCache) {
  return std::make_unique<ReplaceMallocByHugePageMallocPass>(enableMemoryCache);
}

} // namespace onnx_mlir
