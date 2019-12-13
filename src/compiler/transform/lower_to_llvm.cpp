//====- LowerToLLVM.cpp - Lowering from KRNL+Affine+Std to LLVM -----------===//
//
// Copyright 2019 The IBM Research Authors.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Sequence.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/compiler/dialect/krnl/krnl_ops.hpp"
#include "src/compiler/pass/passes.hpp"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// KRNL to LLVM: patterns which need a direct lowering to LLVM.
//===----------------------------------------------------------------------===//

class KrnlMemcpyOpLowering : public ConversionPattern {
 public:
  explicit KrnlMemcpyOpLowering(MLIRContext* context)
      : ConversionPattern(KrnlMemcpyOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation* op, ArrayRef<Value*> operands,
      ConversionPatternRewriter& rewriter) const override {
    auto* context = op->getContext();
    auto loc = op->getLoc();
    auto* llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    assert(llvmDialect && "expected llvm dialect to be registered");

    // Get a symbol reference to the memcpy function, inserting it if necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto memcpyRef = getOrInsertMemcpy(rewriter, parentModule, llvmDialect);

    // First operand.
    Type dstType =
        operands[0]->getType().cast<LLVM::LLVMType>().getStructElementType(1);
    Value* alignedDstMemory = rewriter.create<LLVM::ExtractValueOp>(
        loc, dstType, operands[0], rewriter.getI64ArrayAttr(1));
    Value* alignedInt8PtrDstMemory = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMType::getInt8PtrTy(llvmDialect), alignedDstMemory);

    // Second operand.
    Type srcType =
        operands[1]->getType().cast<LLVM::LLVMType>().getStructElementType(1);
    Value* alignedSrcMemory = rewriter.create<LLVM::ExtractValueOp>(
        loc, srcType, operands[1], rewriter.getI64ArrayAttr(1));
    Value* alignedInt8PtrSrcMemory = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMType::getInt8PtrTy(llvmDialect), alignedSrcMemory);

    // Size.
    Value* int64Size = rewriter.create<LLVM::SExtOp>(
        loc, LLVM::LLVMType::getInt64Ty(llvmDialect), operands[2]);

    // Memcpy call
    rewriter.create<CallOp>(loc, memcpyRef,
        LLVM::LLVMType::getVoidTy(llvmDialect),
        ArrayRef<Value*>(
            {alignedInt8PtrDstMemory, alignedInt8PtrSrcMemory, int64Size}));

    rewriter.eraseOp(op);
    return matchSuccess();
  }

 private:
  /// Return a symbol reference to the memcpy function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertMemcpy(PatternRewriter& rewriter,
      ModuleOp module, LLVM::LLVMDialect* llvmDialect) {
    auto* context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("llvm.memcpy.p0i8.p0i8.i64"))
      return SymbolRefAttr::get("llvm.memcpy.p0i8.p0i8.i64", context);
    // Create a function declaration for memcpy, the signature is:
    //   * `void (i8*, i8* , i64, i1)`
    auto llvmVoidTy = LLVM::LLVMType::getVoidTy(llvmDialect);
    auto llvmI8PtrTy = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    auto llvmI64Ty = LLVM::LLVMType::getInt64Ty(llvmDialect);
    auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmVoidTy,
        ArrayRef<mlir::LLVM::LLVMType>({llvmI8PtrTy, llvmI8PtrTy, llvmI64Ty}),
        false);

    // Insert the memcpy function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(
        module.getLoc(), "llvm.memcpy.p0i8.p0i8.i64", llvmFnType);
    return SymbolRefAttr::get("llvm.memcpy.p0i8.p0i8.i64", context);
  }
};
}  // end namespace

//===----------------------------------------------------------------------===//
// KRNL + Stadard + Affine dialects lowering to LLVM.
//===----------------------------------------------------------------------===//

namespace {
struct KrnlToLLVMLoweringPass : public ModulePass<KrnlToLLVMLoweringPass> {
  void runOnModule() final;
};
}  // end anonymous namespace

void KrnlToLLVMLoweringPass::runOnModule() {
  // Define the target for this lowering i.e. the LLVM dialect.
  ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  // Lower the MemRef types to a representation in LLVM.
  LLVMTypeConverter typeConverter(&getContext());

  // We have a combination of `krnl`, `affine`, and `std` operations. We
  // lower in stages until all the code is in the LLVM dialect.
  OwningRewritePatternList patterns;
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  // Lower from the `krnl` dialect i.e. the Reshape operation.
  patterns.insert<KrnlMemcpyOpLowering>(&getContext());

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  auto module = getModule();
  if (failed(applyFullConversion(module, target, patterns, &typeConverter)))
    signalPassFailure();
}

/// Create the pass for lowering `Krnl`, `Affine` and `Std` dialects to LLVM.
std::unique_ptr<mlir::Pass> mlir::createKrnlLowerToLLVMPass() {
  return std::make_unique<KrnlToLLVMLoweringPass>();
}

static PassRegistration<KrnlToLLVMLoweringPass> pass(
    "lower-all-llvm", "Lower the Krnl Affine and Std dialects to LLVM.");
