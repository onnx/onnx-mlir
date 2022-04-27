
/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlCall.cpp - Lower KrnlCallOp -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlCallOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlCallOpLowering : public ConversionPattern {
public:
  explicit KrnlCallOpLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlCallOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto *context = op->getContext();
    KrnlCallOpAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();
    KrnlCallOp krnlCallOp = llvm::dyn_cast<KrnlCallOp>(op);

    // Get a symbol reference to the function, inserting it if necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    llvm::SmallVector<Type, 4> parameterTypeList;
    llvm::SmallVector<Value, 4> parameterList;
    auto callRef = getOrInsertCall(rewriter, parentModule, krnlCallOp.funcName());

    rewriter.create<CallOp>(loc, callRef, parameterTypeList, parameterList);

    rewriter.eraseOp(op);
    return success();
  }

private:
  FlatSymbolRefAttr getOrInsertCall(
      PatternRewriter &rewriter, ModuleOp module, llvm::StringRef funcName) const {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
      return SymbolRefAttr::get(context, funcName);
    auto llvmVoidTy = LLVM::LLVMVoidType::get(context);
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto llvmFnType = LLVM::LLVMFunctionType::get(
        llvmVoidTy, ArrayRef<mlir::Type>({llvmI64Ty, llvmI64Ty}), false);

    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, llvmFnType);
    return SymbolRefAttr::get(context, funcName);
  }
};

void populateLoweringKrnlCallOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlCallOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
