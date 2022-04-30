
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
    KrnlCallOpAdaptor krnlCallAdaptor(operands);
    auto loc = op->getLoc();
    KrnlCallOp krnlCallOp = llvm::dyn_cast<KrnlCallOp>(op);

    // Get a symbol reference to the function, inserting it if necessary.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    llvm::SmallVector<Type, 4> parameterTypeList;
    llvm::SmallVector<Value, 4> parameterList;
    handleOneParameter(rewriter, op, krnlCallAdaptor.result(), parameterTypeList, parameterList);
    for(auto parameter : krnlCallAdaptor.parameters()) {
      handleOneParameter(rewriter, op, parameter, parameterTypeList, parameterList);
    }

    auto callRef = getOrInsertCall(rewriter, module, krnlCallOp.funcName(), parameterTypeList);
    auto voidTy = LLVM::LLVMVoidType::get(context);
    rewriter.create<CallOp>(loc, callRef, ArrayRef<Type>({}), parameterList);

    rewriter.eraseOp(op);
    return success();
  }

private:
  static void handleOneParameter(PatternRewriter &rewriter,  Operation *op, Value parameter, llvm::SmallVector<Type, 4> &parameterTypeList, llvm::SmallVector<Value, 4> &parameterList) {
    auto *context = op->getContext();
    auto loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    const auto &apiRegistry = RuntimeAPIRegistry::build(module, rewriter);

    Type ty = parameter.getType();
    //if (ty.isa<MemRefType>()) {
      auto int64Ty = IntegerType::get(context, 64);
      auto memRefTy = ty.dyn_cast<LLVM::LLVMStructType>();
      auto memRefRank = krnl::getRankFromMemRefType(memRefTy);
      auto memRefRankVal = rewriter.create<LLVM::ConstantOp>(
          loc, int64Ty, rewriter.getI64IntegerAttr(memRefRank));
      Value omTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
          RuntimeAPI::API::CREATE_OMTENSOR, {memRefRankVal});
  
      krnl::fillOMTensorWithMemRef(parameter, omTensor, false /*outOwning*/, rewriter,
          loc, apiRegistry, module);
      auto int8Ty = IntegerType::get(context, 8);
      auto opaquePtrTy = LLVM::LLVMPointerType::get(int8Ty);
      parameterTypeList.emplace_back(opaquePtrTy);
      parameterList.emplace_back(omTensor);
    //}
  }

  FlatSymbolRefAttr getOrInsertCall(
      PatternRewriter &rewriter, ModuleOp module, llvm::StringRef funcName, ArrayRef<Type> parameterTypeList) const {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
      return SymbolRefAttr::get(context, funcName);
    auto llvmVoidTy = LLVM::LLVMVoidType::get(context);
    auto llvmFnType = LLVM::LLVMFunctionType::get(
        llvmVoidTy, parameterTypeList, false);

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
