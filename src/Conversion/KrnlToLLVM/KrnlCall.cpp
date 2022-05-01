
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
    KrnlCallOpAdaptor krnlCallAdaptor(operands);
    auto loc = op->getLoc();
    KrnlCallOp krnlCallOp = llvm::dyn_cast<KrnlCallOp>(op);

    // Get a symbol reference to the function, inserting it if necessary.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    llvm::SmallVector<Type, 4> parameterTypeList;
    llvm::SmallVector<Value, 4> parameterList;
    handleOneParameter(rewriter, op, krnlCallAdaptor.result(),
        krnlCallOp.result(), parameterTypeList, parameterList);

    // Some type of operands has been converted.
    // It is better to check the type of original operands.
    // Thus, the two kinds of operands are used together.
    auto itConverted = krnlCallAdaptor.parameters().begin();
    auto itOriginal = krnlCallOp.parameters().begin();
    for (; itConverted != krnlCallAdaptor.parameters().end();
         itConverted++, itOriginal++) {
      handleOneParameter(rewriter, op, *itConverted, *itOriginal,
          parameterTypeList, parameterList);
    }

    // Handle the Attributes
    for (auto namedAttr : op->getAttrs()) {
      // Avoid the funcName() Attribute
      if (namedAttr.getName().getValue().equals("funcName"))
        continue;
      handleOneAttribute(rewriter, getTypeConverter(), op, namedAttr.getValue(),
          parameterTypeList, parameterList);
    }

    auto callRef = getOrInsertCall(
        rewriter, module, krnlCallOp.funcName(), parameterTypeList);
    rewriter.create<CallOp>(loc, callRef, ArrayRef<Type>({}), parameterList);

    rewriter.eraseOp(op);
    return success();
  }

private:
  static void handleOneParameter(PatternRewriter &rewriter, Operation *op,
      Value parameter, Value original,
      llvm::SmallVector<Type, 4> &parameterTypeList,
      llvm::SmallVector<Value, 4> &parameterList) {
    auto *context = op->getContext();
    auto loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    const auto &apiRegistry = RuntimeAPIRegistry::build(module, rewriter);

    // Check the original type, not after type conversion
    Type ty = original.getType();
    if (ty.isa<MemRefType>()) {
      auto int64Ty = IntegerType::get(context, 64);
      auto memRefTy = parameter.getType().dyn_cast<LLVM::LLVMStructType>();
      auto memRefRank = krnl::getRankFromMemRefType(memRefTy);
      auto memRefRankVal = rewriter.create<LLVM::ConstantOp>(
          loc, int64Ty, rewriter.getI64IntegerAttr(memRefRank));
      Value omTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
          RuntimeAPI::API::CREATE_OMTENSOR, {memRefRankVal});

      krnl::fillOMTensorWithMemRef(parameter, omTensor, false /*outOwning*/,
          rewriter, loc, apiRegistry, module);
      auto int8Ty = IntegerType::get(context, 8);
      auto opaquePtrTy = LLVM::LLVMPointerType::get(int8Ty);
      parameterTypeList.emplace_back(opaquePtrTy);
      parameterList.emplace_back(omTensor);
    } else {
      parameterTypeList.emplace_back(parameter.getType());
      parameterList.emplace_back(parameter);
    }
  }

  static void handleOneAttribute(PatternRewriter &rewriter,
      TypeConverter *typeConverter, Operation *op, Attribute attribute,
      llvm::SmallVector<Type, 4> &parameterTypeList,
      llvm::SmallVector<Value, 4> &parameterList) {
    auto *context = op->getContext();
    auto loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    if (attribute.isa<StringAttr>()) {
      auto strAttr = attribute.cast<StringAttr>();
      StringRef attrValue = strAttr.getValue();
      LLVM::GlobalOp globalStr = krnl::getOrCreateGlobalString(attrValue, loc,
          rewriter, module, static_cast<LLVMTypeConverter *>(typeConverter));
      Value strPtr = krnl::getPtrToGlobalString(globalStr, loc, rewriter);
      auto int8Ty = IntegerType::get(context, 8);
      auto opaquePtrTy = LLVM::LLVMPointerType::get(int8Ty);
      parameterTypeList.emplace_back(opaquePtrTy);
      parameterList.emplace_back(strPtr);
    } else if (attribute.isa<IntegerAttr>()) {
      auto integerAttr = attribute.cast<IntegerAttr>();
      auto int64Ty = IntegerType::get(context, 64);
      Value cst = rewriter.create<LLVM::ConstantOp>(loc, int64Ty, integerAttr);
      parameterTypeList.emplace_back(int64Ty);
      parameterList.emplace_back(cst);
    } else if (attribute.isa<FloatAttr>()) {
      auto floatAttr = attribute.cast<FloatAttr>();
      auto f64Ty = rewriter.getF64Type();
      Value cst = rewriter.create<LLVM::ConstantOp>(loc, f64Ty, floatAttr);
      parameterTypeList.emplace_back(f64Ty);
      parameterList.emplace_back(cst);
    } else {
      llvm_unreachable(
          "This type of Attribute used by krnl.call has not implemented");
    }
  }

  FlatSymbolRefAttr getOrInsertCall(PatternRewriter &rewriter, ModuleOp module,
      llvm::StringRef funcName, ArrayRef<Type> parameterTypeList) const {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
      return SymbolRefAttr::get(context, funcName);
    auto llvmVoidTy = LLVM::LLVMVoidType::get(context);
    auto llvmFnType =
        LLVM::LLVMFunctionType::get(llvmVoidTy, parameterTypeList, false);

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
