/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- KrnlCall.cpp - Lower KrnlCallOp -----------------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlCallOp operator.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlCallOpLowering : public ConversionPattern {
public:
  explicit KrnlCallOpLowering(
      LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlCallOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    KrnlCallOpAdaptor krnlCallAdaptor(operands);
    Location loc = op->getLoc();
    KrnlCallOp krnlCallOp = llvm::cast<KrnlCallOp>(op);
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    const LLVMTypeConverter *llvmTypeConverter =
        static_cast<const LLVMTypeConverter *>(getTypeConverter());

    // Get a symbol reference to the function, inserting it if necessary.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    llvm::SmallVector<Type, 4> parameterTypeList;
    llvm::SmallVector<Value, 4> parameterList;
    llvm::SmallVector<Value, 4> omTensors;

    // Some type of operands has been converted.
    // It is better to check the type of original operands.
    // Thus, the two kinds of operands are used together.
    auto itConverted = krnlCallAdaptor.getParameters().begin();
    auto itOriginal = krnlCallOp.getParameters().begin();
    for (; itConverted != krnlCallAdaptor.getParameters().end();
         itConverted++, itOriginal++) {
      handleOneParameter(rewriter, op, *itConverted, *itOriginal,
          parameterTypeList, parameterList, omTensors);
    }

    // Handle the Attributes
    for (auto namedAttr : op->getAttrs()) {
      // Avoid the funcName() Attribute
      if (namedAttr.getName().getValue() == "funcName")
        continue;
      if (namedAttr.getName().getValue() == "numOfOutput")
        continue;
      handleOneAttribute(
          rewriter, op, namedAttr.getValue(), parameterTypeList, parameterList);
    }

    ValueRange returns = op->getResults();
    if (returns.size() == 0) {
      // There is no return
      FlatSymbolRefAttr callRef =
          create.llvm.getOrInsertSymbolRef(module, krnlCallOp.getFuncName(),
              LLVM::LLVMVoidType::get(module.getContext()), parameterTypeList);
      create.llvm.call({}, callRef, parameterList);

      rewriter.eraseOp(op);
    } else {
      assert(returns.size() == 1 &&
             "Only one return value is allowed for krnl.call now");
      Type llvmReturnType =
          llvmTypeConverter->convertType(returns[0].getType());

      FlatSymbolRefAttr callRef = create.llvm.getOrInsertSymbolRef(
          module, krnlCallOp.getFuncName(), llvmReturnType, parameterTypeList);
      auto llvmCall =
          create.llvm.call({llvmReturnType}, callRef, parameterList);
      rewriter.replaceOp(op, llvmCall.getDefiningOp()->getResults()[0]);
    }

    // Destroy OMTensor wrappers of parameters.
    const auto &apiRegistry =
        RuntimeAPIRegistry(module, rewriter, *llvmTypeConverter);
    for (Value omt : omTensors) {
      RuntimeAPI::callApi(
          rewriter, loc, apiRegistry, RuntimeAPI::API::DESTROY_OMTENSOR, {omt});
    }

    return success();
  }

private:
  void handleOneParameter(PatternRewriter &rewriter, Operation *op,
      Value parameter, Value original,
      llvm::SmallVector<Type, 4> &parameterTypeList,
      llvm::SmallVector<Value, 4> &parameterList,
      llvm::SmallVector<Value, 4> &omTensors) const {
    MLIRContext *context = op->getContext();
    Location loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    const auto *llvmTypeConverter =
        static_cast<const LLVMTypeConverter *>(getTypeConverter());
    const auto &apiRegistry =
        RuntimeAPIRegistry(module, rewriter, *llvmTypeConverter);

    // Check the original type, not after type conversion
    Type ty = original.getType();
    if (auto originalMemRef = mlir::dyn_cast<MemRefType>(ty)) {
      auto int64Ty = IntegerType::get(context, 64);
      auto memRefTy = mlir::dyn_cast<LLVM::LLVMStructType>(parameter.getType());
      auto memRefRank = krnl::getRankFromMemRefType(memRefTy);
      auto memRefRankVal =
          create.llvm.constant(int64Ty, static_cast<int64_t>(memRefRank));
      Value omTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
          RuntimeAPI::API::CREATE_OMTENSOR, {memRefRankVal});

      Type llvmOrigElemTy =
          llvmTypeConverter->convertType(originalMemRef.getElementType());
      krnl::fillOMTensorWithMemRef(parameter, llvmOrigElemTy, omTensor,
          false /*outOwning*/, rewriter, loc, apiRegistry, module);
      auto int8Ty = IntegerType::get(context, 8);
      auto opaquePtrTy = getPointerType(context, int8Ty);
      parameterTypeList.emplace_back(opaquePtrTy);
      parameterList.emplace_back(omTensor);
      omTensors.emplace_back(omTensor);
    } else if (mlir::isa<NoneType>(ty)) {
      // Generate llvm null pinter for NoneType
      auto int8Ty = IntegerType::get(context, 8);
      auto opaquePtrTy = getPointerType(context, int8Ty);
      parameterTypeList.emplace_back(opaquePtrTy);
      Value nullPtr = create.llvm.null(opaquePtrTy);
      parameterList.emplace_back(nullPtr);
    } else {
      parameterTypeList.emplace_back(parameter.getType());
      parameterList.emplace_back(parameter);
    }
  }

  void handleOneAttribute(PatternRewriter &rewriter, Operation *op,
      Attribute attribute, llvm::SmallVector<Type, 4> &parameterTypeList,
      llvm::SmallVector<Value, 4> &parameterList) const {
    auto *context = op->getContext();
    Location loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MultiDialectBuilder<KrnlBuilder, LLVMBuilder> create(rewriter, loc);
    const LLVMTypeConverter *llvmTypeConverter =
        static_cast<const LLVMTypeConverter *>(getTypeConverter());
    const auto &apiRegistry =
        RuntimeAPIRegistry(module, rewriter, *llvmTypeConverter);

    TypeSwitch<Attribute>(attribute)
        .Case<StringAttr>([&](StringAttr strAttr) {
          StringRef attrValue = strAttr.getValue();
          LLVM::GlobalOp globalStr = krnl::getOrCreateGlobalString(
              attrValue, loc, rewriter, module, llvmTypeConverter);
          Value strPtr = krnl::getPtrToGlobalString(globalStr, loc, rewriter);
          auto int8Ty = IntegerType::get(context, 8);
          auto opaquePtrTy = getPointerType(context, int8Ty);
          parameterTypeList.emplace_back(opaquePtrTy);
          parameterList.emplace_back(strPtr);
        })
        .Case<IntegerAttr>([&](IntegerAttr integerAttr) {
          auto int64Ty = IntegerType::get(context, 64);
          Value cst =
              rewriter.create<LLVM::ConstantOp>(loc, int64Ty, integerAttr);
          parameterTypeList.emplace_back(int64Ty);
          parameterList.emplace_back(cst);
        })
        .Case<FloatAttr>([&](FloatAttr floatAttr) {
          auto f64Ty = rewriter.getF64Type();
          Value cst = rewriter.create<LLVM::ConstantOp>(loc, f64Ty,
              rewriter.getFloatAttr(f64Ty, floatAttr.getValueAsDouble()));
          parameterTypeList.emplace_back(f64Ty);
          parameterList.emplace_back(cst);
        })
        .Case<DenseElementsAttr>([&](DenseElementsAttr denseAttr) {
          // Use krnl.global to handle it
          // Since the attribute is still in tensor type, the code has to cross
          // onnx to krnl, and krnl to llvm.
          // In future, the attributes should be converted in krnl.call builder.
          // This code passed onnx-mlir-opt --convert-krnl-to-llvm test case,
          // but failed in onnx-milr for the tensor type for the attribute
          auto tensorTy = mlir::cast<TensorType>(denseAttr.getType());
          auto memRefTy =
              MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
          Value constantGlobal =
              create.krnl.constant(memRefTy, "constant_", denseAttr);
          Value convertedConstantGlobal =
              rewriter
                  .create<UnrealizedConversionCastOp>(loc,
                      llvmTypeConverter->convertType(memRefTy), constantGlobal)
                  .getResult(0);

          auto int64Ty = IntegerType::get(context, 64);
          auto memRefRank = memRefTy.getRank();
          auto memRefRankVal =
              create.llvm.constant(int64Ty, static_cast<int64_t>(memRefRank));
          Value omTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
              RuntimeAPI::API::CREATE_OMTENSOR, {memRefRankVal});

          Type llvmElemTy =
              llvmTypeConverter->convertType(memRefTy.getElementType());
          krnl::fillOMTensorWithMemRef(convertedConstantGlobal, llvmElemTy,
              omTensor, false /*outOwning*/, rewriter, loc, apiRegistry,
              module);
          auto int8Ty = IntegerType::get(context, 8);
          auto opaquePtrTy = getPointerType(context, int8Ty);
          parameterTypeList.emplace_back(opaquePtrTy);
          parameterList.emplace_back(omTensor);
        })
        .Default([&](Attribute attr) {
          llvm_unreachable("This type of Attribute used by krnl.call is not "
                           "yet implemented");
        });
  }
};

void populateLoweringKrnlCallOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlCallOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
