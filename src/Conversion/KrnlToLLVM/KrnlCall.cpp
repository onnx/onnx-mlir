/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- KrnlCall.cpp - Lower KrnlCallOp -----------------------===//
//
// Copyright 2022 The IBM Research Authors.
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
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlCallOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    KrnlCallOpAdaptor krnlCallAdaptor(operands);
    Location loc = op->getLoc();
    KrnlCallOp krnlCallOp = llvm::cast<KrnlCallOp>(op);
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

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

    FlatSymbolRefAttr callRef =
        create.llvm.getOrInsertSymbolRef(module, krnlCallOp.funcName(),
            LLVM::LLVMVoidType::get(module.getContext()), parameterTypeList);
    create.llvm.call({}, callRef, parameterList);

    rewriter.eraseOp(op);
    return success();
  }

private:
  static void handleOneParameter(PatternRewriter &rewriter, Operation *op,
      Value parameter, Value original,
      llvm::SmallVector<Type, 4> &parameterTypeList,
      llvm::SmallVector<Value, 4> &parameterList) {
    MLIRContext *context = op->getContext();
    Location loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    const auto &apiRegistry = RuntimeAPIRegistry::build(module, rewriter);
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    // Check the original type, not after type conversion
    Type ty = original.getType();
    if (ty.isa<MemRefType>()) {
      auto int64Ty = IntegerType::get(context, 64);
      auto memRefTy = parameter.getType().dyn_cast<LLVM::LLVMStructType>();
      auto memRefRank = krnl::getRankFromMemRefType(memRefTy);
      auto memRefRankVal = create.llvm.constant(int64Ty, (int64_t)memRefRank);
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
    MultiDialectBuilder<KrnlBuilder, LLVMBuilder> create(rewriter, loc);

    TypeSwitch<Attribute>(attribute)
        .Case<StringAttr>([&](StringAttr strAttr) {
          StringRef attrValue = strAttr.getValue();
          LLVM::GlobalOp globalStr =
              krnl::getOrCreateGlobalString(attrValue, loc, rewriter, module,
                  static_cast<LLVMTypeConverter *>(typeConverter));
          Value strPtr = krnl::getPtrToGlobalString(globalStr, loc, rewriter);
          auto int8Ty = IntegerType::get(context, 8);
          auto opaquePtrTy = LLVM::LLVMPointerType::get(int8Ty);
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
          Value cst = rewriter.create<LLVM::ConstantOp>(loc, f64Ty, floatAttr);
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
          const auto &apiRegistry = RuntimeAPIRegistry::build(module, rewriter);
          auto tensorTy = denseAttr.getType().cast<TensorType>();
          auto memRefTy =
              MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
          memRefTy.dump();
          Value constantGlobal =
              create.krnl.constant(memRefTy, "constant_", denseAttr);
          Value convertedConstantGlobal =
              rewriter
                  .create<UnrealizedConversionCastOp>(
                      loc, typeConverter->convertType(memRefTy), constantGlobal)
                  .getResult(0);
          // constantGlobal.setType(typeConverter->convertType(memRefTy));

          auto int64Ty = IntegerType::get(context, 64);
          auto memRefRank = memRefTy.getRank();
          auto memRefRankVal =
              create.llvm.constant(int64Ty, (int64_t)memRefRank);
          Value omTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
              RuntimeAPI::API::CREATE_OMTENSOR, {memRefRankVal});

          krnl::fillOMTensorWithMemRef(convertedConstantGlobal, omTensor,
              false /*outOwning*/, rewriter, loc, apiRegistry, module);
          auto int8Ty = IntegerType::get(context, 8);
          auto opaquePtrTy = LLVM::LLVMPointerType::get(int8Ty);
          parameterTypeList.emplace_back(opaquePtrTy);
          parameterList.emplace_back(omTensor);
        })
        .Default([&](Attribute attr) {
          llvm_unreachable("This type of Attribute used by krnl.call is not "
                           "yet implemented");
        });
  }
};

void populateLoweringKrnlCallOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlCallOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
