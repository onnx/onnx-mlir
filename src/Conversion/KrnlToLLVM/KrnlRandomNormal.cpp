/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlRandomNormal.cpp - Lower KrnlRandomNormalOp ---------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlRandomNormalOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlRandomNormalOpLowering : public ConversionPattern {
public:
  explicit KrnlRandomNormalOpLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, KrnlRandomNormalOp::getOperationName(),
            1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    KrnlRandomNormalOpAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();
    mlir::Type inType = op->getOperand(2).getType();

    // Get a symbol reference to the memcpy function, inserting it if necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto randomNormalFuncRef =
        getOrInsertRandomNormal(rewriter, parentModule, inType);

    // First operand.
    Type outputType = operandAdaptor.output()
                          .getType()
                          .cast<LLVM::LLVMStructType>()
                          .getBody()[1];
    Value alignedOutput = rewriter.create<LLVM::ExtractValueOp>(
        loc, outputType, operandAdaptor.output(), rewriter.getI64ArrayAttr(1));

    // Memcpy call
    rewriter.create<CallOp>(loc, randomNormalFuncRef, ArrayRef<Type>({}),
        ArrayRef<Value>({alignedOutput, operandAdaptor.numberOfValues(),
            operandAdaptor.mean(), operandAdaptor.scale(),
            operandAdaptor.seed()}));

    rewriter.eraseOp(op);
    return success();
  }

private:
  FlatSymbolRefAttr getOrInsertRandomNormal(
      PatternRewriter &rewriter, ModuleOp module, Type inType) const {
    MLIRContext *context = module.getContext();
    StringRef functionName = inType.isF64() ? "get_random_normal_value_f64"
                                            : "get_random_normal_value_f32";
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(functionName.str()))
      return SymbolRefAttr::get(context, functionName.str());

    // Signature of the input is:
    //  "krnl.random_normal"(%0, %c60, %cst, %cst_0, %cst_1)
    // with types:
    //  (memref<3x4x5xf32>, index, f32, f32, f32)
    // or
    //  (memref<3x4x5xf64>, index, f64, f64, f64)
    auto llvmVoidTy = LLVM::LLVMVoidType::get(context);
    auto llvmOptionsTy = FloatType::getF32(context);
    auto llvmOutputTy = LLVM::LLVMPointerType::get(llvmOptionsTy);
    if (inType.isF64()) {
      llvmOptionsTy = FloatType::getF64(context);
      llvmOutputTy = LLVM::LLVMPointerType::get(llvmOptionsTy);
    }
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmVoidTy,
        ArrayRef<mlir::Type>({llvmOutputTy, llvmI64Ty, llvmOptionsTy,
            llvmOptionsTy, llvmOptionsTy}),
        false);

    // Insert the random normal function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(
        module.getLoc(), functionName.str(), llvmFnType);
    return SymbolRefAttr::get(context, functionName.str());
  }
};

void populateLoweringKrnlRandomNormalOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlRandomNormalOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir