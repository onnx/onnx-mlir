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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlRandomNormalOpLowering : public ConversionPattern {
public:
  explicit KrnlRandomNormalOpLowering(
      LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, KrnlRandomNormalOp::getOperationName(),
            1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    KrnlRandomNormalOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();
    Type inType = op->getOperand(2).getType();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    // Get a symbol reference to the memcpy function, inserting it if necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto randomNormalFuncRef =
        getOrInsertRandomNormal(rewriter, parentModule, inType);

    // First operand.
    Type outputType =
        mlir::cast<LLVM::LLVMStructType>(operandAdaptor.getOutput().getType())
            .getBody()[1];
    Value alignedOutput =
        create.llvm.extractValue(outputType, operandAdaptor.getOutput(), {1});

    // Memcpy call
    create.llvm.call({}, randomNormalFuncRef,
        {alignedOutput, operandAdaptor.getNumberOfValues(),
            operandAdaptor.getMean(), operandAdaptor.getScale(),
            operandAdaptor.getSeed()});

    rewriter.eraseOp(op);
    return success();
  }

private:
  FlatSymbolRefAttr getOrInsertRandomNormal(
      PatternRewriter &rewriter, ModuleOp module, Type inType) const {
    MLIRContext *context = module.getContext();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, module.getLoc());
    StringRef functionName = inType.isF64() ? "get_random_normal_value_f64"
                                            : "get_random_normal_value_f32";
    // Signature of the input is:
    //  "krnl.random_normal"(%0, %c60, %cst, %cst_0, %cst_1)
    // with types:
    //  (memref<3x4x5xf32>, index, f32, f32, f32)
    // or
    //  (memref<3x4x5xf64>, index, f64, f64, f64)
    Type llvmVoidTy = LLVM::LLVMVoidType::get(context);
    Type llvmOptionsTy = FloatType::getF32(context);
    Type llvmOutputTy = getPointerType(context, llvmOptionsTy);
    if (inType.isF64()) {
      llvmOptionsTy = FloatType::getF64(context);
      llvmOutputTy = getPointerType(context, llvmOptionsTy);
    }
    Type llvmI64Ty = IntegerType::get(context, 64);
    return create.llvm.getOrInsertSymbolRef(module, functionName, llvmVoidTy,
        {llvmOutputTy, llvmI64Ty, llvmOptionsTy, llvmOptionsTy, llvmOptionsTy});
  }
};

void populateLoweringKrnlRandomNormalOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlRandomNormalOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
