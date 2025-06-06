//===----KrnlRandomUniform.cpp - Lower KrnlRandomUniformOp//-----------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlRandomUniformOp operator.
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

class KrnlRandomUniformOpLowering : public ConversionPattern {
public:
  explicit KrnlRandomUniformOpLowering(
      LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter,
            KrnlRandomUniformOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    KrnlRandomUniformOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();
    Type inType = op->getOperand(2).getType();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    // Get a symbol reference to the memcpy function, inserting it if necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto randomUniformFuncRef =
        getOrInsertRandomUniform(rewriter, parentModule, inType);

    Type outputType =
        mlir::cast<LLVM::LLVMStructType>(operandAdaptor.getOutput().getType())
            .getBody()[1];
    Value alignedOutput =
        create.llvm.extractValue(outputType, operandAdaptor.getOutput(), {1});

    create.llvm.call({}, randomUniformFuncRef,
        {alignedOutput, operandAdaptor.getNumberOfValues(),
            operandAdaptor.getLow(), operandAdaptor.getHigh(),
            operandAdaptor.getSeed()});

    rewriter.eraseOp(op);
    return success();
  }

private:
  FlatSymbolRefAttr getOrInsertRandomUniform(
      PatternRewriter &rewriter, ModuleOp module, Type inType) const {
    MLIRContext *context = module.getContext();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, module.getLoc());
    StringRef functionName = inType.isF64() ? "get_uniform_random_value_f64"
                                            : "get_uniform_random_value_f32";
    Type llvmVoidTy = LLVM::LLVMVoidType::get(context);
    Type llvmOptionsTy = Float32Type::get(context);
    Type llvmOutputTy = getPointerType(context, llvmOptionsTy);
    if (inType.isF64()) {
      llvmOptionsTy = Float64Type::get(context);
      llvmOutputTy = getPointerType(context, llvmOptionsTy);
    }
    Type llvmI64Ty = IntegerType::get(context, 64);
    return create.llvm.getOrInsertSymbolRef(module, functionName, llvmVoidTy,
        {llvmOutputTy, llvmI64Ty, llvmOptionsTy, llvmOptionsTy, llvmOptionsTy});
  }
};

void populateLoweringKrnlRandomUniformOpPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    MLIRContext *ctx) {
  patterns.insert<KrnlRandomUniformOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
