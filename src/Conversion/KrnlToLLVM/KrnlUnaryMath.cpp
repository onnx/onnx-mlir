/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlUnaryMath.cpp - Lower KrnlUnaryMath Ops -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlUnaryMath operators.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

template <typename Op>
struct MathFunctionName {
  static std::string functionName() { return "none"; };
};

template <>
struct MathFunctionName<KrnlErfOp> {
  static std::string functionName(mlir::Type type) {
    if (type.isF32())
      return "erff";
    if (type.isF64())
      return "erf";
    llvm_unreachable("Currently unsupported type for erf");
  }
};

template <>
struct MathFunctionName<KrnlAcosOp> {
  static std::string functionName(mlir::Type type) {
    if (type.isF32())
      return "acosf";
    if (type.isF64())
      return "acos";
    llvm_unreachable("Unsupported type for acos");
  }
};

template <>
struct MathFunctionName<KrnlAcoshOp> {
  static std::string functionName(mlir::Type type) {
    if (type.isF32())
      return "acoshf";
    if (type.isF64())
      return "acosh";
    llvm_unreachable("Unsupported type for acosh");
  }
};

template <>
struct MathFunctionName<KrnlAsinOp> {
  static std::string functionName(mlir::Type type) {
    if (type.isF32())
      return "asinf";
    if (type.isF64())
      return "asin";
    llvm_unreachable("Unsupported type for asin");
  }
};

template <>
struct MathFunctionName<KrnlAsinhOp> {
  static std::string functionName(mlir::Type type) {
    if (type.isF32())
      return "asinhf";
    if (type.isF64())
      return "asinh";
    llvm_unreachable("Unsupported type for asinh");
  }
};

template <>
struct MathFunctionName<KrnlAtanOp> {
  static std::string functionName(mlir::Type type) {
    if (type.isF32())
      return "atanf";
    if (type.isF64())
      return "atan";
    llvm_unreachable("Unsupported type for atan");
  }
};

template <>
struct MathFunctionName<KrnlTanOp> {
  static std::string functionName(mlir::Type type) {
    if (type.isF32())
      return "tanf";
    if (type.isF64())
      return "tan";
    llvm_unreachable("Unsupported type for tan");
  }
};

template <>
struct MathFunctionName<KrnlAtanhOp> {
  static std::string functionName(mlir::Type type) {
    if (type.isF32())
      return "atanhf";
    if (type.isF64())
      return "atanh";
    llvm_unreachable("Unsupported type for atanh");
  }
};

template <typename KrnlScalarMathOp>
class KrnlUnaryMathOpLowering : public ConversionPattern {
public:
  explicit KrnlUnaryMathOpLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlScalarMathOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = op->getContext();
    Location loc = op->getLoc();

    // get the LLVM type for the function args and result
    mlir::Type inType = op->getOperand(0).getType();
    mlir::Type llvmType;
    if (inType.isF32())
      llvmType = FloatType::getF32(context);
    else if (inType.isF64())
      llvmType = FloatType::getF64(context);

    // Insert and/or get reference to elementary math function declaration.
    assert(
        inType.isIntOrFloat() && "Type for math function must be int or float");
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto mathFunctionRef = getOrInsertUnaryMathFunction(rewriter, parentModule,
        MathFunctionName<KrnlScalarMathOp>().functionName(inType), llvmType);

    // Emit function call.
    auto funcCall = rewriter.create<func::CallOp>(
        loc, mathFunctionRef, llvmType, ArrayRef<Value>({operands[0]}));
    rewriter.replaceOp(op, funcCall.getResults()[0]);
    return success();
  }

private:
  // This function emits a declaration of the form:
  //
  // declare float <mathFuncName>(float)
  //
  FlatSymbolRefAttr getOrInsertUnaryMathFunction(PatternRewriter &rewriter,
      ModuleOp module, std::string mathFuncName, mlir::Type llvmType) const {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(mathFuncName))
      return SymbolRefAttr::get(context, mathFuncName);

    // Create function declaration.
    // auto llvmF32Ty = FloatType::get(context);
    auto llvmFnType =
        LLVM::LLVMFunctionType::get(llvmType, ArrayRef<mlir::Type>({llvmType}));

    // Insert the unary math function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(
        module.getLoc(), mathFuncName, llvmFnType);
    return SymbolRefAttr::get(context, mathFuncName);
  }
};

void populateLoweringKrnlUnaryMathOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlUnaryMathOpLowering<KrnlErfOp>>(typeConverter, ctx);
  patterns.insert<KrnlUnaryMathOpLowering<KrnlAcosOp>>(typeConverter, ctx);
  patterns.insert<KrnlUnaryMathOpLowering<KrnlAcoshOp>>(typeConverter, ctx);
  patterns.insert<KrnlUnaryMathOpLowering<KrnlAsinOp>>(typeConverter, ctx);
  patterns.insert<KrnlUnaryMathOpLowering<KrnlAsinhOp>>(typeConverter, ctx);
  patterns.insert<KrnlUnaryMathOpLowering<KrnlAtanOp>>(typeConverter, ctx);
  patterns.insert<KrnlUnaryMathOpLowering<KrnlAtanhOp>>(typeConverter, ctx);
  patterns.insert<KrnlUnaryMathOpLowering<KrnlTanOp>>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
