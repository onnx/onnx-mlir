/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ KrnlFindIndex.cpp - Lowering KrnlFindIndexOp ------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlFindIndexOp operator.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlFindIndexOpLowering : public ConversionPattern {
public:
  explicit KrnlFindIndexOpLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlFindIndexOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto findIndexOp = cast<KrnlFindIndexOp>(op);
    MLIRContext *context = findIndexOp.getContext();
    Location loc = findIndexOp.getLoc();
    KrnlFindIndexOpAdaptor operandAdaptor(operands);

    // Get a symbol reference to the runtime function to use, creating one if
    // necessary.
    ModuleOp module = findIndexOp->getParentOfType<ModuleOp>();
    FlatSymbolRefAttr findIndexRef =
        getOrInsertFindIndex(rewriter, module, findIndexOp.input().getType());

    // Select the value to pass to as the first argument based on the operator
    // input type.
    Value firstOperand;
    TypeSwitch<Type>(findIndexOp.input().getType())
        .Case<IntegerType>([&](IntegerType type) {
          assert(type.getWidth() == 64 && "expecting an i64 type");
          firstOperand = operandAdaptor.input();
        })
        .Case<StringType>([&](StringType type) {
          Type ptrType = operandAdaptor.input()
                             .getType()
                             .cast<LLVM::LLVMStructType>()
                             .getBody()[1];
          firstOperand = rewriter.create<LLVM::ExtractValueOp>(loc, ptrType,
              operandAdaptor.input(), rewriter.getI64ArrayAttr(1));
        })
        .Default([](Type) { llvm_unreachable("unexpected inputType"); });

    Type GType =
        operandAdaptor.G().getType().cast<LLVM::LLVMStructType>().getBody()[1];
    Type VType =
        operandAdaptor.V().getType().cast<LLVM::LLVMStructType>().getBody()[1];

    // Remaining operands.
    Value extractedGPtr = rewriter.create<LLVM::ExtractValueOp>(
        loc, GType, operandAdaptor.G(), rewriter.getI64ArrayAttr(1));
    Value extractedVPtr = rewriter.create<LLVM::ExtractValueOp>(
        loc, VType, operandAdaptor.V(), rewriter.getI64ArrayAttr(1));
    Value length = operandAdaptor.len();

    // Generate the call to the runtime function.
    Type retType = IntegerType::get(context, 64);
    auto funcCall = rewriter.create<CallOp>(loc, findIndexRef, retType,
        ArrayRef<Value>({firstOperand, extractedGPtr, extractedVPtr, length}));

    rewriter.replaceOp(op, funcCall.getResults()[0]);
    return success();
  }

private:
  /// Return a symbol reference to the appropriate 'find_index_*' runtime
  /// function, inserting it into the module if necessary.
  static FlatSymbolRefAttr getOrInsertFindIndex(
      PatternRewriter &rewriter, ModuleOp module, Type inputType) {
    MLIRContext *ctx = module.getContext();
    Type i8Type = IntegerType::get(ctx, 8);
    Type i32Type = IntegerType::get(ctx, 32);
    Type i64Type = IntegerType::get(ctx, 64);
    Type i8PtrType = LLVM::LLVMPointerType::get(i8Type);
    Type i32PtrType = LLVM::LLVMPointerType::get(i32Type);

    // Select the runtime function to use based on the input type.
    std::string funcName = "find_index_";
    Type firstArgType;
    TypeSwitch<Type>(inputType)
        .Case<IntegerType>([&](IntegerType type) {
          assert(type.getWidth() == 64 && "expecting an i64 type");
          funcName += "i64";
          firstArgType = i64Type;
        })
        .Case<StringType>([&](StringType type) {
          funcName += "str";
          firstArgType = i8PtrType;
        })
        .Default([](Type) { llvm_unreachable("unexpected type"); });

    Optional<FlatSymbolRefAttr> optFuncDecl =
        krnl::getFunctionDeclaration(module, funcName);
    if (optFuncDecl.hasValue())
      return optFuncDecl.getValue();

    // Create 'find_index_*' signature: `i64 ([i8*|i64], i32*, i32*, i32)`
    Type fnType = LLVM::LLVMFunctionType::get(i64Type,
        ArrayRef<Type>({firstArgType, i32PtrType, i32PtrType, i32Type}), false);

    // Insert the function declaration the module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, fnType);

    return SymbolRefAttr::get(ctx, funcName);
  }
};

void populateLoweringKrnlFindIndexOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlFindIndexOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
