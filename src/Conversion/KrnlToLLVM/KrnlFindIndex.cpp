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

#include "llvm/ADT/TypeSwitch.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;

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
    MLIRContext *ctx = findIndexOp.getContext();
    Location loc = findIndexOp.getLoc();
    KrnlFindIndexOpAdaptor operandAdaptor(operands);
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

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
          Type i8Type = IntegerType::get(ctx, 8);
          Type i8PtrType = LLVM::LLVMPointerType::get(i8Type);
          firstOperand = rewriter.create<LLVM::IntToPtrOp>(
              loc, i8PtrType, operandAdaptor.input());
        })
        .Default([](Type type) {
          llvm::errs() << "type: " << type << "\n";
          llvm_unreachable("unexpected inputType");
        });

    Type GType =
        operandAdaptor.G().getType().cast<LLVM::LLVMStructType>().getBody()[1];
    Type VType =
        operandAdaptor.V().getType().cast<LLVM::LLVMStructType>().getBody()[1];

    // Remaining operands.
    Value extractedGPtr =
        create.llvm.extractValue(GType, operandAdaptor.G(), {1});
    Value extractedVPtr =
        create.llvm.extractValue(VType, operandAdaptor.V(), {1});
    Value length = operandAdaptor.len();

    // Generate the call to the runtime function.
    Type retType = IntegerType::get(ctx, 64);
    Value funcCall = create.llvm.call(retType, findIndexRef,
        ArrayRef<Value>({firstOperand, extractedGPtr, extractedVPtr, length}));

    rewriter.replaceOp(op, funcCall);
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
    MultiDialectBuilder<LLVMBuilder> create(rewriter, module.getLoc());

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
        .Default([](Type type) {
          llvm::errs() << "type: " << type << "\n";
          llvm_unreachable("unexpected type");
        });

    // Create 'find_index_*' signature: `i64 ([i8*|i64], i32*, i32*, i32)`
    return create.llvm.getOrInsertSymbolRef(module, StringRef(funcName),
        i64Type, {firstArgType, i32PtrType, i32PtrType, i32Type});
  }
};

void populateLoweringKrnlFindIndexOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlFindIndexOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
