/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlPrint.cpp - Lower KrnlPrintOp -----------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlPrintOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "src/Conversion/KrnlToLLVM/KrnlPrint.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVM.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {

LogicalResult KrnlPrintOpLowering::matchAndRewrite(Operation *op,
    ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const {
  auto printOp = cast<KrnlPrintOp>(op);
  Location loc = printOp.getLoc();
  KrnlPrintOpAdaptor operandAdaptor(operands);

  Value input = operandAdaptor.input();
  StringRef format = printOp.format();
  ModuleOp module = printOp->getParentOfType<ModuleOp>();

  // Get a symbol reference to the runtime function to use, creating one if
  // necessary.
  auto printfFuncRef = getOrInsertPrintf(rewriter, module);

  // Printf call.
  LLVM::GlobalOp formatSpec = getOrCreateGlobalString(format, loc, rewriter,
      module, static_cast<LLVMTypeConverter *>(getTypeConverter()));
  Value formatSpecPtr = getPtrToGlobalString(formatSpec, loc, rewriter);

  if (input)
    rewriter.create<CallOp>(loc, printfFuncRef, ArrayRef<Type>({}),
        ArrayRef<Value>({formatSpecPtr, input}));
  else
    rewriter.create<CallOp>(loc, printfFuncRef, ArrayRef<Type>({}),
        ArrayRef<Value>({formatSpecPtr}));

  rewriter.eraseOp(op);
  return success();
}

FlatSymbolRefAttr KrnlPrintOpLowering::getOrInsertPrintf(
    PatternRewriter &rewriter, ModuleOp module) {
  // Insert the printf declaration if it is not already present.
  auto printfFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("printf");
  MLIRContext *ctx = rewriter.getContext();

  if (!printfFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto voidType = LLVM::LLVMVoidType::get(ctx);
    Type i8Type = IntegerType::get(ctx, 8);
    Type i8PtrType = LLVM::LLVMPointerType::get(i8Type);
    printfFunc =
        rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), "printf",
            LLVM::LLVMFunctionType::get(voidType, i8PtrType,
                /*isVarArg=*/true));
  }
  return SymbolRefAttr::get(ctx, "printf");
}

} // namespace onnx_mlir
