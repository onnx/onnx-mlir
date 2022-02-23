/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlPrintTensor.cpp - Lower KrnlPrintTensorOp ----------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlPrintTensorOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "onnx/onnx_pb.h"

#include "src/Conversion/KrnlToLLVM/KrnlPrintTensor.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVM.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Conversion/KrnlToLLVM/RuntimeAPI.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {

LogicalResult KrnlPrintTensorOpLowering::matchAndRewrite(Operation *op,
    ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const {
  auto printTensorOp = cast<KrnlPrintTensorOp>(op);
  MLIRContext *context = printTensorOp.getContext();
  Location loc = printTensorOp.getLoc();
  KrnlPrintTensorOpAdaptor operandAdaptor(operands);

  StringRef msg = printTensorOp.msg();
  Value input = operandAdaptor.input();
  assert(input.getType().isa<LLVM::LLVMStructType>() &&
         "expecting LLVMStructType");

  ModuleOp module = printTensorOp->getParentOfType<ModuleOp>();
  const auto &apiRegistry = RuntimeAPIRegistry::build(module, rewriter);

  // Get a symbol reference to the runtime function to use, creating one if
  // necessary.
  auto int64Ty = IntegerType::get(context, 64);
  auto memRefTy = input.getType().dyn_cast<LLVM::LLVMStructType>();
  auto memRefRank = onnx_mlir::getRankFromMemRefType(memRefTy);
  auto memRefRankVal = rewriter.create<LLVM::ConstantOp>(
      loc, int64Ty, rewriter.getI64IntegerAttr(memRefRank));
  Value omTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
      RuntimeAPI::API::CREATE_OMTENSOR, {memRefRankVal});

  onnx_mlir::fillOMTensorWithMemRef(
      input, omTensor, false /*outOwning*/, rewriter, loc, apiRegistry, module);
  LLVM::GlobalOp globalStr = getOrCreateGlobalString(msg, loc, rewriter, module,
      static_cast<LLVMTypeConverter *>(getTypeConverter()));
  Value strPtr = getPtrToGlobalString(globalStr, loc, rewriter);

  RuntimeAPI::callApi(rewriter, loc, apiRegistry,
      RuntimeAPI::API::PRINT_OMTENSOR, {strPtr, omTensor});

  rewriter.eraseOp(op);
  return success();
}

} // namespace onnx_mlir
