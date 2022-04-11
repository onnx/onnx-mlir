/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlSeqExtract.cpp - Lower KrnlSeqExtractOp ------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlSeqExtractOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlSeqExtractOpLowering : public ConversionPattern {
public:
  explicit KrnlSeqExtractOpLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlSeqExtractOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    KrnlSeqExtractOpAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();

    auto output = rewriter
                      .create<memref::LoadOp>(
                          loc, operandAdaptor.seq(), operandAdaptor.index())
                      .getResult();

    // TODO: overwrite the element in seq so that runtime error can be detected
    // if the element is read from seq after extracted, or deep deallocation
    // is added when seq is freed

    rewriter.replaceOp(op, output);
    return success();
  }
};

void populateLoweringKrnlSeqExtractOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlSeqExtractOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
