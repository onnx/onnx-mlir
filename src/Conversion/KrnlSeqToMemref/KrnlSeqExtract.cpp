/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlSeqExtract.cpp - Lower KrnlSeqExtractOp ------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlSeqExtractOp operator.
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
    KrnlSeqExtractOp thisOp = mlir::dyn_cast<KrnlSeqExtractOp>(op);
    Location loc = op->getLoc();
    MultiDialectBuilder<MathBuilder, MemRefBuilder> create(rewriter, loc);

    auto output = rewriter
                      .create<memref::LoadOp>(loc, operandAdaptor.getSeq(),
                          operandAdaptor.getIndex())
                      .getResult();

    // TODO: overwrite the element in seq so that runtime error can be detected
    // if the element is read from seq after extracted, or deep deallocation
    // is added when seq is freed

    if (thisOp.getCopy() == 0) {
      rewriter.replaceOp(op, output);
      return success();
    } else {
      if (!mlir::isa<MemRefType>(output.getType()))
        llvm_unreachable(
            "Not implemented: type of onnx seq element is not tensor");
      auto outputType = mlir::cast<MemRefType>(output.getType());
      SmallVector<Value, 4> allocParams;
      for (size_t i = 0; i < outputType.getShape().size(); i++) {
        if (outputType.isDynamicDim(i)) {
          allocParams.emplace_back(create.mem.dim(output, i));
        }
      }
      Value alloc = create.mem.alignedAlloc(outputType, allocParams);
      rewriter.create<memref::CopyOp>(loc, output, alloc);
      rewriter.replaceOp(op, alloc);
      return success();
    }
  }
};

void populateLoweringKrnlSeqExtractOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlSeqExtractOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
