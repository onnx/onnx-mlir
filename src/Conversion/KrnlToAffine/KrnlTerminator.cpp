/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- KrnlMatmul.cpp - Lower KrnlMatmulOp -------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlMatmulOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "src/Conversion/KrnlToAffine/ConvertKrnlToAffine.hpp"
#include "src/Conversion/KrnlToLLVM/RuntimeAPI.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_affine"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlTerminatorLowering : public ConversionPattern {
public:
  explicit KrnlTerminatorLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlTerminatorOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AffineYieldOp>(op);
    return success();
  }
};

void populateLoweringKrnlTerminatorOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlTerminatorLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
