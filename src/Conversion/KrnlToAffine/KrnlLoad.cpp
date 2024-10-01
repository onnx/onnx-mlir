/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- KrnlLoad.cpp - Lower KrnlLoadOp -----------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlLoadOp operator.
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

namespace onnx_mlir {
namespace krnl {

/// KrnlLoad will be lowered to std.load or affine.load, depending on whether
/// the access indices are all affine maps or not.
class KrnlLoadLowering : public ConversionPattern {
public:
  explicit KrnlLoadLowering(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlLoadOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loadOp = mlir::cast<KrnlLoadOp>(op);
    KrnlLoadOpAdaptor operandAdaptor(loadOp);

    // Prepare inputs.
    Value memref = operandAdaptor.getMemref();
    SmallVector<Value, 4> indices = operandAdaptor.getIndices();

    // Check whether all indices are affine maps or not.
    bool affineIndices =
        !llvm::any_of(indices, [](Value v) { return !affine::isValidDim(v); });

    if (affineIndices)
      rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(op, memref, indices);
    else
      rewriter.replaceOpWithNewOp<memref::LoadOp>(op, memref, indices);

    return success();
  }
};

void populateLoweringKrnlLoadOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlLoadLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
