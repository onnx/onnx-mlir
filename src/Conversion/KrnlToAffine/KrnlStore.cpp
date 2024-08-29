/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- KrnlStore.cpp - Lower KrnlStoreOp ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlStoreOp operator.
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

/// KrnlStore will be lowered to std.store or affine.store, depending on whether
/// the access indices are all affine maps or not.
class KrnlStoreLowering : public ConversionPattern {
public:
  explicit KrnlStoreLowering(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlStoreOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto storeOp = mlir::cast<KrnlStoreOp>(op);
    KrnlStoreOpAdaptor operandAdaptor(storeOp);

    // Prepare inputs.
    Value value = operandAdaptor.getValue();
    Value memref = operandAdaptor.getMemref();
    SmallVector<Value, 4> indices = operandAdaptor.getIndices();

    // Check whether all indices are affine maps or not.
    bool affineIndices =
        !llvm::any_of(indices, [](Value v) { return !affine::isValidDim(v); });

    if (affineIndices)
      rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
          op, value, memref, indices);
    else
      rewriter.replaceOpWithNewOp<memref::StoreOp>(op, value, memref, indices);

    return success();
  }
};

void populateLoweringKrnlStoreOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlStoreLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
