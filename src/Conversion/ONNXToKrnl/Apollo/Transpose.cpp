/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Transpose.cpp - Transpose Op -----------------------===//
//
// Copyright 2021 Microsoft
//
// =============================================================================
//
// This file lowers ONNX transpose operator to a function call
// that will be lowered to Apollo-specific code.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;

namespace {
struct ONNXTransposeOpApolloLowering : public ConversionPattern {

  ONNXTransposeOpApolloLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXTransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXTransposeOpAdaptor operandAdaptor(operands);
    ONNXTransposeOp transposeOp = llvm::cast<ONNXTransposeOp>(op);
    auto loc = op->getLoc();

    // Operands and attributes.
    Value data = operandAdaptor.data();
    auto permAttr = transposeOp.perm();

    // Basic information.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    int64_t rank = memRefType.getShape().size();

    SmallVector<unsigned int> perm;
    for (decltype(rank) i = 0; i < rank; ++i) {
      perm.push_back(ArrayAttrIntVal(permAttr, i));
    }

    // Insert an allocation and deallocation for the result of this
    // operation.
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType)) {
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    } else {
      op->emitWarning("This operation produces unsupported by "
                      "current target dynamically sized tensor.");
      return failure();
    }

    rewriter.create<linalg::CopyOp>(loc, operandAdaptor.data(), alloc,
        AffineMap(), AffineMap::getPermutationMap(perm, getContext()));

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

} // namespace

void populateLoweringONNXTransposeOpApolloPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXTransposeOpApolloLowering>(ctx);
}
