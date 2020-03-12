//===---- constant.cpp - Lowering Constant Op -----------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Constant Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/conversion/onnx_to_krnl/onnx_to_krnl_common.hpp"

using namespace mlir;

struct ONNXConstantOpLowering : public ConversionPattern {
  ONNXConstantOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXConstantOp::getOperationName(), 1, ctx) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto constantOp = llvm::dyn_cast<ONNXConstantOp>(op);

    if (constantOp.sparse_value().hasValue()) {
      emitError(loc, "Only support dense values at this time");
    }

    auto memRefType = convertToMemRefType(*op->result_type_begin());

    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      emitError(loc, "unexpected output has non-Constant shape");

    DenseElementsAttr constantValue =
        constantOp.value().getValue().cast<DenseElementsAttr>();

    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;
    for (auto i : llvm::seq<int64_t>(
             0, *std::max_element(valueShape.begin(), valueShape.end())))
      constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, i));

    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.

    // TODO (tungld): Refactor the code to use a common function to walk through
    // the attribute.
    if (memRefType.getElementType().isa<IntegerType>()) {
      SmallVector<Value, 2> indices;
      auto valueIt = constantValue.getValues<IntegerAttr>().begin();
      std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
        // The last dimension is the base case of the recursion, at this point
        // we store the element at the given index.
        if (dimension == valueShape.size()) {
          rewriter.create<AffineStoreOp>(loc,
              rewriter.create<ConstantOp>(loc, *valueIt++), alloc,
              llvm::makeArrayRef(indices));
          return;
        }

        // Otherwise, iterate over the current dimension and add the indices to
        // the list.
        for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
          indices.push_back(constantIndices[i]);
          storeElements(dimension + 1);
          indices.pop_back();
        }
      };

      // Start the element storing recursion from the first dimension.
      storeElements(/*dimension=*/0);
    } else if (memRefType.getElementType().isa<FloatType>()) {
      SmallVector<Value, 2> indices;
      auto valueIt = constantValue.getValues<FloatAttr>().begin();
      std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
        // The last dimension is the base case of the recursion, at this point
        // we store the element at the given index.
        if (dimension == valueShape.size()) {
          rewriter.create<AffineStoreOp>(loc,
              rewriter.create<ConstantOp>(loc, *valueIt++), alloc,
              llvm::makeArrayRef(indices));
          return;
        }

        // Otherwise, iterate over the current dimension and add the indices to
        // the list.
        for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
          indices.push_back(constantIndices[i]);
          storeElements(dimension + 1);
          indices.pop_back();
        }
      };

      // Start the element storing recursion from the first dimension.
      storeElements(/*dimension=*/0);
    }

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

void populateLoweringONNXConstantOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXConstantOpLowering>(ctx);
}
