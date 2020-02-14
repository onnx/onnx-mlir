//===- onnx_rewrite.cpp - ONNX High Level Optimizer -----------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters for operations in the ONNX dialect
// that can be rewritten by using other ONNX operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "src/dialect/onnx/onnx_ops.hpp"

using namespace mlir;

namespace {

// There are two ways to write rewrite rules:
// - Declarative manner: specify rewrite rules in a TableGen record, and
// - Manual Manner: subclass the mlir::RewritePattern.
//
// We prefer to use the former way as much as possible. However, there is a
// limitation about operation definition specification (ODS) in TableGen that
// requires us to write custom builders, that is
// "all ODS-generated `build()` methods require specifying the result type(s),
// unless the op has known traits like `SameOperandsAndResultType` that we can
// use to auto-generate a `build()` method with result type deduction".
//
// More information about the limitation can be found here:
// https://github.com/llvm/llvm-project/blob/master/mlir/docs/DeclarativeRewrites.md#building-operations
//
// Currently, we use the latter way of writing rewrite rules. There are two
// reasons for this decision:
// - To insert custom builders for operations, it is better to change the script
// gen_doc.py to generate all possibles custom builders for a large class of
// operations. At the time of this patch created, the gen_doc.py was changing,
// so we decided to write manually to reduce conflicts.
// - In declarative rewriting, we should deal with optional attributes. E.g. for
// to handle optional attributes, but I haven't tried it yet.
//
// Once we have done the above issues, we will switch to use the declarative
// manner.

//===----------------------------------------------------------------------===//
// ONNXReduceL1Op %X = ONNXReduceSumOp (ONNXAbsOp %X)
//===----------------------------------------------------------------------===//
struct ReduceL1OpPattern : public RewritePattern {
  ReduceL1OpPattern(MLIRContext *context)
      : RewritePattern(ONNXReduceL1Op::getOperationName(),
                       {ONNXAbsOp::getOperationName(),
                        ONNXReduceSumOp::getOperationName()},
                       1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto opInput = op->getOperands()[0]; // %X
    auto opResults = op->getResults();
    auto opAttrs = op->getAttrs();

    // Rewrite
    ONNXAbsOp absOp;
    {
      auto elementType = opInput.getType().cast<TensorType>().getElementType();
      absOp = rewriter.create<ONNXAbsOp>(
          loc, UnrankedTensorType::get(elementType), opInput);
    }

    ONNXReduceSumOp sumOp;
    {
      SmallVector<Type, 4> types;
      for (auto v : opResults) {
        types.emplace_back(v.getType());
      }

      SmallVector<Value, 1> values;
      values.emplace_back(absOp.getResult());

      SmallVector<NamedAttribute, 4> attrs;
      for (auto attr : opAttrs) {
        attrs.emplace_back(attr);
      }

      sumOp = rewriter.create<ONNXReduceSumOp>(loc, types, values, attrs);
    }

    rewriter.replaceOp(op, sumOp.getResult());
    return matchSuccess();
  };
};

//===----------------------------------------------------------------------===//
// ONNXReduceL2Op %X = ONNXSqrtOp (ONNXReduceSumSquareOp (%X))
//===----------------------------------------------------------------------===//
struct ReduceL2OpPattern : public RewritePattern {
  ReduceL2OpPattern(MLIRContext *context)
      : RewritePattern(ONNXReduceL2Op::getOperationName(),
                       {ONNXSqrtOp::getOperationName(),
                        ONNXReduceSumSquareOp::getOperationName()},
                       1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto opInput = op->getOperands()[0]; // %X
    auto opResults = op->getResults();
    auto opAttrs = op->getAttrs();

    // Rewrite
    ONNXReduceSumSquareOp sumSquareOp;
    {
      auto elementType = opInput.getType().cast<TensorType>().getElementType();
      sumSquareOp = rewriter.create<ONNXReduceSumSquareOp>(
          loc, UnrankedTensorType::get(elementType), opInput, opAttrs);
    }

    ONNXSqrtOp sqrtOp;
    {
      SmallVector<Type, 4> types;
      for (auto v : opResults) {
        types.emplace_back(v.getType());
      }
      sqrtOp = rewriter.create<ONNXSqrtOp>(loc, types, sumSquareOp.getResult());
    }

    rewriter.replaceOp(op, sqrtOp.getResult());
    return matchSuccess();
  };
};

//===----------------------------------------------------------------------===//
// ONNXReduceLogSumOp %X = ONNXLogOp (ONNXReduceSumOp (%X))
//===----------------------------------------------------------------------===//
struct ReduceLogSumOpPattern : public RewritePattern {
  ReduceLogSumOpPattern(MLIRContext *context)
      : RewritePattern(ONNXReduceLogSumOp::getOperationName(),
                       {ONNXReduceSumOp::getOperationName(),
                        ONNXLogOp::getOperationName()},
                       1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto opInput = op->getOperands()[0]; // %X
    auto opResults = op->getResults();
    auto opAttrs = op->getAttrs();

    // Rewrite
    ONNXReduceSumOp sumOp;
    {
      auto elementType = opInput.getType().cast<TensorType>().getElementType();
      sumOp = rewriter.create<ONNXReduceSumOp>(
          loc, UnrankedTensorType::get(elementType), opInput, opAttrs);
    }

    ONNXLogOp logOp;
    {
      SmallVector<Type, 4> types;
      for (auto v : opResults) {
        types.emplace_back(v.getType());
      }
      logOp = rewriter.create<ONNXLogOp>(loc, types, sumOp.getResult());
    }

    rewriter.replaceOp(op, logOp.getResult());
    return matchSuccess();
  };
};

//===----------------------------------------------------------------------===//
// ONNXReduceLogSumExpOp %X = ONNXReduceLogSumOp (ONNXExpOp %X)
//===----------------------------------------------------------------------===//
struct ReduceLogSumExpOpPattern : public RewritePattern {
  ReduceLogSumExpOpPattern(MLIRContext *context)
      : RewritePattern(ONNXReduceLogSumExpOp::getOperationName(),
                       {ONNXExpOp::getOperationName(),
                        ONNXReduceLogSumOp::getOperationName()},
                       1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto opInput = op->getOperands()[0]; // %X
    auto opResults = op->getResults();
    auto opAttrs = op->getAttrs();

    // Rewrite
    ONNXExpOp expOp;
    {
      auto elementType = opInput.getType().cast<TensorType>().getElementType();
      expOp = rewriter.create<ONNXExpOp>(
          loc, UnrankedTensorType::get(elementType), opInput);
    }

    ONNXReduceLogSumOp logSumOp;
    {
      SmallVector<Type, 4> types;
      for (auto v : opResults) {
        types.emplace_back(v.getType());
      }

      SmallVector<Value, 1> values;
      values.emplace_back(expOp.getResult());

      SmallVector<NamedAttribute, 4> attrs;
      for (auto attr : opAttrs) {
        attrs.emplace_back(attr);
      }
      logSumOp = rewriter.create<ONNXReduceLogSumOp>(loc, types, values, attrs);
    }

    rewriter.replaceOp(op, logSumOp.getResult());
    return matchSuccess();
  };
};

//===----------------------------------------------------------------------===//
// ONNXReduceSumSquareOp %X = ONNXReduceSumOp (ONNXMulOp %X, %X)
//===----------------------------------------------------------------------===//
struct ReduceSumSquareOpPattern : public RewritePattern {
  ReduceSumSquareOpPattern(MLIRContext *context)
      : RewritePattern(ONNXReduceSumSquareOp::getOperationName(),
                       {ONNXMulOp::getOperationName(),
                        ONNXReduceSumOp::getOperationName()},
                       1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto opInput = op->getOperands()[0]; // %X
    auto opResults = op->getResults();
    auto opAttrs = op->getAttrs();

    // Rewrite
    ONNXMulOp mulOp;
    {
      auto elementType = opInput.getType().cast<TensorType>().getElementType();
      mulOp = rewriter.create<ONNXMulOp>(
          loc, UnrankedTensorType::get(elementType), opInput, opInput);
    }

    ONNXReduceSumOp sumOp;
    {
      SmallVector<Type, 4> types;
      for (auto v : opResults) {
        types.emplace_back(v.getType());
      }

      SmallVector<Value, 1> values;
      values.emplace_back(mulOp.getResult());

      SmallVector<NamedAttribute, 4> attrs;
      for (auto attr : opAttrs) {
        attrs.emplace_back(attr);
      }
      sumOp = rewriter.create<ONNXReduceSumOp>(loc, types, values, attrs);
    }

    rewriter.replaceOp(op, sumOp.getResult());
    return matchSuccess();
  };
};

//===----------------------------------------------------------------------===//
// Rewrite:
// %0 = onnx.ConvNoBiasOp(%D : tensor<DShape>, %K)
//     {pads = [b0, b1, ... bK, e0, e1, ..., eK]} ->
//         tensor<OutShape>
//
// as:
// %0 = onnx.PadConstantValuePasOp(%D)
//     {pads = [0, 0, b0, b1, ... bK, 0, 0, e0, e1, ..., eK]} ->
//     tensor<DPaddedShape>
// %1 = onnx.ConvNoBias(%0 : tensor<DPaddedShape>, %K) {pads = [0, ..., 0]} ->
//     tensor<OutShape>
//===----------------------------------------------------------------------===//
struct SplitConvOpPattern : public RewritePattern {
  SplitConvOpPattern(MLIRContext *context)
      : RewritePattern(ONNXConvNoBiasOp::getOperationName(),
                       {ONNXPadConstantValuePadOp::getOperationName(),
                        ONNXConvNoBiasOp::getOperationName()},
                       1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // If convolution does not use padding then no rewrite is required.
    ONNXConvNoBiasOp convOp = llvm::dyn_cast<ONNXConvNoBiasOp>(op);
    auto padsAttribute = convOp.padsAttr();
    if (!padsAttribute)
      return matchFailure();

    // If auto_pad is VALID then no padding happens and no rewrite isrequired.
    auto autoPad = convOp.auto_pad();
    if (autoPad == "VALID")
      return matchFailure();

    auto data = op->getOperands()[0];
    auto inputShape = data.getType().cast<TensorType>().getShape();

    // Dimensionality of the input:
    //              inputRank
    //      |----------------------|
    // D : (N x C x D1 x D2 x ... DK)
    //              |______________|
    //                  inputDims
    //
    int64_t inputRank = inputShape.size();
    int64_t inputDims = inputRank - 2;

    // If all pads values are equal to zero then no rewrite is required.
    bool allZeros = true;
    for (auto padsValue : padsAttribute.getValue()) {
      if (padsValue.cast<IntegerAttr>().getInt() > 0) {
        allZeros = false;
        break;
      }
    }

    if (allZeros)
      return matchFailure();

    // Create padding vector for the explicit padding op attribute.
    SmallVector<int64_t, 4> pads(2 * inputRank, 0);
    SmallVector<int64_t, 4> outPaddedShape(inputRank, 0);
    outPaddedShape[0] = inputShape[0];
    outPaddedShape[1] = inputShape[1];
    for (int i = 0; i < inputDims; ++i) {
      int64_t beginPad =
          padsAttribute.getValue()[i].cast<IntegerAttr>().getInt();
      int64_t endPad =
          padsAttribute.getValue()[inputDims + i].cast<IntegerAttr>().getInt();
      pads[i + 2] = beginPad;
      pads[inputRank + i + 2] = endPad;
      outPaddedShape[i + 2] += beginPad + inputShape[i + 2] + endPad;
    }

    // Create padding operation.
    auto inputElemType = data.getType().cast<TensorType>().getElementType();
    ONNXPadConstantValuePadOp paddingOp =
        rewriter.create<ONNXPadConstantValuePadOp>(
            loc, RankedTensorType::get(outPaddedShape, inputElemType), data,
            rewriter.getI64ArrayAttr(pads), FloatAttr::get(inputElemType, 0),
            StringAttr::get("constant", loc->getContext()));

    SmallVector<int64_t, 4> newConvPads(2 * inputDims, 0);
    auto tensorType = (*op->result_type_begin()).cast<TensorType>();
    ONNXConvNoBiasOp newConvOp = rewriter.create<ONNXConvNoBiasOp>(
            loc, tensorType, paddingOp.getResult(), convOp.getOperands()[1],
            convOp.auto_padAttr(), convOp.dilationsAttr(),
            convOp.groupAttr(), convOp.kernel_shapeAttr(),
            rewriter.getI64ArrayAttr(newConvPads),
            convOp.stridesAttr());

    rewriter.replaceOp(op, newConvOp.getResult());
    return matchSuccess();
  };
};
} // end anonymous namespace

/// on the ONNXReduceL1Op.
void ONNXReduceL1Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ReduceL1OpPattern>(context);
}
/// on the ONNXReduceL2Op.
void ONNXReduceL2Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ReduceL2OpPattern>(context);
}

/// on the ONNXReduceLogSumOp.
void ONNXReduceLogSumOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ReduceLogSumOpPattern>(context);
}

/// on the ONNXReduceLogSumExpOp.
void ONNXReduceLogSumExpOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ReduceLogSumExpOpPattern>(context);
}

/// on the ONNXReduceSumSquareOp.
void ONNXReduceSumSquareOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ReduceSumSquareOpPattern>(context);
}

/// on the ONNXReduceSumSquareOp.
void ONNXConvNoBiasOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SplitConvOpPattern>(context);
}
