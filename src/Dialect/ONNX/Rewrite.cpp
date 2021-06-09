/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXRewrite.cpp - ONNX High Level Optimizer --------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters for operations in the ONNX dialect
// that can be rewritten by using other ONNX operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"

using namespace mlir;

namespace {

// If 'lhs' is not NoneType, return 'lhs - rhs'.
// Otherwise, return '-rhs'.
Value subtractOrNeg(
    PatternRewriter &rewriter, Location loc, Value lhs, Value rhs) {
  if (lhs.getType().isa<NoneType>()) {
    Value result = rewriter.create<ONNXNegOp>(loc, rhs);
    return result;
  } else {
    Value result = rewriter.create<ONNXSubOp>(loc, lhs, rhs);
    return result;
  }
}

// Create an ArrayAttr of IntergerAttr(s) of values in [1, N].
ArrayAttr createArrayAttrOfOneToN(PatternRewriter &rewriter, int N) {
  SmallVector<int64_t, 4> vals;
  for (int i = 1; i <= N; ++i)
    vals.emplace_back(i);
  return rewriter.getI64ArrayAttr(vals);
}

// Create an ArrayAttr of IntergerAttr(s) of values in [N, M].
ArrayAttr createArrayAttrOfNToM(PatternRewriter &rewriter, int N, int M) {
  SmallVector<int64_t, 4> vals;
  for (int i = N; i <= M; ++i)
    vals.emplace_back(i);
  return rewriter.getI64ArrayAttr(vals);
}
// Check whether an ArrayAttr contains non-zero values or not.
bool hasNonZeroInArrayAttr(ArrayAttr attrs) {
  bool allZeros = true;
  if (attrs) {
    for (auto attr : attrs.getValue()) {
      if (attr.cast<IntegerAttr>().getInt() > 0) {
        allZeros = false;
        break;
      }
    }
  }
  return !allZeros;
}

// Create an ArrayAttr of IntergerAttr(s) of zero values.
// This function is used for padding attribute in Conv.
ArrayAttr createArrayAttrOfZeros(
    PatternRewriter &rewriter, ArrayAttr origAttrs) {
  int nElements = origAttrs.getValue().size();
  SmallVector<int64_t, 4> vals(nElements, 0);
  return rewriter.getI64ArrayAttr(vals);
}

DenseElementsAttr createDenseFloatAttrOfValue(
    PatternRewriter &rewriter, Value origValue, float constantValue) {
  Type elementType = origValue.getType().cast<TensorType>().getElementType();
  SmallVector<float, 1> wrapper(1, 0);
  wrapper[0] = constantValue;
  return DenseElementsAttr::get(
      RankedTensorType::get(wrapper.size(), elementType),
      llvm::makeArrayRef(wrapper));
}

// Pad a ArrayAttr with zeros.
//
// pads = [B1, B2, ... Bk, E1, E2, ..., Ek]
//
// becomes:
//
// pads = [0,... 0, B1, B2, ... Bk, 0,... 0, E1, E2, ..., Ek]
//         |_____|                  |_____|
//                 nZeros                    nZeros
//
// This function is used for padding attribute in Conv.
DenseElementsAttr insertZerosForNonPaddedDims(
    PatternRewriter &rewriter, ArrayAttr origAttrs, int extensionLength) {
  int nDims = (int)origAttrs.getValue().size() / 2;
  int nElements = (nDims + extensionLength) * 2;
  SmallVector<int64_t, 4> pads(nElements, 0);
  for (int i = 0; i < nDims; ++i) {
    int64_t beginPad = origAttrs.getValue()[i].cast<IntegerAttr>().getInt();
    int64_t endPad =
        origAttrs.getValue()[nDims + i].cast<IntegerAttr>().getInt();
    pads[i + extensionLength] = beginPad;
    pads[nDims + extensionLength + i + extensionLength] = endPad;
  }

  mlir::Type elementType = rewriter.getIntegerType(64);
  llvm::ArrayRef<int64_t> tensorDims(pads.data(), pads.size());
  mlir::ShapedType tensorType =
      mlir::RankedTensorType::get(tensorDims, elementType);
  return rewriter.getI64TensorAttr(llvm::makeArrayRef(pads));
}

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Dialect/ONNX/ONNXRewrite.inc"

} // end anonymous namespace

/// on the ONNXBatchNormalizationTestModeOp.
void ONNXBatchNormalizationTestModeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<FuseBatchNormTestModeConvPattern>(context);
}

/// on the ONNXShapeOp.
void ONNXShapeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<ShapeToConstantPattern>(context);
}

/// on the ONNXSizeOp.
void ONNXSizeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<SizeToConstantPattern>(context);
}

/// on the ONNXGlobalAveragePoolOp.
void ONNXGlobalAveragePoolOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<GlobalAveragePoolPattern>(context);
}

/// on the ONNXGlobalMaxPoolOp.
void ONNXGlobalMaxPoolOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<GlobalMaxPoolPattern>(context);
}

/// on the ONNXSizeOp.
void ONNXConstantOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<ConstantOpNormalizationPattern1>(context);
  results.insert<ConstantOpNormalizationPattern2>(context);
  results.insert<ConstantOpNormalizationPattern3>(context);
  results.insert<ConstantOpNormalizationPattern4>(context);
  results.insert<ConstantOpNormalizationPattern5>(context);
  results.insert<ConstantOpNormalizationPattern6>(context);
}
