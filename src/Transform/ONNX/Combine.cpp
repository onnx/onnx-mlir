//===--------- ONNXCombine.cpp - ONNX High Level Optimizer ----------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of simple combiners for optimizing operations in
// the ONNX dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include <numeric>

using namespace mlir;

namespace {
// // Check whether an ArrayAttr contains non-zero values or not.
// bool HasSameType(Value origValue, IntegerAttr attr) {
// 	Type input_elementType = origValue.getType().dyn_cast<ShapedType>().getElementType()
// 	int64_t targetType = attr.getValue().getInt();
// 	Type elementType = convertONNXTypeToMLIRType(
// 	  builder, static_cast<onnx::TensorProto_DataType>(targetType))
// 	if (input_elementType == elementType) 	return true;	     
// 	else return false;	      
// }

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Transform/ONNX/ONNXCombine.inc"
} // end anonymous namespace

/// Register optimization patterns as "canonicalization" patterns
/// on the ONNXMatMultOp.
void ONNXAddOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<MulAddToGemmOptPattern>(context);
}

void ONNXGemmOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<FuseGemmFollowedByAddition>(context);
}
/// on the ONNXIdentityOp.
void ONNXIdentityOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<IdentityEliminationPattern>(context);
}

/// on the ONNXPadConstantValueOp.
void ONNXPadConstantValueOp::getCanonicalizationPatterns(
    OwningRewritePatternList &result, MLIRContext *context) {
  result.insert<ConstantPadPattern>(context);
}

/// on the ONNXCastOp.
void ONNXCastOp::getCanonicalizationPatterns(
    OwningRewritePatternList &result, MLIRContext *context) {
  result.insert<CastEliminationPattern>(context);
}