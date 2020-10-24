//===----------------ONNXShapeHelper.hpp - help for shapes----------------=== //
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file has the computations to compute the shapes using the new index expr
// approach.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "src/Dialect/ONNX/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

size_t ArrayAttrSize(ArrayAttr a);
size_t ArrayAttrSize(Optional<ArrayAttr> a);
int64_t ArrayAttrIntVal(ArrayAttr a, int i);
int64_t ArrayAttrIntVal(Optional<ArrayAttr> a, int i);
// Returns the ConstantOp which defines an MLIR Value or null.
ONNXConstantOp getONNXConstantOp(Value value);

// Create an index from reading from an array defined by operand. Get a literal
// value if we can, otherwise create the code to read the array. Returns
// Undefined IndexExpression on failure. Second call returns a default value
// when no actual value was found.
IndexExpr GetIndexExprFromArrayAt(
    IndexExprContext &context, Operation *op, Value operand, uint64_t i);
IndexExpr GetIndexExprFromArrayAt(IndexExprContext &context, Operation *op,
    Value operand, uint64_t i, int64_t defaultIntLit);

bool getIntegerLiteralFromValue(Value value, int64_t &intLit);

LogicalResult HandleSliceOpParams(ONNXSliceOp *sliceOp,
    ONNXSliceOpAdaptor operandAdaptor, IndexExprContext &context,
    SmallVectorImpl<IndexExpr> &startIndices,
    SmallVectorImpl<IndexExpr> &endIndices,
    SmallVectorImpl<IndexExpr> &stepIndices,
    SmallVectorImpl<IndexExpr> &outputDims);
