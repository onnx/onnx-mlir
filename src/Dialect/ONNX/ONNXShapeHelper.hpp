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

#include "mlir/IR/Operation.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "src/Dialect/ONNX/IndexExpr.hpp"
#include "ONNXOps.hpp"

using namespace mlir;

size_t ArrayAttrSize(ArrayAttr a);
size_t ArrayAttrSize(Optional<ArrayAttr> a);
int64_t ArrayAttrIntVal(ArrayAttr a, int i);
int64_t ArrayAttrIntVal(Optional<ArrayAttr> a, int i);
// Returns the ConstantOp which defines an MLIR Value or null.
ONNXConstantOp getONNXConstantOp(Value value); 

LogicalResult GetIndexExprFromOperandValueAtIndex(
    Operation *op, Value operand, uint64_t i, IndexExpr &indexExpr);

LogicalResult GetIndexExprFromOperandValueAtIndex(Operation *op, Value operand,
    uint64_t i, int64_t defaultIntLit, IndexExpr &indexExpr);

LogicalResult HandleSliceOpParams(ONNXSliceOp *sliceOp,
    ONNXSliceOpAdaptor operandAdaptor, 
    IndexExprContainer &container,
    SmallVectorImpl<IndexExpr> &startIndices,
    SmallVectorImpl<IndexExpr> &endIndices,
    SmallVectorImpl<IndexExpr> &stepIndices,
    SmallVectorImpl<IndexExpr> &outputDims);


