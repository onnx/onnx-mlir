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

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper
//===----------------------------------------------------------------------===//

/// When defining support for a new op, add one such stuct which mÍÎust minimally
/// compute the outputDims present in the parent class. Computation should be
/// performed using a `Compute` function. Return success on successful
/// computation of all the IndexExpr. During shape inference, object is built
/// using a null-ptr rewriter; during lowering, the rewriter is nonnull and will
/// be used to generate code.
template <class OP>
struct ONNXOpShapeHelper {
  ONNXOpShapeHelper(OP *newOp, ConversionPatternRewriter *rewriter);

  // Define in every children.
  LogicalResult Compute(ONNXSliceOpAdaptor operandAdaptor) {
    llvm_unreachable("implement in child structs");
  }

  // Data that must be present for every ShapeHelper operation. Op and context
  // are initialized in the constructor, and outputDims is computed by the
  // child's struct `Compute` function.
  OP *op;
  IndexExprContext context;
  SmallVector<IndexExpr, 4> outputDims;
};

struct ONNXSliceOpShapeHelper : public ONNXOpShapeHelper<ONNXSliceOp> {
  ONNXSliceOpShapeHelper(
      ONNXSliceOp *newOp, ConversionPatternRewriter *rewriter);

  LogicalResult Compute(ONNXSliceOpAdaptor operandAdaptor);

  // Additional data for SliceOp.
  SmallVector<IndexExpr, 4> starts;
  SmallVector<IndexExpr, 4> ends;
  SmallVector<IndexExpr, 4> steps;
};

//===----------------------------------------------------------------------===//
// Low Level Helpers
//===----------------------------------------------------------------------===//

size_t ArrayAttrSize(ArrayAttr a);
size_t ArrayAttrSize(Optional<ArrayAttr> a);
int64_t ArrayAttrIntVal(ArrayAttr a, int i);
int64_t ArrayAttrIntVal(Optional<ArrayAttr> a, int i);
// Returns the ConstantOp which defines an MLIR Value or null.
ONNXConstantOp getONNXConstantOp(Value value);

DenseElementsAttr getDenseElementAttributeFromValue(Value value);

bool getIntegerLiteralFromValue(Value value, int64_t &intLit);
