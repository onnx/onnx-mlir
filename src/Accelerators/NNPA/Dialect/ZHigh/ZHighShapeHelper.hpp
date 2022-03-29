/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------ZHighShapeHelper.hpp - help for shapes ---------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains shape computation for ZHigh operations.
// IndexExp is used in order to handle both static and dynamic shapes.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighHelper.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Dialect/ONNX/IndexExpr.hpp"

namespace onnx_mlir {
namespace zhigh {

// Steps to add a new op XXX:
// 1) Create a new shape inference type inside this file, ZHighShapeHelper.hpp.
// 2) Create new shape inference implementation inside ZHighShapeHelper.cpp.
// 3) Use the new object in ZHighOps.cpp and ZHighToZLog lowering for XXX.

//===----------------------------------------------------------------------===//
// ZHigh Op Shape Helper
//
// It would be ideal if we can direclty use the base class, ONNXOpShapeHelper.
// However, the current implemention of ONNXOpShapeHelper requires adding the
// newly defined struct into "ONNXShapeHelper.cpp". So we redefine a similar
// base class for ZHigh here.
//===----------------------------------------------------------------------===//

using DimsExpr = llvm::SmallVector<mlir::IndexExpr, 4>;

/// When defining support for a new op, add one such stuct which must
/// minimally compute the outputDims present in the parent class. Computation
/// should be performed using a `computeShape` function. Return success on
/// successful computation of all the IndexExpr. During shape inference, object
/// is built using a null-ptr rewriter; during lowering, the rewriter is nonnull
/// and will be used to generate code.
///
/// By adding here the ability of a ShapeHelper to be created in the
/// IndexExprScope of another ShapeHelper, this enables us to nest ShapeHelper.
/// For example, there is a case where ExpandOp needs to find out specific
/// details of an ShapeOp that provides info to the ExpandOp. We can now invoke
/// the ShapeOp shape helper in the context of the ExpandOp shape helper while
/// having all of the IndexExpr info in the same context and thus be generally
/// usable. Support is here to provide an IndexExprScope, which can be added to
/// any subclasses of ZHighOpShapeHelper when this nesting becomes useful to
/// other ops as well.

template <class OP>
struct ZHighOpShapeHelper {
  // Constructor for shape inference. Reuse scope if given, otherwise create one
  // now and free it in destructor.
  ZHighOpShapeHelper(
      OP *newOp, int numResults, mlir::IndexExprScope *inScope = nullptr);
  // Constructor when code can be generated. Reuse scope if given, otherwise
  // create one now and free it in destructor.
  ZHighOpShapeHelper(OP *newOp, int numResults, mlir::OpBuilder *rewriter,
      mlir::ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      mlir::ArrayValueIndexCapture::LoadVal fLoadVal,
      mlir::IndexExprScope *inScope = nullptr);
  ~ZHighOpShapeHelper() {
    if (ownScope)
      delete scope;
  }

  // Every child class is expected to create a computeShape with the following
  // signature. This method is responsible to compute at a minimum the output
  // dims.
  //
  // LogicalResult computeShape(<<OpAdaptor>> operandAdaptor);
  //
  // Use the op to get attributes, and operandAdaptor to get the input/output
  // tensors.

  // Return output dims for the N-th output.
  DimsExpr &dimsForOutput(int n = 0) { return outputsDims[n]; }

  // Set the number of outputs.
  void setNumberOfOutputs(int n) { outputsDims.resize(n); }

  // Data that must be present for every ShapeHelper operation. Op and scope
  // are initialized in the constructor, and outputsDims is computed by the
  // child's struct `computeShape` function.
  OP *op;
  mlir::IndexExprScope *scope;

protected:
  // Function to get a dense value from an attribute.
  mlir::ArrayValueIndexCapture::GetDenseVal fGetDenseVal;
  // Function to load a value from an array.
  mlir::ArrayValueIndexCapture::LoadVal fLoadVal;

private:
  llvm::SmallVector<DimsExpr, 1> outputsDims;
  bool ownScope;
};

// =============================================================================
// Shape helper for StickForLSTMOp.

struct ZHighStickForLSTMOpShapeHelper
    : public ZHighOpShapeHelper<ZHighStickForLSTMOp> {
  ZHighStickForLSTMOpShapeHelper(ZHighStickForLSTMOp *newOp);
  ZHighStickForLSTMOpShapeHelper(
      ZHighStickForLSTMOp *newOp, mlir::OpBuilder *rewriter);
  mlir::LogicalResult computeShape(ZHighStickForLSTMOpAdaptor operandAdaptor);
};

//===----------------------------------------------------------------------===//
// Shape helper for StickForGRUOp.

struct ZHighStickForGRUOpShapeHelper
    : public ZHighOpShapeHelper<ZHighStickForGRUOp> {
  ZHighStickForGRUOpShapeHelper(ZHighStickForGRUOp *newOp);
  ZHighStickForGRUOpShapeHelper(
      ZHighStickForGRUOp *newOp, mlir::OpBuilder *rewriter);
  mlir::LogicalResult computeShape(ZHighStickForGRUOpAdaptor operandAdaptor);
};

//===----------------------------------------------------------------------===//
// Shape helper for MatMulOp.

struct ZHighMatMulOpShapeHelper : public ZHighOpShapeHelper<ZHighMatMulOp> {
  ZHighMatMulOpShapeHelper(ZHighMatMulOp *newOp);
  ZHighMatMulOpShapeHelper(ZHighMatMulOp *newOp, mlir::OpBuilder *rewriter);
  mlir::LogicalResult computeShape(ZHighMatMulOpAdaptor operandAdaptor);
  // Broadcast case: X:3DS - Y:2D
  bool isBroadcasted = false;
  // Stack case: X:3DS - Y:3DS
  bool isStacked = false;
  // Keep original dimensions in this order: m, n, p if 2D or s, m, n, p if 3D.
  DimsExpr allOriginalDims;
};

//===----------------------------------------------------------------------===//
// Shape helper for LSTMOp.

struct ZHighLSTMOpShapeHelper : public ZHighOpShapeHelper<ZHighLSTMOp> {
  ZHighLSTMOpShapeHelper(ZHighLSTMOp *newOp);
  ZHighLSTMOpShapeHelper(ZHighLSTMOp *newOp, mlir::OpBuilder *rewriter);
  mlir::LogicalResult computeShape(ZHighLSTMOpAdaptor operandAdaptor);
  // Used to initialize optional biases.
  DimsExpr hc0Shape, biasShape;
  // Keep original dimensions in this order: direction, steps, batchsize,
  // inputsize, hiddensize
  DimsExpr allOriginalDims;
};

//===----------------------------------------------------------------------===//
// Shape helper for GRUOp.

struct ZHighGRUOpShapeHelper : public ZHighOpShapeHelper<ZHighGRUOp> {
  ZHighGRUOpShapeHelper(ZHighGRUOp *newOp);
  ZHighGRUOpShapeHelper(ZHighGRUOp *newOp, mlir::OpBuilder *rewriter);
  mlir::LogicalResult computeShape(ZHighGRUOpAdaptor operandAdaptor);
  // Used to initialize optional biases.
  DimsExpr h0Shape, biasShape;
  // Keep original dimensions in this order: direction, steps, batchsize,
  // inputsize, hiddensize
  DimsExpr allOriginalDims;
};

//===----------------------------------------------------------------------===//
// Shape helper for Conv2DOp.

struct ZHighConv2DOpShapeHelper : public ZHighOpShapeHelper<ZHighConv2DOp> {
  ZHighConv2DOpShapeHelper(ZHighConv2DOp *newOp);
  ZHighConv2DOpShapeHelper(ZHighConv2DOp *newOp, mlir::OpBuilder *rewriter);
  mlir::LogicalResult computeShape(ZHighConv2DOpAdaptor operandAdaptor);
  // Keep original dimensions in this order: batchsize, channel_in, height_in,
  // weight_in, channel_out, height_out, weight_out.
  DimsExpr allOriginalDims;
};

//===----------------------------------------------------------------------===//
// Shape helper for PoolingOp.

template <typename OP, typename OP_ADATOR>
struct ZHighPoolingOpShapeHelper : public ZHighOpShapeHelper<OP> {
  ZHighPoolingOpShapeHelper(OP *newOp);
  ZHighPoolingOpShapeHelper(OP *newOp, mlir::OpBuilder *rewriter);
  mlir::LogicalResult computeShape(OP_ADATOR operandAdaptor);
  // Keep original dimensions in this order: batchsize, channel_in, height_in,
  // weight_in, height_out, weight_out.
  DimsExpr allOriginalDims;
};

} // namespace zhigh
} // namespace onnx_mlir
