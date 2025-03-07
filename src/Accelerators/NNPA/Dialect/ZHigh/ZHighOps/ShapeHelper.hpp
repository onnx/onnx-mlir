/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------ShapeHelper.hpp - shape helpers for ZHigh ------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains shape computation for ZHigh operations.
// IndexExp is used in order to handle both static and dynamic shapes.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ZHIGH_SHAPE_HELPER_H
#define ONNX_MLIR_ZHIGH_SHAPE_HELPER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// Shape helper for Stick/Unstick/MeanReduce2D ops.
//===----------------------------------------------------------------------===//

#define DECLARE_SHAPE_HELPER_ZHIGH(SHAPE_HELPER)                               \
  class SHAPE_HELPER : public ONNXOpShapeHelper {                              \
  public:                                                                      \
    SHAPE_HELPER(mlir::Operation *op,                                          \
        mlir::ArrayRef<mlir::Value> operands = {},                             \
        IndexExprBuilder *ieBuilder = nullptr,                                 \
        IndexExprScope *scope = nullptr)                                       \
        : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}                 \
    virtual ~SHAPE_HELPER() {}                                                 \
    mlir::LogicalResult computeShape() final;                                  \
  };
DECLARE_SHAPE_HELPER_ZHIGH(ZHighDLF16ToF32OpShapeHelper)
DECLARE_SHAPE_HELPER_ZHIGH(ZHighF32ToDLF16OpShapeHelper)
DECLARE_SHAPE_HELPER_ZHIGH(ZHighFixGRUYhOpShapeHelper)
DECLARE_SHAPE_HELPER_ZHIGH(ZHighMeanReduce2DOpShapeHelper)
DECLARE_SHAPE_HELPER_ZHIGH(ZHighStickForGRUOpShapeHelper)
DECLARE_SHAPE_HELPER_ZHIGH(ZHighStickForLSTMOpShapeHelper)
DECLARE_SHAPE_HELPER_ZHIGH(ZHighStickifiedConstantOfShapeOpShapeHelper)
DECLARE_SHAPE_HELPER_ZHIGH(ZHighStickOpShapeHelper)
DECLARE_SHAPE_HELPER_ZHIGH(ZHighQuantizedStickOpShapeHelper)
DECLARE_SHAPE_HELPER_ZHIGH(ZHighUnstickOpShapeHelper)
DECLARE_SHAPE_HELPER_ZHIGH(ZHighReshapeOpShapeHelper)
#undef DECLARE_SHAPE_HELPER_ZHIGH

//===----------------------------------------------------------------------===//
// Shape helper for MatMulOp.
//===----------------------------------------------------------------------===//

struct ZHighMatMulOpShapeHelper : public ONNXOpShapeHelper {
  ZHighMatMulOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands = {},
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ZHighMatMulOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Broadcast 1 case: X:2D - Y:3DS
  bool isBroadcasted1 = false;
  // Broadcast 23 case: X:3DS - Y:2D
  bool isBroadcasted23 = false;
  // Stack case: X:3DS - Y:3DS
  bool isStacked = false;
  // Keep original dimensions in this order: m, n, p if 2D or s, m, n, p if 3D.
  DimsExpr allOriginalDims;
};

//===----------------------------------------------------------------------===//
// Shape helper for QuantizedMatMulOp.
//===----------------------------------------------------------------------===//

struct ZHighQuantizedMatMulOpShapeHelper : public ONNXOpShapeHelper {
  ZHighQuantizedMatMulOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands = {},
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ZHighQuantizedMatMulOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Broadcast case: X:3DS - Y:2D
  bool isBroadcasted = false;
  // Stack case: X:3DS - Y:3DS
  bool isStacked = false;
  // Keep original dimensions in this order: m, n, p if 2D or s, m, n, p if 3D.
  DimsExpr allOriginalDims;
};

//===----------------------------------------------------------------------===//
// Shape helper for LSTMOp.
//===----------------------------------------------------------------------===//

struct ZHighLSTMOpShapeHelper : public ONNXOpShapeHelper {
  ZHighLSTMOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands = {},
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ZHighLSTMOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Used to initialize optional biases.
  DimsExpr hc0Shape, biasShape;
  // Keep original dimensions in this order: direction, steps, batchsize,
  // inputsize, hiddensize
  DimsExpr allOriginalDims;
};

//===----------------------------------------------------------------------===//
// Shape helper for GRUOp.
//===----------------------------------------------------------------------===//

struct ZHighGRUOpShapeHelper : public ONNXOpShapeHelper {
  ZHighGRUOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands = {},
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ZHighGRUOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Used to initialize optional biases.
  DimsExpr h0Shape, biasShape;
  // Keep original dimensions in this order: direction, steps, batchsize,
  // inputsize, hiddensize
  DimsExpr allOriginalDims;
};

//===----------------------------------------------------------------------===//
// Shape helper for Conv2DOp.
//===----------------------------------------------------------------------===//

struct ZHighConv2DOpShapeHelper : public ONNXOpShapeHelper {
  ZHighConv2DOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands = {},
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ZHighConv2DOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Keep original dimensions in this order: batchsize, channel_in, height_in,
  // weight_in, channel_out, height_out, weight_out.
  DimsExpr allOriginalDims;
};

//===----------------------------------------------------------------------===//
// Shape helper for PoolingOp.
//===----------------------------------------------------------------------===//

template <typename OP>
struct ZHighPoolingOpShapeHelper : public ONNXOpShapeHelper {
  ZHighPoolingOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands = {},
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ZHighPoolingOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Keep original dimensions in this order: batchsize, channel_in, height_in,
  // weight_in, height_out, weight_out.
  DimsExpr allOriginalDims;
};

//===----------------------------------------------------------------------===//
// Shape helper for ReductionOp.
//===----------------------------------------------------------------------===//

template <typename OP_TYPE>
struct ZHighReductionOpShapeHelper : public ONNXOpShapeHelper {
  ZHighReductionOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands = {},
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ZHighReductionOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
};

using ZHighReduceMaxOpShapeHelper =
    ZHighReductionOpShapeHelper<ZHighReduceMaxOp>;
using ZHighReduceMinOpShapeHelper =
    ZHighReductionOpShapeHelper<ZHighReduceMinOp>;

//===----------------------------------------------------------------------===//
// Shape helper for UnaryOp.
//===----------------------------------------------------------------------===//

struct ZHighUnaryOpShapeHelper : public ONNXUnaryOpShapeHelper {
public:
  ZHighUnaryOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands = {},
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXUnaryOpShapeHelper(op, operands, ieBuilder, scope) {}
};

//===----------------------------------------------------------------------===//
// Shape helper for BinaryOp.
// ZHigh BinaryOps do not support broadcasting at this moment. Borrow UnaryOp
// shapeHelper.
//===----------------------------------------------------------------------===//

struct ZHighBinaryOpShapeHelper : public ONNXUnaryOpShapeHelper {
public:
  ZHighBinaryOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands = {},
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXUnaryOpShapeHelper(op, operands, ieBuilder, scope) {}
};

using ZHighFixGRUYOpShapeHelper = ONNXUnaryOpShapeHelper;

} // namespace zhigh
} // namespace onnx_mlir
#endif
