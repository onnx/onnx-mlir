//===- onnx_ops.cpp - MLIR ONNX Operations --------------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file defines ONNX operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"

#include "onnx_ops.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;

//===----------------------------------------------------------------------===//
// ONNX Helper functions
//===----------------------------------------------------------------------===//

static size_t ArrayAttrSize(ArrayAttr a) { return a.size(); }

static size_t ArrayAttrSize(Optional<ArrayAttr> a) {
  return a.getValue().size();
}

static int64_t ArrayAttrIntVal(ArrayAttr a, int i) {
  return (a.getValue()[i]).cast<IntegerAttr>().getInt();
}

static int64_t ArrayAttrIntVal(Optional<ArrayAttr> a, int i) {
  return (a.getValue().getValue()[i]).cast<IntegerAttr>().getInt();
}

//===----------------------------------------------------------------------===//
// Get reduction type
//===----------------------------------------------------------------------===//
RankedTensorType getReductionOutputType(
    RankedTensorType operandTy, Optional<ArrayAttr> axesAttrs, APInt keepdims) {
  int64_t rank = operandTy.getRank();

  SmallVector<int64_t, 4> axes;
  if (axesAttrs != llvm::None) {
    for (auto axisAttr : axesAttrs.getValue()) {
      int64_t axis = axisAttr.cast<IntegerAttr>().getInt();
      axis = axis >= 0 ? axis : (rank + axis);
      assert(axis >= -rank && axis <= rank - 1);
      if (std::find(axes.begin(), axes.end(), axis) == axes.end())
        axes.emplace_back(axis);
    }
  } else {
    for (decltype(rank) i = 0; i < rank; ++i) {
      axes.emplace_back(i);
    }
  }

  // Mark reduction axes.
  SmallVector<bool, 4> isReductionAxis;
  for (decltype(rank) i = 0; i < rank; ++i) {
    if (std::find(axes.begin(), axes.end(), i) != axes.end())
      isReductionAxis.emplace_back(true);
    else
      isReductionAxis.emplace_back(false);
  }

  // KeepDims
  bool isKeepdims = (keepdims == 1) ? true : false;

  SmallVector<int64_t, 4> dims;
  for (decltype(rank) i = 0; i < rank; ++i) {
    if (isReductionAxis[i]) {
      if (isKeepdims)
        dims.emplace_back(1); // reduction dimension
    } else {
      dims.emplace_back(operandTy.getShape()[i]);
    }
  }

  return RankedTensorType::get(dims, operandTy.getElementType());
}

//===----------------------------------------------------------------------===//
// ONNXOpsDialect
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
ONNXOpsDialect::ONNXOpsDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx) {
  addOperations<
#define GET_OP_LIST
#include "src/onnx.cpp.inc"
      >();
}

void ONNXEntryPointOp::build(mlir::Builder *builder,
    mlir::OperationState &state, mlir::FuncOp function, int numInputs,
    int numOutputs) {
  state.addAttribute(ONNXEntryPointOp::getEntryPointFuncAttrName(),
      builder->getSymbolRefAttr(function));
  state.addAttribute(ONNXEntryPointOp::getNumInputsAttrName(),
      builder->getI32IntegerAttr(numInputs));
  state.addAttribute(ONNXEntryPointOp::getNumOutputsAttrName(),
      builder->getI32IntegerAttr(numOutputs));
}

ONNXEntryPointOp ONNXEntryPointOp::create(mlir::Location location,
    mlir::FuncOp &func, int numInputs, int numOutputs) {
  mlir::OperationState state(location, "onnx.EntryPoint");
  Builder builder(location->getContext());
  mlir::ONNXEntryPointOp::build(&builder, state, func, numInputs, numOutputs);
  Operation *op = mlir::Operation::create(state);
  auto onnxEntryOp = llvm::cast<mlir::ONNXEntryPointOp>(op);
  return onnxEntryOp;
}

//===----------------------------------------------------------------------===//
// ONNX Operations
//===----------------------------------------------------------------------===//
// Exp
/// Infer the output shape of the ONNXExpOp. This method is required by the
/// shape inference interface.
void ONNXExpOp::inferShapes() { getResult().setType(getOperand().getType()); }

//===----------------------------------------------------------------------===//
// Tanh
/// Infer the output shape of the ONNXTanhOp. This method is required by the
/// shape inference interface.
void ONNXTanhOp::inferShapes() { getResult().setType(getOperand().getType()); }

//===----------------------------------------------------------------------===//
// Sinh
/// Infer the output shape of the ONNXSinhOp. This method is required by the
/// shape inference interface.
void ONNXSinhOp::inferShapes() { getResult().setType(getOperand().getType()); }

//===----------------------------------------------------------------------===//
// Cosh
/// Infer the output shape of the ONNXCoshOp. This method is required by the
/// shape inference interface.
void ONNXCoshOp::inferShapes() { getResult().setType(getOperand().getType()); }

//===----------------------------------------------------------------------===//
// Cos
/// Infer the output shape of the ONNXCosOp. This method is required by the
/// shape inference interface.
void ONNXCosOp::inferShapes() { getResult().setType(getOperand().getType()); }

//===----------------------------------------------------------------------===//
// Log
/// Infer the output shape of the ONNXLogOp. This method is required by the
/// shape inference interface.
void ONNXLogOp::inferShapes() { getResult().setType(getOperand().getType()); }

//===----------------------------------------------------------------------===//
// HardSigmoid
/// Infer the output shape of the ONNXHardSigmoidOp. This method is required by
/// the shape inference interface.
void ONNXHardSigmoidOp::inferShapes() {
  getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// Sigmoid
/// Infer the output shape of the ONNXSigmoidOp. This method is required by the
/// shape inference interface.
void ONNXSigmoidOp::inferShapes() {
  getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// Elu
/// Infer the output shape of the ONNXEluOp. This method is required by the
/// shape inference interface.
void ONNXEluOp::inferShapes() { getResult().setType(getOperand().getType()); }

//===----------------------------------------------------------------------===//
// Relu
/// Infer the output shape of the ONNXReluOp. This method is required by the
/// shape inference interface.
void ONNXReluOp::inferShapes() { getResult().setType(getOperand().getType()); }

//===----------------------------------------------------------------------===//
// LeakyRelu
/// Infer the output shape of the ONNXLeakyReluOp. This method is required by
/// the shape inference interface.
void ONNXLeakyReluOp::inferShapes() {
  getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// Selu
/// Infer the output shape of the ONNXSeluOp. This method is required by
/// the shape inference interface.
void ONNXSeluOp::inferShapes() { getResult().setType(getOperand().getType()); }

//===----------------------------------------------------------------------===//
// Reciprocal
/// Infer the output shape of the ONNXReciprocalOp. This method is required by
/// the shape inference interface.
void ONNXReciprocalOp::inferShapes() {
  getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// Softmax
/// Infer the output shape of the ONNXSoftmaxOp. This method is required by
/// the shape inference interface.
void ONNXSoftmaxOp::inferShapes() {
  getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// Softplus
/// Infer the output shape of the ONNXSoftplusOp. This method is required by
/// the shape inference interface.
void ONNXSoftplusOp::inferShapes() {
  getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// Softsign
/// Infer the output shape of the ONNXSoftsignOp. This method is required by
/// the shape inference interface.
void ONNXSoftsignOp::inferShapes() {
  getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// Sqrt
/// Infer the output shape of the ONNXSqrtOp. This method is required by
/// the shape inference interface.
void ONNXSqrtOp::inferShapes() { getResult().setType(getOperand().getType()); }

//===----------------------------------------------------------------------===//
// Sign
/// Infer the output shape of the ONNXSignOp. This method is required by
/// the shape inference interface.
void ONNXSignOp::inferShapes() { getResult().setType(getOperand().getType()); }

//===----------------------------------------------------------------------===//
// Add
/// Infer the output shape of the ONNXAddOp. This method is required by the
/// shape inference interface.
void ONNXAddOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return;
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
}

//===----------------------------------------------------------------------===//
// Mul
/// Infer the output shape of the ONNXMulOp. This method is required by the
/// shape inference interface.
void ONNXMulOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return;
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
}

//===----------------------------------------------------------------------===//
// Div
/// Infer the output shape of the ONNXDivOp. This method is required by the
/// shape inference interface.
void ONNXDivOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return;
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
}

//===----------------------------------------------------------------------===//
// Sub
/// Infer the output shape of the ONNXSubOp. This method is required by the
/// shape inference interface.
void ONNXSubOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return;
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
}

//===----------------------------------------------------------------------===//
// And
/// Infer the output shape of the ONNXAndOp. This method is required by the
/// shape inference interface.
void ONNXAndOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return;
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
}

//===----------------------------------------------------------------------===//
// Or
/// Infer the output shape of the ONNXOrOp. This method is required by the
/// shape inference interface.
void ONNXOrOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return;
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
}

//===----------------------------------------------------------------------===//
// Xor
/// Infer the output shape of the ONNXXorOp. This method is required by the
/// shape inference interface.
void ONNXXorOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return;
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
}

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Sum
/// Infer the output shape of the ONNXSumOp. This method is required by the
/// shape inference interface.
void ONNXSumOp::inferShapes() {
  for (int i = 0; i < getNumOperands(); ++i) {
    if (!getOperand(i).getType().cast<RankedTensorType>())
      return;
  }
  Type resultTy = getOperand(0).getType().cast<RankedTensorType>();
  for (int i = 1; i < getNumOperands(); ++i) {
    Type nextTy = getOperand(i).getType().cast<RankedTensorType>();
    resultTy = getBroadcastedType(resultTy, nextTy);
  }
  getResult().setType(resultTy);
}

//===----------------------------------------------------------------------===//
// Max
/// Infer the output shape of the ONNXMaxOp. This method is required by the
/// shape inference interface.
void ONNXMaxOp::inferShapes() {
  for (int i = 0; i < getNumOperands(); ++i) {
    if (!getOperand(i).getType().cast<RankedTensorType>())
      return;
  }
  Type resultTy = getOperand(0).getType().cast<RankedTensorType>();
  for (int i = 1; i < getNumOperands(); ++i) {
    Type nextTy = getOperand(i).getType().cast<RankedTensorType>();
    resultTy = getBroadcastedType(resultTy, nextTy);
  }
  getResult().setType(resultTy);
}

//===----------------------------------------------------------------------===//
// Min
/// Infer the output shape of the ONNXMinOp. This method is required by the
/// shape inference interface.
void ONNXMinOp::inferShapes() {
  for (int i = 0; i < getNumOperands(); ++i) {
    if (!getOperand(i).getType().cast<RankedTensorType>())
      return;
  }
  Type resultTy = getOperand(0).getType().cast<RankedTensorType>();
  for (int i = 1; i < getNumOperands(); ++i) {
    Type nextTy = getOperand(i).getType().cast<RankedTensorType>();
    resultTy = getBroadcastedType(resultTy, nextTy);
  }
  getResult().setType(resultTy);
}

//===----------------------------------------------------------------------===//
// Identity
/// Infer the output shape of the ONNXIdentityOp. This method is required by the
/// shape inference interface.
void ONNXIdentityOp::inferShapes() {
  getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//

// MatMul

void ONNXMatMulOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return;

  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();

  SmallVector<int64_t, 2> dims;
  auto lhsShape = lhsTy.getShape();
  auto rhsShape = rhsTy.getShape();

  if (lhsShape.size() < 1 && rhsShape.size() < 1) {
    // Multiplication by scalars is not allowed.
    emitError("Multiplication by scalar arguments not allowed.");
  } else if (lhsShape.size() == 1 && rhsShape.size() == 1) {
    // Special case when both arrays are 1-dimensional and according to
    // numpy rules the types need to be extended to 1xN and Nx1. Helper sizes
    // need to be removed after the multiplication but cannot be removed if all
    // sizes are 1.
    if (lhsShape[0] != -1 && rhsShape[0] != -1 && lhsShape[0] != rhsShape[0])
      emitError("Attempt to multiply incompatible matrices.");
    dims.emplace_back(1);
  } else if (lhsShape.size() == 1 && rhsShape.size() >= 2) {
    // If the first argument is 1-D, it is promoted to a matrix by prepending a
    // 1 to its dimensions. After matrix multiplication the prepended 1 is
    // removed.
    //
    // N MATMUL (s1 x s2 x... x sK x N x P)
    // =>
    // (s1 x s2 x... x sK x P)

    // Check legality of matrix multiplication.
    unsigned rhsRank = rhsShape.size();
    if (lhsShape[0] != -1 && rhsShape[rhsRank - 2] != -1 &&
        lhsShape[0] != rhsShape[rhsRank - 2])
      emitError("Attempt to multiply incompatible matrices.");

    for (decltype(rhsRank) i = 0; i < rhsRank - 2; ++i)
      dims.emplace_back(rhsShape[i]);
    dims.emplace_back(rhsShape[rhsRank - 1]);
  } else if (lhsShape.size() >= 2 && rhsShape.size() == 1) {
    // If the second argument is 1-D, it is promoted to a matrix by appending a
    // 1 to its dimensions. After matrix multiplication the appended 1 is
    // removed.
    //
    // (s1 x s2 x... x sK x M x N) MATMUL N
    // =>
    // (s1 x s2 x... x sK x M)

    // Check legality of matrix multiplication.
    unsigned lhsRank = lhsShape.size();
    if (lhsShape[lhsRank - 1] != -1 && rhsShape[0] != -1 &&
        lhsShape[lhsRank - 1] != rhsShape[0])
      emitError("Attempt to multiply incompatible matrices.");

    for (decltype(lhsRank) i = 0; i < lhsRank - 2; ++i)
      dims.emplace_back(lhsShape[i]);
    dims.emplace_back(lhsShape[lhsRank - 2]);
  } else if (lhsShape.size() > 2 && rhsShape.size() == 2) {
    // (s1 x s2 x... x sK x M x N) MATMUL (N x P)
    // =>
    // (s1 x s2 x... x sK x M x P)

    // Check legality of matrix multiplication.
    unsigned lhsRank = lhsShape.size();
    if (lhsShape[lhsRank - 1] != -1 && rhsShape[0] != -1 &&
        lhsShape[lhsRank - 1] != rhsShape[0])
      emitError("Attempt to multiply incompatible matrices.");

    for (decltype(lhsRank) i = 0; i < lhsRank - 1; ++i)
      dims.emplace_back(lhsShape[i]);
    dims.emplace_back(rhsShape[1]);
  } else if (lhsShape.size() == 2 && rhsShape.size() > 2) {
    // (M x N) MATMUL (s1 x s2 x... x sK x N x P)
    // =>
    // (s1 x s2 x... x sK x M x P)

    // Check legality of matrix multiplication.
    unsigned rhsRank = rhsShape.size();
    if (lhsShape[1] != -1 && rhsShape[rhsRank - 2] != -1 &&
        lhsShape[1] != rhsShape[rhsRank - 2])
      emitError("Attempt to multiply incompatible matrices.");

    for (decltype(rhsRank) i = 0; i < rhsRank - 2; ++i)
      dims.emplace_back(rhsShape[i]);
    dims.emplace_back(lhsShape[0]);
    dims.emplace_back(rhsShape[rhsRank - 1]);
  } else if (lhsShape.size() > 2 && rhsShape.size() > 2) {
    // (s1 x s2 x... x sK x M x N) MATMUL (t1 x t2 x... x tK x N x P)
    // =>
    // (u1 x u2 x... x uK x M x P)

    // Check legality of matrix multiplication.
    unsigned lhsRank = lhsShape.size();
    unsigned rhsRank = rhsShape.size();
    if (lhsShape[lhsRank - 1] != -1 && rhsShape[rhsRank - 2] != -1 &&
        lhsShape[lhsRank - 1] != rhsShape[rhsRank - 2])
      emitError("Attempt to multiply incompatible matrices.");

    // Check and perform broadcasting for the shapes.
    SmallVector<int64_t, 2> lhsBcastShape;
    for (decltype(lhsRank) i = 0; i < lhsRank - 2; ++i)
      lhsBcastShape.emplace_back(lhsShape[i]);
    SmallVector<int64_t, 2> rhsBcastShape;
    for (decltype(rhsRank) i = 0; i < rhsRank - 2; ++i)
      rhsBcastShape.emplace_back(rhsShape[i]);
    if (!getBroadcastedShape(lhsBcastShape, rhsBcastShape, dims))
      emitError("Broadcasted dimensions are incompatible.");

    dims.emplace_back(lhsShape[lhsRank - 2]);
    dims.emplace_back(rhsShape[rhsRank - 1]);
  } else {
    // This case covers all remaining combinations of 1 and 2-D matrices.
    int64_t lhsDim = lhsShape[0];
    int64_t rhsDim = rhsShape[0];
    if (lhsShape.size() > 1) {
      lhsDim = lhsShape[1];
      dims.emplace_back(lhsShape[0]);
    }

    // Check legality of matrix multiplication.
    if (lhsDim != -1 && rhsDim != -1 && lhsDim != rhsDim)
      emitError("Attempt to multiply incompatible matrices.");

    if (rhsShape.size() > 1)
      dims.emplace_back(rhsShape[1]);
  }

  getResult().setType(RankedTensorType::get(dims, lhsTy.getElementType()));
}

//===----------------------------------------------------------------------===//

// Gemm

void ONNXGemmOp::inferShapes() {
  bool hasBias = !getOperand(2).getType().isa<NoneType>();
  // Cannot infer shape if no shape exists.
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>() ||
      (hasBias && !getOperand(2).getType().isa<RankedTensorType>()))
    return;
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();

  int64_t M, N, K_A, K_B;
  M = (transA() == 0) ? lhsTy.getShape()[0] : lhsTy.getShape()[1];
  K_A = (transA() == 0) ? lhsTy.getShape()[1] : lhsTy.getShape()[0];
  N = (transB() == 0) ? rhsTy.getShape()[1] : rhsTy.getShape()[0];
  K_B = (transB() == 0) ? rhsTy.getShape()[0] : rhsTy.getShape()[1];

  if ((K_A != -1) and (K_B != -1) and (K_A != K_B)) {
    emitError("Tensor shapes mismatched.");
  }

  if (hasBias) {
    // Check whether bias is unidirectional broadcasting or not.
    auto biasTy = getOperand(2).getType().cast<RankedTensorType>();
    auto shape = biasTy.getShape();
    int rank = shape.size();
    if ((rank > 2) ||
        (rank >= 1 && shape[rank - 1] != -1 && N != -1 &&
            N != shape[rank - 1] && shape[rank - 1] != 1) ||
        (rank == 2 && shape[rank - 2] != -1 && M != -1 &&
            M != shape[rank - 2] && shape[rank - 2] != 1)) {
      emitError("Bias shape mismatched.");
    }
  }

  SmallVector<int64_t, 2> dims;
  dims.emplace_back(M);
  dims.emplace_back(N);
  getResult().setType(RankedTensorType::get(dims, lhsTy.getElementType()));
}

/// BatchNormalizationTestMode
void ONNXBatchNormalizationTestModeOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>() ||
      !getOperand(2).getType().isa<RankedTensorType>() ||
      !getOperand(3).getType().isa<RankedTensorType>() ||
      !getOperand(4).getType().isa<RankedTensorType>())
    return;

  auto input = getOperand(0).getType().cast<RankedTensorType>();
  auto scale = getOperand(1).getType().cast<RankedTensorType>();
  auto bias = getOperand(2).getType().cast<RankedTensorType>();
  auto mean = getOperand(3).getType().cast<RankedTensorType>();
  auto variance = getOperand(4).getType().cast<RankedTensorType>();

  // Check whether the shapes of scale, bias, mean and variance are valid.
  // Operand's dimensions can be in the form of NxCxD1xD2x...xDn or N.
  // In case of N, C is assumed to be 1.
  // Shapes of scale, bias, mean and variance must be C.
  int64_t c = -1;
  if (input.getShape().size() == 1) {
    c = 1;
  } else if (input.getShape().size() > 2) {
    c = (input.getShape()[1] != -1) ? input.getShape()[1] : -1;
  } else {
    emitError("Wrong rank for the input.");
  }

  if (c != -1) {
    auto s = scale.getShape();
    auto b = bias.getShape();
    auto m = mean.getShape();
    auto v = variance.getShape();

    if ((s.size() != 1) || (s[0] != -1 && s[0] != c))
      emitError("Wrong rank for the scale.");
    if ((b.size() != 1) || (b[0] != -1 && b[0] != c))
      emitError("Wrong rank for the bias.");
    if ((m.size() != 1) || (m[0] != -1 && m[0] != c))
      emitError("Wrong rank for the mean.");
    if ((v.size() != 1) || (v[0] != -1 && v[0] != c))
      emitError("Wrong rank for the variance.");
  }

  // The output tensor of the same shape as the input.
  getResult().setType(getOperand(0).getType());
}

// TODO:
//   Verify that matrix sizes are valid for multiplication and addition.
//   Take into account the dimensionality of the matrix.

//===----------------------------------------------------------------------===//

// Reshape

void ONNXReshapeOp::inferShapes() {
  // Cannot infer shape if no shape tensor is specified.
  if (!getOperand(1).getType().isa<RankedTensorType>())
    emitError("Shape tensor not ranked.");

  auto inputTensorTy = getOperand(0).getType().cast<RankedTensorType>();
  auto shapeTensorTy = getOperand(1).getType().cast<RankedTensorType>();

  // Only rank 1 shape tensors are supported.
  if (shapeTensorTy.getShape().size() != 1)
    emitError("Shape tensor must have rank one.");

  int64_t outputRank = shapeTensorTy.getShape()[0];

  // Shape tensor must have constant shape.
  if (outputRank < 0)
    emitError("Shape tensor must have constant shape.");

  SmallVector<int64_t, 2> dims;
  for (int i = 0; i < outputRank; ++i)
    dims.emplace_back(-1);

  getResult().setType(
      RankedTensorType::get(dims, inputTensorTy.getElementType()));
}

//===----------------------------------------------------------------------===//

// Transpose

void ONNXTransposeOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!getOperand().getType().isa<RankedTensorType>())
    return;

  // Naive transposition which handles the default case of
  // reversing the shape of the tensor (similar to numpy.transpose).
  auto arrayTy = getOperand().getType().cast<RankedTensorType>();
  SmallVector<int64_t, 2> dims;
  auto permutation = ONNXTransposeOp::permAttr();
  if (permutation) {
    // Perform transposition according to perm attribute.
    for (auto perm : permutation.getValue())
      dims.emplace_back(arrayTy.getShape()[perm.cast<IntegerAttr>().getInt()]);
  } else {
    // Default
    for (auto dim : llvm::reverse(arrayTy.getShape()))
      dims.emplace_back(dim);
  }

  getResult().setType(RankedTensorType::get(dims, arrayTy.getElementType()));
}

//===----------------------------------------------------------------------===//

// ReduceMax

void ONNXReduceMaxOp::inferShapes() {
  if (!getOperand().getType().isa<RankedTensorType>()) {
    emitError("Shape tensor not ranked.");
    return;
  }

  auto operandTy = getOperand().getType().cast<RankedTensorType>();
  getResult().setType(getReductionOutputType(operandTy, axes(), keepdims()));
}

//===----------------------------------------------------------------------===//

// ReduceMin

void ONNXReduceMinOp::inferShapes() {
  if (!getOperand().getType().isa<RankedTensorType>()) {
    emitError("Shape tensor not ranked.");
    return;
  }

  auto operandTy = getOperand().getType().cast<RankedTensorType>();
  getResult().setType(getReductionOutputType(operandTy, axes(), keepdims()));
}

//===----------------------------------------------------------------------===//

// ReduceProd

void ONNXReduceProdOp::inferShapes() {
  if (!getOperand().getType().isa<RankedTensorType>()) {
    emitError("Shape tensor not ranked.");
    return;
  }

  auto operandTy = getOperand().getType().cast<RankedTensorType>();
  getResult().setType(getReductionOutputType(operandTy, axes(), keepdims()));
}

//===----------------------------------------------------------------------===//

// ReduceSum

void ONNXReduceSumOp::inferShapes() {
  if (!getOperand().getType().isa<RankedTensorType>()) {
    emitError("Shape tensor not ranked.");
    return;
  }

  auto operandTy = getOperand().getType().cast<RankedTensorType>();
  getResult().setType(getReductionOutputType(operandTy, axes(), keepdims()));
}

// Conv

// For this operation, we define the attributes once in the original Conv
// operation class. There is no need to redefine the attribute names for the
// other classes based on Conv.
void ONNXConvNoBiasOp::inferShapes() {
  // Generic shape for data input X and weight tensor W:
  // X: (N x C x D1 x D2 ... x Dn)
  // W: (M x C/group x k1 x k2 x ... x kn)

  // Cannot infer shape if no shape exists.
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return;

  auto dataTy = getOperand(0).getType().cast<RankedTensorType>();
  auto weightTy = getOperand(1).getType().cast<RankedTensorType>();
  auto dataShape = dataTy.getShape();
  auto weightShape = weightTy.getShape();

  // Lowest supported convolution is a one dimensional convolution.
  if (dataShape.size() < 3)
    emitError("Data input shape must be at least (NxCxD1).");

  // Check that shape of weight and data have same length.
  if (dataShape.size() != weightShape.size())
    emitError("Weight size not compatible with data size.");

  // Required attribute auto_pad defaults to NOTSET.
  auto autoPad = auto_pad();
  // Group is a required attribute and should have default value of 1.
  int64_t group =
      ONNXConvNoBiasOp::group().getSExtValue(); //.getLimitedValue();
  // Check that the X.shape[1] == (W.shape[1] * group) == C condition holds.
  if (dataShape[1] != (weightShape[1] * group))
    emitError("Channel dimension mismatch.");

  // Note: the value of the group attribut only impacts the way the
  // computation is carried out and not the actual output size.

  // First two output dimensions consist of the number of batches and the
  // number of kernels being applied.
  //
  SmallVector<int64_t, 2> dims;
  // Insert batch size.
  dims.emplace_back(dataShape[0]);
  // Insert number of filters being applied (number of output channels).
  dims.emplace_back(weightShape[0]);

  // Spatial dimensions of the output are computed using the formula:
  //
  // dim = (inputDim - kernelDim + startPadding + endPadding) / stride + 1
  //
  SmallVector<int64_t, 2> outSpatialDims;
  // Number of spatial dimensions.
  int32_t nDims = dataShape.size() - 2;

  // Initialize dimenions based on the input spatial dimensions.
  for (int i = 2; i < dataShape.size(); ++i)
    outSpatialDims.emplace_back(dataShape[i]);

  // Use kernel_shape attribute if present otherwise use size from weight
  // argument.
  SmallVector<int64_t, 2> kernelDims;
  if (auto kernelShape = kernel_shapeAttr()) {
    if (kernelShape.getValue().size() != nDims)
      emitError("kernel_shape length incompatible with spatial dimensions.");
    for (int i = 0; i < nDims; ++i)
      kernelDims.emplace_back(
          (kernelShape.getValue()[i]).cast<IntegerAttr>().getInt());
  } else {
    for (int i = 0; i < nDims; ++i)
      kernelDims.emplace_back(weightShape[i + 2]);
  }

  // Check if dilations attribute is present.
  // If it is then compute new kernel size that includes the receptive field.
  // In this calculation we assume that the receptive field pixels must all be
  // within the bounds of the image. In this case the new kernel size is given
  // by:
  //
  // ( K + 1 ) * d - 1
  // where K is a kernel dimension and d is the dilation along that axis.
  //
  // From a dimensionality perspective the kernel size becomes the dilated
  // kernel size.
  if (auto dilations = dilationsAttr()) {
    if (dilations.getValue().size() != nDims)
      emitError("dilations length incompatible with spatial dimensions.");
    for (int i = 0; i < nDims; ++i)
      kernelDims[i] =
          (kernelDims[i] + 1) *
              (dilations.getValue()[i]).cast<IntegerAttr>().getInt() -
          1;
  }

  // Subtract kernel dimensions from input data dimensions.
  for (int i = 0; i < nDims; ++i)
    outSpatialDims[i] -= kernelDims[i];

  // Add padding information.
  if (autoPad == "NOTSET") {
    // Use pads to to determine the padding. If attribute is not
    // present then pads is considered to be all zeros (no padding).
    if (auto pads = padsAttr()) {
      // pads consists of two entries for each spatial axis.
      if (pads.getValue().size() != 2 * nDims)
        emitError("pads size is not twice the spatial size.");

      for (int i = 0; i < nDims; ++i) {
        // Padding for beginning of axis.
        int32_t p = (pads.getValue()[i]).cast<IntegerAttr>().getInt();
        outSpatialDims[i] += p;
        // Padding for end of axis.
        p = (pads.getValue()[i + nDims]).cast<IntegerAttr>().getInt();
        outSpatialDims[i] += p;
      }
    }
  } else if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") {
    // Pad input so that output size matches input size.
    // Each spatial dimension needs to be padded by a total of:
    //
    // K - 1
    //
    // where K is a kernel spatial dimension.
    // Pad as if stride is 1.
    for (int i = 0; i < nDims; ++i)
      outSpatialDims[i] += kernelDims[i] - 1;
  } else if (autoPad == "VALID") {
    // No padding
  } else {
    emitError("Unexpected attribute value for auto_pad.");
  }

  // Strides
  if (auto strides = ONNXConvNoBiasOp::stridesAttr()) {
    if (strides.getValue().size() != nDims)
      emitError("strides length incompatible with spatial dimensions.");
    for (int i = 0; i < nDims; ++i) {
      int64_t stride = strides.getValue()[i].cast<IntegerAttr>().getInt();
      outSpatialDims[i] = floor(outSpatialDims[i] / stride);
    }
  }

  for (int i = 0; i < nDims; ++i)
    outSpatialDims[i] += 1;

  dims.append(outSpatialDims.begin(), outSpatialDims.end());
  getResult().setType(RankedTensorType::get(dims, dataTy.getElementType()));
}

//===----------------------------------------------------------------------===//

// MaxPoolSingleOut
// Infer shape attributes output:
//   -  auto_pad set to NOTSET;
//   -  dilations, strides: set to 1 if not defined by user;
//   -  pads: set to proper value, 0 if not defined by user.

void ONNXMaxPoolSingleOutOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!X().getType().isa<RankedTensorType>())
    return;
  auto builder = mlir::Builder(this->getContext());

  // 1) Get shape of input.
  auto xTy = X().getType().cast<RankedTensorType>();
  auto xShape = xTy.getShape();
  auto xRank = xShape.size();

  // 2) Analyse parameters. Get kernel sizes from kernel_shape attribute.
  auto kernelShape = kernel_shape();
  if (!kernelShape)
    emitError(
        "kernel_shape is a mandatory attribute for which there is no default");
  auto kernelRank = ArrayAttrSize(kernelShape);
  if (kernelRank > xRank)
    emitError("kernel_shape spatial dimension is too large");
  auto kernelOffset = xRank - kernelRank;

  // Ceil mode.
  auto ceilMode = ceil_mode().getSExtValue();

  // Dilatation.
  auto dilationsOpt = dilations();
  if (dilationsOpt.hasValue()) {
    if (ArrayAttrSize(dilationsOpt) != kernelRank)
      emitError("dialation rank is not the same as the spatial rank");
    // Test values.
    for (int i = 0; i < kernelRank; ++i) {
      if (ArrayAttrIntVal(dilationsOpt, i) < 1)
        emitError("dialation value must be nonzero positive");
    }
  } else {
    // Default dilatation is needed.
    SmallVector<int64_t, 4> defaultVals(kernelRank, 1);
    // Convert to ArrayRef, then build attribute, then store attribute.
    ArrayRef<int64_t> defaultRefs(defaultVals);
    auto defaultAttr = builder.getI64ArrayAttr(defaultRefs);
    dilationsAttr(defaultAttr);
    dilationsOpt = dilations();
  }

  // Storage order.
  auto storageOrder = storage_order().getSExtValue();
  if (storageOrder != 0)
    emitError("column major storage order not supported at this time");

  // Strides.
  auto stridesOpt = strides();
  if (stridesOpt.hasValue()) {
    if (ArrayAttrSize(stridesOpt) != kernelRank)
      emitError("strides rank is not the same as the spatial rank");
    // Check values.
    for (int i = 0; i < kernelRank; ++i) {
      if (ArrayAttrIntVal(stridesOpt, i) < 1)
        emitError("strides value must be nonzero positive");
    }
  } else {
    SmallVector<int64_t, 4> defaultVals(kernelRank, 1);
    // Convert to ArrayRef, then build attribute, then store attribute.
    ArrayRef<int64_t> defaultRefs(defaultVals);
    auto defaultAttr = builder.getI64ArrayAttr(defaultRefs);
    stridesAttr(defaultAttr);
    stridesOpt = strides();
  }

  // Now try to find padding, getting auto_pad attribute first.
  auto autoPad = auto_pad();
  // And then investigate the various different cases.
  SmallVector<int64_t, 4> actualPads(2 * kernelRank, 0);
  if (autoPad == "NOTSET") {
    auto padsOpt = pads();
    if (padsOpt.hasValue()) {
      // Pads consists of two entries for each spatial axis.
      if (ArrayAttrSize(padsOpt) != 2 * kernelRank)
        emitError("pads rank is not twice the spatial rank");
      // Check values
      for (int i = 0; i < 2 * kernelRank; ++i) {
        int64_t p = ArrayAttrIntVal(padsOpt, i);
        if (p < 0)
          emitError("pads value must be nonnegative");
        actualPads[i] = p;
      }
    }
  } else if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") {
    for (int i = 0; i < kernelRank; ++i) {
      auto inputSpatialShape = xShape[kernelOffset + i];
      auto kernelSpatialShape = ArrayAttrIntVal(kernelShape, i);
      auto dilations = ArrayAttrIntVal(dilationsOpt, i);
      auto strideSpatialShape = ArrayAttrIntVal(stridesOpt, i);
      int64_t outputSpatialShape =
          ceil((1.0 * inputSpatialShape) / (1.0 * strideSpatialShape));
      auto sumOfPad = (outputSpatialShape - 1) * strideSpatialShape +
                      ((kernelSpatialShape - 1) * dilations + 1) -
                      inputSpatialShape;
      actualPads[i] = actualPads[kernelRank + i] = sumOfPad / 2;
      if (sumOfPad % 2 != 0) {
        if (autoPad == "SAME_UPPER") {
          actualPads[kernelRank + i] += 1;
        } else {
          actualPads[i] += 1;
        }
      }
    }
  } else if (autoPad != "VALID") {
    emitError("auto_pad of unknown / unsupported value.");
  }
  // Set pads values in attributes.
  {
    ArrayRef<int64_t> defaultRefs(actualPads);
    auto defaultAttr = builder.getI64ArrayAttr(defaultRefs);
    padsAttr(defaultAttr);
    auto defaultAutoPadAttr = builder.getStringAttr("NOTSET");
    auto_padAttr(defaultAutoPadAttr);
  }

  // Initialize output shape.
  SmallVector<int64_t, 4> yShape(xShape.begin(), xShape.end());
  // Process for all kernel dimensions.
  for (int i = 0; i < kernelRank; ++i) {
    auto inputSpatialShape = xShape[kernelOffset + i];
    auto padShape = actualPads[i] + actualPads[kernelRank + i];
    auto kernelSpatialShape = ArrayAttrIntVal(kernelShape, i);
    auto dilations = ArrayAttrIntVal(dilationsOpt, i);
    auto strideSpatialShape = ArrayAttrIntVal(stridesOpt, i);
    double numerator = inputSpatialShape + padShape -
                       ((kernelSpatialShape - 1) * dilations + 1);
    double denominator = strideSpatialShape;
    int64_t res;
    if (ceilMode) {
      res = ceil(numerator / denominator) + 1;
    } else {
      res = floor(numerator / denominator) + 1;
    }
    yShape[kernelOffset + i] = res;
  }
  auto arrayTy = getOperand().getType().cast<RankedTensorType>();
  getResult().setType(RankedTensorType::get(yShape, arrayTy.getElementType()));
}

//===----------------------------------------------------------------------===//

// Unsqueeze

void ONNXUnsqueezeOp::inferShapes() {
  if (!getOperand().getType().isa<RankedTensorType>())
    return;

  auto operandTy = getOperand().getType().cast<RankedTensorType>();
  int inRank = operandTy.getRank();

  ArrayAttr axisAttrs = axesAttr();
  SmallVector<int, 4> axes;
  int outRank = 0;
  if (axisAttrs) {
    outRank = inRank + axisAttrs.getValue().size();
    for (auto axisAttr : axisAttrs.getValue()) {
      int axis = axisAttr.cast<IntegerAttr>().getInt();
      axis = axis >= 0 ? axis : (outRank + axis);
      // Valid range
      assert(axis >= -outRank && axis <= outRank - 1);
      if (std::find(axes.begin(), axes.end(), axis) == axes.end())
        axes.emplace_back(axis);
      else
        emitError("Duplicated axes.");
    }
  } else {
    emitError("Axes attribute is required.");
  }

  SmallVector<int64_t, 4> dims;
  for (int i = 0, j = 0; i < outRank || j < inRank; ++i) {
    if (std::find(axes.begin(), axes.end(), i) != axes.end()) {
      dims.emplace_back(1);
    } else {
      dims.emplace_back(operandTy.getShape()[j++]);
    }
  }
  getResult().setType(RankedTensorType::get(dims, operandTy.getElementType()));
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/onnx.cpp.inc"
