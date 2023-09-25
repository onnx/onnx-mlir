/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- PerfModel.cpp - Estimate if CPU or NNPA is faster  ----------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains methods that estimates the computational time of an ONNX
// operation on a CPU and NNPA for a z16 and indicates which device is faster.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/PerfModel.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

#include <cmath>
#include <functional>

#define DEBUG_TYPE "zhigh-perf-model"

using namespace mlir;
using namespace onnx_mlir;

// hi alex namespace {
//===----------------------------------------------------------------------===//
// Auto-generated model.
using PERF_MODEL3 = std::function<bool(double, double, double)>;
using PERF_MODEL4 = std::function<bool(double, double, double, double)>;

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/PerfModel.inc"

//===----------------------------------------------------------------------===//
// Support functions for reporting.

// Return true with a debug message reporting reason for success on NNPA.
inline bool fasterOnNNPA(Operation *op, std::string msg) {
  LLVM_DEBUG({
    llvm::dbgs() << "Faster on NNPA: " << msg << " for op: ";
    op->dump();
  });
  return true;
}

// Return false with a debug message reporting reason for failure on NNPA.
inline bool fasterOnCPU(Operation *op, std::string msg) {
  LLVM_DEBUG({
    llvm::dbgs() << "Faster on CPU: " << msg << " for op: ";
    op->dump();
  });
  return false;
}

//===----------------------------------------------------------------------===//
// Support for unary/binary elementwise with possibly unknown dimensions.

bool isElementwiseOpFasterOnNNPA(Operation *op, Value oper,
    const DimAnalysis *dimAnalysis, PERF_MODEL3 modelForCPU,
    PERF_MODEL3 modelForNNPA) {

  // At this time, use only 1 of the two operands.
  ShapedType operType = oper.getType().dyn_cast_or_null<ShapedType>();
  assert(operType && operType.hasRank() && "expected shaped type with rank");
  int64_t operRank = operType.getRank();
  assert(operRank <= 4 && "expected rank <= 4");
  llvm::ArrayRef<int64_t> shape = operType.getShape();
  int64_t e4 = operRank >= 4 ? shape[operRank - 4] : 1;
  int64_t e3 = operRank >= 3 ? shape[operRank - 3] : 1;
  int64_t e2 = operRank >= 2 ? shape[operRank - 2] : 1;
  int64_t e1 = operRank >= 1 ? shape[operRank - 1] : 1;
  // Handle dynamic shapes, eventually it would be good to have ranges given by
  // the user.
  std::string msg = "";
  if (e4 == ShapedType::kDynamic) {
    e4 = 1; // Assume small.
    LLVM_DEBUG(msg += " E4=1: empty because dyn.");
  }
  if (e3 == ShapedType::kDynamic) {
    e3 = 1; // Assume small.
    LLVM_DEBUG(msg += " E3=1: empty because dyn.");
  }
  if (e2 == ShapedType::kDynamic) {
    e2 = 32; // Assume full.
    LLVM_DEBUG(msg += " E2=32: full because dyn.");
  }
  if (e1 == ShapedType::kDynamic) {
    e1 = 64; // Assume full.
    LLVM_DEBUG(msg += " E1=64: full because dyn.");
  }
  if (modelForCPU(e4 * e3, e2, e1) < modelForNNPA(e4 * e3, e2, e1))
    return fasterOnNNPA(op, "Model estimates faster time on NNPA." + msg);
  return fasterOnCPU(op, "Model estimates faster time on CPU" + msg);
}

//===----------------------------------------------------------------------===//
// Support for matmul with possibly unknown dimensions.

bool isMatMulOpFasterOnNNPA(Operation *op, Value a, Value b, bool aTransposed,
    bool bTransposed, const DimAnalysis *dimAnalysis) {
  // Scanning A.
  ShapedType aType = a.getType().dyn_cast_or_null<ShapedType>();
  assert(aType && aType.hasRank() && "expected shaped type with A rank");
  int64_t aRank = aType.getRank();
  assert(aRank >= 2 && aRank <= 3 && "expected A rank 2..3");
  llvm::ArrayRef<int64_t> aShape = aType.getShape();
  int64_t aB = aRank >= 3 ? aShape[aRank - 3] : 1;
  int64_t aNIndex = aTransposed ? aRank - 1 : aRank - 2;
  int64_t aMIndex = aTransposed ? aRank - 2 : aRank - 1;
  int64_t aN = aShape[aNIndex];
  int64_t aM = aShape[aMIndex];
  // Scanning B.
  ShapedType bType = b.getType().dyn_cast_or_null<ShapedType>();
  assert(bType && bType.hasRank() && "expected shaped type with B rank");
  int64_t bRank = bType.getRank();
  assert(bRank >= 2 && bRank <= 3 && "expected B rank 2..3");
  llvm::ArrayRef<int64_t> bShape = bType.getShape();
  int64_t bB = bRank >= 3 ? bShape[bRank - 3] : 1;
  int64_t bMIndex = bTransposed ? bRank - 1 : bRank - 2;
  int64_t bKIndex = bTransposed ? bRank - 2 : bRank - 1;
  int64_t bM = bShape[bMIndex];
  int64_t bK = bShape[bKIndex];
  assert(aM == bM && "expected M dims to be identical");
  // Rules common to matmul with/without broadcast.
  // Ideally we would have ranges to estimate cost when dynamic.
  std::string msg = "";
  if (aN == ShapedType::kDynamic) {
    // Assume the N dim of the matmul will be small.
    aN = 1;
    LLVM_DEBUG(msg += " N=1: empty because dyn.");
  }
  if (aM == ShapedType::kDynamic) {
    // Assume the dynamic lower dim of the matmul will be large enough.
    aM = 64;
    LLVM_DEBUG(msg += " M=32: full because dyn.");
  }
  if (bK == ShapedType::kDynamic) {
    // Assume the dynamic lower dim of the matmul will be large enough.
    bK = 64;
    LLVM_DEBUG(msg += " K=32: full because dyn.");
  }
  // Determine if we have a broadcast (will change cost calculations).
  bool hasBroadcast = true;
  if (aRank == 2 && bRank == 2) // No broadcast dim.
    hasBroadcast = false;
  else if (aB == 1 && bB == 1) // No broadcast because both 1.
    hasBroadcast = false;
  else if (aRank == 3 && bRank == 3 &&
           dimAnalysis->sameDim(a, aRank - 3, b, bRank - 3))
    hasBroadcast = false;
  // Assume the B dim of the matmul will be small.
  if (aB == ShapedType::kDynamic) {
    aB = 1;
    LLVM_DEBUG(msg += " aB=1: assume no broadcast because dyn.");
  }
  if (bB == ShapedType::kDynamic) {
    bB = 1;
    LLVM_DEBUG(msg += " bB=1: assume no broadcast because dyn.");
  }
  // Handle case without broadcast. Right now, broadcast cases use the same
  // method.
  if (!hasBroadcast ||
      hasBroadcast /* no perf measurement yet for broadcast case*/) {
    if (estimatedTimeForCPU_MatMul_3ds(aB, aN, aM, bK) <
        estimatedTimeForNNPA_MatMul_3ds(aB, aN, aM, bK))
      return fasterOnNNPA(op, "Model estimates faster time on NNPA." + msg);
    return fasterOnCPU(op, "Model estimates faster time on CPU" + msg);
  }
  llvm_unreachable("should not get here");
}

//===----------------------------------------------------------------------===//
// Processing for each op: binary elementwise.

template <typename OP_TYPE>
bool checkIfOpFasterOnNNPA(OP_TYPE op, const DimAnalysis *dimAnalysis) {
  llvm_unreachable("should have a model for all defined ops");
}

template <>
bool checkIfOpFasterOnNNPA<ONNXAddOp>(
    ONNXAddOp op, const DimAnalysis *dimAnalysis) {
  fprintf(stderr, "    check if add is faster\n");
  return isElementwiseOpFasterOnNNPA(op.getOperation(), op.getOperand(0),
      dimAnalysis, estimatedTimeForCPU_Add_3ds, estimatedTimeForNNPA_Add_3ds);
}

template <>
bool checkIfOpFasterOnNNPA<ONNXDivOp>(
    ONNXDivOp op, const DimAnalysis *dimAnalysis) {
  return isElementwiseOpFasterOnNNPA(op.getOperation(), op.getOperand(0),
      dimAnalysis, estimatedTimeForCPU_Div_3ds, estimatedTimeForNNPA_Div_3ds);
}

template <>
bool checkIfOpFasterOnNNPA<ONNXMaxOp>(
    ONNXMaxOp op, const DimAnalysis *dimAnalysis) {
  return isElementwiseOpFasterOnNNPA(op.getOperation(), op.getOperand(0),
      dimAnalysis, estimatedTimeForCPU_Max_3ds, estimatedTimeForNNPA_Max_3ds);
}

template <>
bool checkIfOpFasterOnNNPA<ONNXMinOp>(
    ONNXMinOp op, const DimAnalysis *dimAnalysis) {
  return isElementwiseOpFasterOnNNPA(op.getOperation(), op.getOperand(0),
      dimAnalysis, estimatedTimeForCPU_Min_3ds, estimatedTimeForNNPA_Min_3ds);
}

template <>
bool checkIfOpFasterOnNNPA<ONNXMulOp>(
    ONNXMulOp op, const DimAnalysis *dimAnalysis) {
  return isElementwiseOpFasterOnNNPA(op.getOperation(), op.getOperand(0),
      dimAnalysis, estimatedTimeForCPU_Mul_3ds, estimatedTimeForNNPA_Mul_3ds);
}

template <>
bool checkIfOpFasterOnNNPA<ONNXPowOp>(
    ONNXPowOp op, const DimAnalysis *dimAnalysis) {
  int64_t exponentValue;
  if (hasIntegerPowerExponent(&op, exponentValue)) {
    if (exponentValue == 2)
      return isElementwiseOpFasterOnNNPA(op.getOperation(), op.getOperand(0),
          dimAnalysis, estimatedTimeForCPU_Pow_2_3ds,
          estimatedTimeForNNPA_Pow_2_3ds);
    if (exponentValue == 3)
      return isElementwiseOpFasterOnNNPA(op.getOperation(), op.getOperand(0),
          dimAnalysis, estimatedTimeForCPU_Pow_3_3ds,
          estimatedTimeForNNPA_Pow_3_3ds);
    if (exponentValue == 4)
      return isElementwiseOpFasterOnNNPA(op.getOperation(), op.getOperand(0),
          dimAnalysis, estimatedTimeForCPU_Pow_4_3ds,
          estimatedTimeForNNPA_Pow_4_3ds);
  }
  // For other power exponent, just use pow of 8.
  return isElementwiseOpFasterOnNNPA(op.getOperation(), op.getOperand(0),
      dimAnalysis, estimatedTimeForCPU_Pow_8_3ds,
      estimatedTimeForNNPA_Pow_8_3ds);
}

template <>
bool checkIfOpFasterOnNNPA<ONNXSubOp>(
    ONNXSubOp op, const DimAnalysis *dimAnalysis) {
  return isElementwiseOpFasterOnNNPA(op.getOperation(), op.getOperand(0),
      dimAnalysis, estimatedTimeForCPU_Sub_3ds, estimatedTimeForNNPA_Sub_3ds);
}

//===----------------------------------------------------------------------===//
// Processing for each op: unary elementwise.

template <>
bool checkIfOpFasterOnNNPA<ONNXExpOp>(
    ONNXExpOp op, const DimAnalysis *dimAnalysis) {
  return isElementwiseOpFasterOnNNPA(op.getOperation(), op.getOperand(),
      dimAnalysis, estimatedTimeForCPU_Exp_3ds, estimatedTimeForNNPA_Exp_3ds);
}

template <>
bool checkIfOpFasterOnNNPA<ONNXLogOp>(
    ONNXLogOp op, const DimAnalysis *dimAnalysis) {
  return isElementwiseOpFasterOnNNPA(op.getOperation(), op.getOperand(),
      dimAnalysis, estimatedTimeForCPU_Log_3ds, estimatedTimeForNNPA_Log_3ds);
}

template <>
bool checkIfOpFasterOnNNPA<ONNXReluOp>(
    ONNXReluOp op, const DimAnalysis *dimAnalysis) {
  return isElementwiseOpFasterOnNNPA(op.getOperation(), op.getOperand(),
      dimAnalysis, estimatedTimeForCPU_Relu_3ds, estimatedTimeForNNPA_Relu_3ds);
}

template <>
bool checkIfOpFasterOnNNPA<ONNXSigmoidOp>(
    ONNXSigmoidOp op, const DimAnalysis *dimAnalysis) {
  return isElementwiseOpFasterOnNNPA(op.getOperation(), op.getOperand(),
      dimAnalysis, estimatedTimeForCPU_Sigmoid_3ds,
      estimatedTimeForNNPA_Sigmoid_3ds);
}

template <>
bool checkIfOpFasterOnNNPA<ONNXSoftmaxOp>(
    ONNXSoftmaxOp op, const DimAnalysis *dimAnalysis) {
  return isElementwiseOpFasterOnNNPA(op.getOperation(), op.getOperand(),
      dimAnalysis, estimatedTimeForCPU_Softmax_3ds,
      estimatedTimeForNNPA_Softmax_3ds);
}

template <>
bool checkIfOpFasterOnNNPA<ONNXTanhOp>(
    ONNXTanhOp op, const DimAnalysis *dimAnalysis) {
  return isElementwiseOpFasterOnNNPA(op.getOperation(), op.getOperand(),
      dimAnalysis, estimatedTimeForCPU_Tanh_3ds, estimatedTimeForNNPA_Tanh_3ds);
}

//===----------------------------------------------------------------------===//
// Processing for each op: MatMul.

template <>
bool checkIfOpFasterOnNNPA<ONNXMatMulOp>(
    ONNXMatMulOp op, const DimAnalysis *dimAnalysis) {
  return isMatMulOpFasterOnNNPA(op.getOperation(), op.getOperand(0),
      op.getOperand(1), false /*a transposed*/, false /*b transposed*/,
      dimAnalysis);
}

// hi alex } // namespace

//===----------------------------------------------------------------------===//
// Function to perform evaluation.

namespace onnx_mlir {

bool isOpFasterOnNNPA(mlir::Operation *op, const DimAnalysis *dimAnalysis) {
  // Binary elementwise NNPA candidate ops.
  fprintf(stderr, "  start analyzing this op\n    ");
  op->dump();
  if (auto addOp = dyn_cast<ONNXAddOp>(op))
    return checkIfOpFasterOnNNPA(addOp, dimAnalysis);
  if (auto divOp = dyn_cast<ONNXDivOp>(op))
    return checkIfOpFasterOnNNPA(divOp, dimAnalysis);
  if (auto maxOp = dyn_cast<ONNXMaxOp>(op))
    return checkIfOpFasterOnNNPA(maxOp, dimAnalysis);
  if (auto minOp = dyn_cast<ONNXMinOp>(op))
    return checkIfOpFasterOnNNPA(minOp, dimAnalysis);
  if (auto mulOp = dyn_cast<ONNXMulOp>(op))
    return checkIfOpFasterOnNNPA(mulOp, dimAnalysis);
  if (auto powOp = dyn_cast<ONNXPowOp>(op))
    return checkIfOpFasterOnNNPA(powOp, dimAnalysis);
  if (auto subOp = dyn_cast<ONNXSubOp>(op))
    return checkIfOpFasterOnNNPA(subOp, dimAnalysis);
  // Unary elementwise NNPA candidate ops.
  if (auto expOp = dyn_cast<ONNXExpOp>(op))
    return checkIfOpFasterOnNNPA(expOp, dimAnalysis);
  if (auto logOp = dyn_cast<ONNXLogOp>(op))
    return checkIfOpFasterOnNNPA(logOp, dimAnalysis);
  if (auto reluOp = dyn_cast<ONNXReluOp>(op))
    return checkIfOpFasterOnNNPA(reluOp, dimAnalysis);
  if (auto sigmoidOp = dyn_cast<ONNXSigmoidOp>(op))
    return checkIfOpFasterOnNNPA(sigmoidOp, dimAnalysis);
  if (auto softmaxOp = dyn_cast<ONNXSoftmaxOp>(op))
    return checkIfOpFasterOnNNPA(softmaxOp, dimAnalysis);
  if (auto tanhOp = dyn_cast<ONNXTanhOp>(op))
    return checkIfOpFasterOnNNPA(tanhOp, dimAnalysis);
  // Matmul.
  if (auto matMulOp = dyn_cast<ONNXMatMulOp>(op))
    return checkIfOpFasterOnNNPA(matMulOp, dimAnalysis);

  if (auto addOp = dyn_cast<ONNXAddOp>(op))
    return checkIfOpFasterOnNNPA(addOp, dimAnalysis);
  if (auto addOp = dyn_cast<ONNXAddOp>(op))
    return checkIfOpFasterOnNNPA(addOp, dimAnalysis);

  // Unknown, issue a warning and assume its faster on NNPA
  llvm::errs()
      << "The operation below is a candidate for NNPA execution but has no "
         "cost benefit analysis. Please add analysis.\n";
  return true;
}

} // namespace onnx_mlir
