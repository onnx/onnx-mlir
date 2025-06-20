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

namespace {
//===----------------------------------------------------------------------===//
// Auto-generated model.
using PERF_MODEL3 = std::function<double(double, double, double)>;
using PERF_MODEL4 = std::function<double(double, double, double, double)>;

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/PerfModel.inc"

//===----------------------------------------------------------------------===//
// Support functions

// Summarize higher dims (leaving ub..rank-1 untouched). If none, return size
// of 1. Otherwise, returns the cumulative multiplication of each of the static
// sizes. HasDyn is set to true if one or more dynamic sizes are found.

inline int64_t summarizeHigherDims(
    llvm::ArrayRef<int64_t> shape, int64_t ub, bool &hasDyn) {
  int64_t cumulativeConstVal = 1;
  hasDyn = false;
  for (int64_t i = 0; i < ub; ++i) {
    if (shape[i] == ShapedType::kDynamic)
      hasDyn = true;
    else
      cumulativeConstVal *= shape[i];
  }
  return cumulativeConstVal;
}

//===----------------------------------------------------------------------===//
// Support for unary/binary elementwise with possibly unknown dimensions.

void processDim(Value oper, int64_t &e4, int64_t &e3, int64_t &e2, int64_t &e1,
    std::string &msg) {
  // At this time, use only 1 of the two operands.
  ShapedType operType = mlir::dyn_cast_or_null<ShapedType>(oper.getType());
  assert(operType && operType.hasRank() && "expected shaped type with rank");
  int64_t operRank = operType.getRank();
  assert(operRank <= 4 && "expected rank <= 4");
  llvm::ArrayRef<int64_t> shape = operType.getShape();
  // Gather all 4th...nth ranked shape together. If all dynamic; assume size
  // of 1.
  LLVM_DEBUG(msg = "");
  bool hasDynamicE4;
  e4 = summarizeHigherDims(shape, operRank - 3, hasDynamicE4);
  e3 = operRank >= 3 ? shape[operRank - 3] : 1;
  e2 = operRank >= 2 ? shape[operRank - 2] : 1;
  e1 = operRank >= 1 ? shape[operRank - 1] : 1;
  // Handle dynamic shapes, eventually it would be good to have ranges given by
  // the user.
  if (hasDynamicE4) {
    LLVM_DEBUG(msg += " E4+: assume size 1 for dynamic dims.");
  }
  if (e3 == ShapedType::kDynamic) {
    e3 = 1; // Assume small.
    LLVM_DEBUG(msg += " E3=1: dyn, assume size 1.");
  }
  if (e2 == ShapedType::kDynamic) {
    e2 = 32; // Assume full.
    LLVM_DEBUG(msg += " E2=32: dyn, assume full tile.");
  }
  if (e1 == ShapedType::kDynamic) {
    e1 = 64; // Assume full.
    LLVM_DEBUG(msg += " E1=64: dyn, assume full tile.");
  }
}

void estimateTimeForElementwiseOp(Operation *op, Value oper,
    const DimAnalysis *dimAnalysis, PERF_MODEL3 modelForCPU,
    PERF_MODEL3 modelForNNPA, double &cpuEstimatedTime,
    double &nnpaEstimatedTime) {

  // Process dim (collapse and handle dynamic sizes).
  int64_t e4, e3, e2, e1;
  std::string msg;
  processDim(oper, e4, e3, e2, e1, msg);

  cpuEstimatedTime = modelForCPU(e4 * e3, e2, e1);
  nnpaEstimatedTime = modelForNNPA(e4 * e3, e2, e1);
  LLVM_DEBUG(llvm::dbgs() << "  Estimated times for op " << op->getName()
                          << ": nnpa " << nnpaEstimatedTime << ", cpu "
                          << cpuEstimatedTime << "." << msg.c_str() << "\n");
}

//===----------------------------------------------------------------------===//
// Support for matmul with possibly unknown dimensions.

void estimateTimeForMatMulOp(Operation *op, Value a, Value b, bool aTransposed,
    bool bTransposed, const DimAnalysis *dimAnalysis, double &cpuEstimatedTime,
    double &nnpaEstimatedTime) {
  // Scanning A.
  ShapedType aType = mlir::dyn_cast_or_null<ShapedType>(a.getType());
  assert(aType && aType.hasRank() && "expected shaped type with A rank");
  int64_t aRank = aType.getRank();
  llvm::ArrayRef<int64_t> aShape = aType.getShape();
  bool aBDynamic;
  int64_t aB = summarizeHigherDims(aShape, aRank - 2, aBDynamic);
  int64_t aNIndex = aTransposed ? aRank - 1 : aRank - 2;
  int64_t aMIndex = aTransposed ? aRank - 2 : aRank - 1;
  int64_t aN = aShape[aNIndex];
  int64_t aM = aShape[aMIndex];
  // Scanning B.
  ShapedType bType = mlir::dyn_cast_or_null<ShapedType>(b.getType());
  assert(bType && bType.hasRank() && "expected shaped type with B rank");
  int64_t bRank = bType.getRank();
  llvm::ArrayRef<int64_t> bShape = bType.getShape();
  bool bBDynamic;
  int64_t bB = summarizeHigherDims(bShape, bRank - 2, bBDynamic);
  int64_t bMIndex = bTransposed ? bRank - 1 : bRank - 2;
  int64_t bKIndex = bTransposed ? bRank - 2 : bRank - 1;
  int64_t bM = bShape[bMIndex];
  int64_t bK = bShape[bKIndex];
  // Verify dimensions and now use common names from there below.
  assert(aM == bM && "expected M dims to be identical");
  int64_t N = aN, M = aM, K = bK;
  // Rules common to matmul with/without broadcast.
  // Ideally we would have ranges to estimate cost when dynamic.
  std::string msg;
  LLVM_DEBUG(msg = "");
  // Assume the broadcast B dim of the matmul will be small.
  if (aBDynamic) {
    LLVM_DEBUG(msg += " B+ for input A: assume size 1 for dynamic dims.");
  }
  if (bBDynamic) {
    LLVM_DEBUG(msg += " B+ for input B: assume size 1 for dynamic dims.");
  }
  if (N == ShapedType::kDynamic) {
    // Assume the N dim of the matmul will be small.
    N = 1;
    LLVM_DEBUG(msg += " N=1: empty because dyn.");
  }
  if (M == ShapedType::kDynamic) {
    // Assume the dynamic lower dim of the matmul will be large enough.
    M = 64;
    LLVM_DEBUG(msg += " M=32: full because dyn.");
  }
  if (K == ShapedType::kDynamic) {
    // Assume the dynamic lower dim of the matmul will be large enough.
    K = 64;
    LLVM_DEBUG(msg += " K=32: full because dyn.");
  }
  // Determine if we have a broadcast (will change cost calculations).
  bool hasBroadcast;
  if (aB == 1 && !aBDynamic && bB == 1 && !bBDynamic)
    // No broadcast because both 1, which may be an artefact that one or both
    // are 2D (as 2D default to size 1 with no dynamic).
    hasBroadcast = false;
  else {
    // Assume no broadcast until shown otherwise.
    hasBroadcast = false;
    int64_t maxRank = std::max(aRank, bRank);
    for (int64_t i = 2; i < maxRank; ++i) {
      if (i < aRank && i < bRank) {
        // Both have a shape, make sure that they are compatible.
        if (!dimAnalysis->sameDim(a, aRank - i - 1, b, bRank - i - 1)) {
          hasBroadcast = true;
          break;
        }
      } else if (i < aRank && aShape[aRank - i - 1] != 1) {
        // Only A had a shape; when it is not 1, we have broadcast.
        hasBroadcast = true;
        break;
      } else if (i < bRank && bShape[bRank - i - 1] != 1) {
        // Only B had a shape; when it is not 1, we have broadcast.
        hasBroadcast = true;
        break;
      } else {
        llvm_unreachable("should not get here");
      }
    }
  }
  LLVM_DEBUG({
    if (hasBroadcast)
      msg += " Has broadcast.";
  });

  // Handle case without broadcast (aka !hasBroadcast). Right now, broadcast
  // cases (aka hasBroadcast) use the same method. So invoke in all cases.
  if (/*!hasBroadcast || hasBroadcast */ true) {
    // For no broadcast, pick the largest B dimension.
    int64_t B = std::max(aB, bB);
    nnpaEstimatedTime = estimatedTimeForNNPA_MatMul_3ds(B, N, M, K);
    cpuEstimatedTime = estimatedTimeForCPU_MatMul_3ds(B, N, M, K);
    LLVM_DEBUG(llvm::dbgs()
               << "  Estimated times for op " << op->getName() << " with dim ("
               << B << ", " << N << ", " << M << ", " << K << "): nnpa "
               << nnpaEstimatedTime << ", cpu " << cpuEstimatedTime << "."
               << msg.c_str() << "\n");

    return;
  }
  llvm_unreachable("should not get here");
}

//===----------------------------------------------------------------------===//
// Processing for each op: binary elementwise.

template <typename OP_TYPE>
void estimateTimeForOp(OP_TYPE op, const DimAnalysis *dimAnalysis,
    double &cpuEstimatedTime, double &nnpaEstimatedTime) {
  llvm_unreachable("should have a model for all defined ops");
}

template <>
void estimateTimeForOp<ONNXAddOp>(ONNXAddOp op, const DimAnalysis *dimAnalysis,
    double &cpuEstimatedTime, double &nnpaEstimatedTime) {
  estimateTimeForElementwiseOp(op.getOperation(), op.getOperand(0), dimAnalysis,
      estimatedTimeForCPU_Add_3ds, estimatedTimeForNNPA_Add_3ds,
      cpuEstimatedTime, nnpaEstimatedTime);
}

template <>
void estimateTimeForOp<ONNXDivOp>(ONNXDivOp op, const DimAnalysis *dimAnalysis,
    double &cpuEstimatedTime, double &nnpaEstimatedTime) {
  estimateTimeForElementwiseOp(op.getOperation(), op.getOperand(0), dimAnalysis,
      estimatedTimeForCPU_Div_3ds, estimatedTimeForNNPA_Div_3ds,
      cpuEstimatedTime, nnpaEstimatedTime);
}

template <>
void estimateTimeForOp<ONNXMaxOp>(ONNXMaxOp op, const DimAnalysis *dimAnalysis,
    double &cpuEstimatedTime, double &nnpaEstimatedTime) {
  estimateTimeForElementwiseOp(op.getOperation(), op.getOperand(0), dimAnalysis,
      estimatedTimeForCPU_Max_3ds, estimatedTimeForNNPA_Max_3ds,
      cpuEstimatedTime, nnpaEstimatedTime);
}

template <>
void estimateTimeForOp<ONNXMinOp>(ONNXMinOp op, const DimAnalysis *dimAnalysis,
    double &cpuEstimatedTime, double &nnpaEstimatedTime) {
  estimateTimeForElementwiseOp(op.getOperation(), op.getOperand(0), dimAnalysis,
      estimatedTimeForCPU_Min_3ds, estimatedTimeForNNPA_Min_3ds,
      cpuEstimatedTime, nnpaEstimatedTime);
}

template <>
void estimateTimeForOp<ONNXMulOp>(ONNXMulOp op, const DimAnalysis *dimAnalysis,
    double &cpuEstimatedTime, double &nnpaEstimatedTime) {
  estimateTimeForElementwiseOp(op.getOperation(), op.getOperand(0), dimAnalysis,
      estimatedTimeForCPU_Mul_3ds, estimatedTimeForNNPA_Mul_3ds,
      cpuEstimatedTime, nnpaEstimatedTime);
}

template <>
void estimateTimeForOp<ONNXPowOp>(ONNXPowOp op, const DimAnalysis *dimAnalysis,
    double &cpuEstimatedTime, double &nnpaEstimatedTime) {
  int64_t exponentValue;
  if (hasIntegerPowerExponent(&op, exponentValue)) {
    if (exponentValue == 2)
      estimateTimeForElementwiseOp(op.getOperation(), op.getOperand(0),
          dimAnalysis, estimatedTimeForCPU_Pow2_3ds,
          estimatedTimeForNNPA_Pow2_3ds, cpuEstimatedTime, nnpaEstimatedTime);
    if (exponentValue == 3)
      estimateTimeForElementwiseOp(op.getOperation(), op.getOperand(0),
          dimAnalysis, estimatedTimeForCPU_Pow3_3ds,
          estimatedTimeForNNPA_Pow3_3ds, cpuEstimatedTime, nnpaEstimatedTime);
    if (exponentValue == 4)
      estimateTimeForElementwiseOp(op.getOperation(), op.getOperand(0),
          dimAnalysis, estimatedTimeForCPU_Pow4_3ds,
          estimatedTimeForNNPA_Pow4_3ds, cpuEstimatedTime, nnpaEstimatedTime);
  }
  // For other power exponent, just use pow of 8.
  estimateTimeForElementwiseOp(op.getOperation(), op.getOperand(0), dimAnalysis,
      estimatedTimeForCPU_Pow8_3ds, estimatedTimeForNNPA_Pow8_3ds,
      cpuEstimatedTime, nnpaEstimatedTime);
}

template <>
void estimateTimeForOp<ONNXSubOp>(ONNXSubOp op, const DimAnalysis *dimAnalysis,
    double &cpuEstimatedTime, double &nnpaEstimatedTime) {
  estimateTimeForElementwiseOp(op.getOperation(), op.getOperand(0), dimAnalysis,
      estimatedTimeForCPU_Sub_3ds, estimatedTimeForNNPA_Sub_3ds,
      cpuEstimatedTime, nnpaEstimatedTime);
}

//===----------------------------------------------------------------------===//
// Processing for each op: unary elementwise.

template <>
void estimateTimeForOp<ONNXExpOp>(ONNXExpOp op, const DimAnalysis *dimAnalysis,
    double &cpuEstimatedTime, double &nnpaEstimatedTime) {
  estimateTimeForElementwiseOp(op.getOperation(), op.getOperand(), dimAnalysis,
      estimatedTimeForCPU_Exp_3ds, estimatedTimeForNNPA_Exp_3ds,
      cpuEstimatedTime, nnpaEstimatedTime);
}

template <>
void estimateTimeForOp<ONNXLogOp>(ONNXLogOp op, const DimAnalysis *dimAnalysis,
    double &cpuEstimatedTime, double &nnpaEstimatedTime) {
  estimateTimeForElementwiseOp(op.getOperation(), op.getOperand(), dimAnalysis,
      estimatedTimeForCPU_Log_3ds, estimatedTimeForNNPA_Log_3ds,
      cpuEstimatedTime, nnpaEstimatedTime);
}

template <>
void estimateTimeForOp<ONNXReluOp>(ONNXReluOp op,
    const DimAnalysis *dimAnalysis, double &cpuEstimatedTime,
    double &nnpaEstimatedTime) {
  estimateTimeForElementwiseOp(op.getOperation(), op.getOperand(), dimAnalysis,
      estimatedTimeForCPU_Relu_3ds, estimatedTimeForNNPA_Relu_3ds,
      cpuEstimatedTime, nnpaEstimatedTime);
}

template <>
void estimateTimeForOp<ONNXSigmoidOp>(ONNXSigmoidOp op,
    const DimAnalysis *dimAnalysis, double &cpuEstimatedTime,
    double &nnpaEstimatedTime) {
  estimateTimeForElementwiseOp(op.getOperation(), op.getOperand(), dimAnalysis,
      estimatedTimeForCPU_Sigmoid_3ds, estimatedTimeForNNPA_Sigmoid_3ds,
      cpuEstimatedTime, nnpaEstimatedTime);
}

template <>
void estimateTimeForOp<ONNXSoftmaxOp>(ONNXSoftmaxOp op,
    const DimAnalysis *dimAnalysis, double &cpuEstimatedTime,
    double &nnpaEstimatedTime) {
  estimateTimeForElementwiseOp(op.getOperation(), op.getOperand(), dimAnalysis,
      estimatedTimeForCPU_Softmax_3ds, estimatedTimeForNNPA_Softmax_3ds,
      cpuEstimatedTime, nnpaEstimatedTime);
}

template <>
void estimateTimeForOp<ONNXTanhOp>(ONNXTanhOp op,
    const DimAnalysis *dimAnalysis, double &cpuEstimatedTime,
    double &nnpaEstimatedTime) {
  estimateTimeForElementwiseOp(op.getOperation(), op.getOperand(), dimAnalysis,
      estimatedTimeForCPU_Tanh_3ds, estimatedTimeForNNPA_Tanh_3ds,
      cpuEstimatedTime, nnpaEstimatedTime);
}

//===----------------------------------------------------------------------===//
// Processing for each op: ReduceMean.

template <>
void estimateTimeForOp<ONNXReduceMeanV13Op>(ONNXReduceMeanV13Op op,
    const DimAnalysis *dimAnalysis, double &cpuEstimatedTime,
    double &nnpaEstimatedTime) {
  estimateTimeForElementwiseOp(op.getOperation(), op.getOperand(), dimAnalysis,
      estimatedTimeForCPU_ReduceMean_4d, estimatedTimeForNNPA_ReduceMean_4d,
      cpuEstimatedTime, nnpaEstimatedTime);
}

//===----------------------------------------------------------------------===//
// Processing for each op: MatMul.

template <>
void estimateTimeForOp<ONNXMatMulOp>(ONNXMatMulOp op,
    const DimAnalysis *dimAnalysis, double &cpuEstimatedTime,
    double &nnpaEstimatedTime) {
  estimateTimeForMatMulOp(op.getOperation(), op.getOperand(0), op.getOperand(1),
      false /*a transposed*/, false /*b transposed*/, dimAnalysis,
      cpuEstimatedTime, nnpaEstimatedTime);
}

template <>
void estimateTimeForOp<ONNXGemmOp>(ONNXGemmOp op,
    const DimAnalysis *dimAnalysis, double &cpuEstimatedTime,
    double &nnpaEstimatedTime) {
  estimateTimeForMatMulOp(op.getOperation(), op.getA(), op.getB(),
      op.getTransA(), op.getTransB(), dimAnalysis, cpuEstimatedTime,
      nnpaEstimatedTime);
}

} // namespace

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Estimate time for ops that have a model.

double estimateTimeForStickOp(Value oper) {
  // Process dim (collapse and handle dynamic sizes).
  int64_t e4, e3, e2, e1;
  std::string msg;
  processDim(oper, e4, e3, e2, e1, msg);
  return estimatedTimeForNNPA_Stick_3ds(e4 * e3, e2, e1);
}

double estimateTimeForUnstickOp(Value oper) {
  // Process dim (collapse and handle dynamic sizes).
  int64_t e4, e3, e2, e1;
  std::string msg;
  processDim(oper, e4, e3, e2, e1, msg);
  return estimatedTimeForNNPA_Unstick_3ds(e4 * e3, e2, e1);
}

bool estimateTimeForOpWithModel(Operation *op, const DimAnalysis *dimAnalysis,
    double &cpuEstimatedTime, double &nnpaEstimatedTime) {
  bool opHasModel = true;
  if (auto addOp = mlir::dyn_cast<ONNXAddOp>(op))
    estimateTimeForOp(addOp, dimAnalysis, cpuEstimatedTime, nnpaEstimatedTime);
  else if (auto divOp = mlir::dyn_cast<ONNXDivOp>(op))
    estimateTimeForOp(divOp, dimAnalysis, cpuEstimatedTime, nnpaEstimatedTime);
  else if (auto maxOp = mlir::dyn_cast<ONNXMaxOp>(op))
    estimateTimeForOp(maxOp, dimAnalysis, cpuEstimatedTime, nnpaEstimatedTime);
  else if (auto minOp = mlir::dyn_cast<ONNXMinOp>(op))
    estimateTimeForOp(minOp, dimAnalysis, cpuEstimatedTime, nnpaEstimatedTime);
  else if (auto mulOp = mlir::dyn_cast<ONNXMulOp>(op))
    estimateTimeForOp(mulOp, dimAnalysis, cpuEstimatedTime, nnpaEstimatedTime);
  else if (auto powOp = mlir::dyn_cast<ONNXPowOp>(op))
    estimateTimeForOp(powOp, dimAnalysis, cpuEstimatedTime, nnpaEstimatedTime);
  else if (auto subOp = mlir::dyn_cast<ONNXSubOp>(op))
    estimateTimeForOp(subOp, dimAnalysis, cpuEstimatedTime, nnpaEstimatedTime);
  // Unary elementwise NNPA candidate ops.
  else if (auto expOp = mlir::dyn_cast<ONNXExpOp>(op))
    estimateTimeForOp(expOp, dimAnalysis, cpuEstimatedTime, nnpaEstimatedTime);
  else if (auto logOp = mlir::dyn_cast<ONNXLogOp>(op))
    estimateTimeForOp(logOp, dimAnalysis, cpuEstimatedTime, nnpaEstimatedTime);
  else if (auto reluOp = mlir::dyn_cast<ONNXReluOp>(op))
    estimateTimeForOp(reluOp, dimAnalysis, cpuEstimatedTime, nnpaEstimatedTime);
  else if (auto sigmoidOp = mlir::dyn_cast<ONNXSigmoidOp>(op))
    estimateTimeForOp(
        sigmoidOp, dimAnalysis, cpuEstimatedTime, nnpaEstimatedTime);
  else if (auto softmaxOp = mlir::dyn_cast<ONNXSoftmaxOp>(op))
    estimateTimeForOp(
        softmaxOp, dimAnalysis, cpuEstimatedTime, nnpaEstimatedTime);
  else if (auto tanhOp = mlir::dyn_cast<ONNXTanhOp>(op))
    estimateTimeForOp(tanhOp, dimAnalysis, cpuEstimatedTime, nnpaEstimatedTime);
  // Reduce
  else if (auto reduceMeanOp = mlir::dyn_cast<ONNXReduceMeanV13Op>(op))
    estimateTimeForOp(
        reduceMeanOp, dimAnalysis, cpuEstimatedTime, nnpaEstimatedTime);
  // Matmul.
  else if (auto matMulOp = mlir::dyn_cast<ONNXMatMulOp>(op))
    estimateTimeForOp(
        matMulOp, dimAnalysis, cpuEstimatedTime, nnpaEstimatedTime);
  else if (auto gemmOp = mlir::dyn_cast<ONNXGemmOp>(op))
    estimateTimeForOp(gemmOp, dimAnalysis, cpuEstimatedTime, nnpaEstimatedTime);
  else
    opHasModel = false;

  return opHasModel;
}

} // namespace onnx_mlir
