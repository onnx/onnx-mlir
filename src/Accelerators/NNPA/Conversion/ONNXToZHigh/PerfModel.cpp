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

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/PerfModel.hpp"

#include <cmath>

#define DEBUG_TYPE "zhigh-perf-model"

using namespace mlir;
using namespace onnx_mlir;

namespace {
//===----------------------------------------------------------------------===//
// Auto-generated model.
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
// Support for unknown dimensions.

//===----------------------------------------------------------------------===//
// Processing for each op.

template <OP_TYPE>
bool isOpFasterOnNNPA(OP_TYPE *op, const DimAnalysis *dimAnalysis) {
  llvm_unreachable("should have a model for all defined ops");
}

template <>
bool isOpFasterOnNNPA(ONNXAddOp *op, const DimAnalysis *dimAnalysis) {
}

} // namespace

//===----------------------------------------------------------------------===//
// Function to perform evaluation.

namespace onnx_mlir {

bool isOpFasterOnNNPA(mlir::Operation *op, const DimAnalysis *dimAnalysis) {
  if (auto addOp = dyn_cast<ONNXAddOp>(op))
    return isOpFasterOnNNPA(addOp, dimAnalysis);

  // Unknown, issue a warning and assume its faster on NNPA
  llvm::errs()
      << "The operation below is a candidate for NNPA execution but has no "
         "cost benefit analysis. Please add analysis.\n";
  return true;
}

} // namespace onnx_mlir
