/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ONNXDimAnalysis.hpp - ONNX Dimension Analysis ---------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements an analysis on unknown dimensions in ONNX ops.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinTypes.h"

namespace onnx_mlir {

class DimAnalysis {
public:
  // Dimension type is a pair of a tensor and a dimension axis.
  using DimT = std::pair<mlir::Value, uint64_t>;
  // A set of dimensions.
  using DimSetT = llvm::SmallDenseSet<DimT, 4>;
  // A mapping between an ID and a set of dimensions.
  using DimSetMapT = llvm::SmallDenseMap<uint64_t, DimSetT, 4>;

public:
  /// Create a new analysis for specific values.
  DimAnalysis(llvm::ArrayRef<mlir::Value> vals);

  /// Create a new analysis for all values in a module.
  DimAnalysis(mlir::ModuleOp op);

  /// Analyzes the relationship amongs unknown dimensions.
  void analyze();

  /// Returns the grouping result of unknown dimensions.
  DimSetMapT getGroupingResult() const { return dimSetMap; }

  /// Test if two unknown dimensions are the same or not.
  /// Each dimension is identified by its tensor and axis.
  bool areSame(mlir::Value tensor1, uint64_t dimAxis1, mlir::Value tensor2,
      uint64_t dimAxis2) const;

  /// Dumps the analysis information.
  void dump() const;

private:
  /// Initializes the internal mappings.
  void build(mlir::Value val);

  /// Update each set of unknown dimensions to include the same unknown
  /// dimensions. This is a local update in the sense that the search space
  /// includes unknown dimensions that directly link to the dimensions in the
  /// set via defining operations.
  bool updateDimSets();

  /// Merge sets of unknown dimensions. Two sets with a common dimension will
  /// be merged into a single set consisting of elements from each set.
  void mergeDimSets();

  /// Visit an unknown dimension and find new same unknown dimensions.
  void visitDim(DimT &dim, DimSetT &sameDims) const;

private:
  int64_t setCounter = 0;
  int64_t numOfUnknownDims = 0;
  /// This mapping maps each unknown dimension in the tensor to a set of same
  /// unknown dimensions.
  DimSetMapT dimSetMap;
};

} // namespace onnx_mlir
