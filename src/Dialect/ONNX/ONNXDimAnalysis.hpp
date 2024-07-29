/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ONNXDimAnalysis.hpp - ONNX Dimension Analysis ---------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements an analysis on dynamic dimensions in ONNX ops.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ONNX_DIM_ANALYSIS_H
#define ONNX_MLIR_ONNX_DIM_ANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

namespace onnx_mlir {

class DimAnalysis {
public:
  // Dimension type is a pair of a tensor and a dimension axis.
  using DimT = std::pair<mlir::Value, uint64_t>;
  // A set of dimensions.
  using DimSetT = llvm::SmallDenseSet<DimT, 4>;
  // A mapping between an ID and a set of dimensions. The ID is called set ID.
  // This data structure is to store the (intermediate or final) result of
  // analysis.
  using DimSetMapT = llvm::SmallDenseMap<uint64_t, DimSetT, 4>;

public:
  /// Create a new analysis for specific values.
  DimAnalysis(llvm::ArrayRef<mlir::Value> vals);

  /// Create a new analysis for all values in a module.
  DimAnalysis(mlir::ModuleOp op);

  /// Analyzes the relationship among dynamic dimensions.
  /// Current implementation uses a fixed-point iteration algorithm,
  /// where there are two phases at each iteration:
  ///   - Expand: find and add same dynamic dimensions to each set. Same dynamic
  ///     dimensions are discovered by utilizing ShapeHelper of an operation or
  ///     utilizing the current result of this analysis.
  ///   - Merge: sets that have common elements will be merged into a single
  ///     set. The set with larger ID will be merged into the set with smaller
  ///     ID.
  /// The fixed point condition is: there is no update in each set.
  void analyze();

  /// Returns the grouping result of dynamic dimensions.
  DimSetMapT getGroupingResult() const { return dimSetMap; }

  /// Test if two dimensions are the same or not.
  /// Each dimension is identified by its tensor and axis. Negative axis is
  /// interpreted as index from the innermost dimension. Out of bound axis
  /// results in sameDim to return false.
  bool sameDim(mlir::Value tensor1, int64_t dimAxis1, mlir::Value tensor2,
      int64_t dimAxis2) const;

  /// Test if two dynamic dimensions are the same or not.
  /// Each dimension is identified by its tensor and axis. Negative axis is
  /// interpreted as index from the innermost dimension. Out of bound axis
  /// results in sameDynDim to return false.
  bool sameDynDim(mlir::Value tensor1, int64_t dimAxis1, mlir::Value tensor2,
      int64_t dimAxis2) const;

  /// Test if two tensors have the same shape or not.
  bool sameShape(mlir::Value tensor1, mlir::Value tensor2) const;

  /// Test if dynamic dimensions of two tensors are the same or not.
  /// Static vs static dimension is ignored.
  /// Static vs dynamic dimension is false as usual.
  /// For example: return true for tensor<?x2xf32> and tensor<?x5xf32> if their
  /// first dimensions are the same.
  bool sameDynShape(mlir::Value tensor1, mlir::Value tensor2) const;

  /// Test if `tensor1` is broadcasting to `tensor2` using the last dimension.
  /// For example: return true for tensor<?x1xf32> and tensor<?x5xf32> if their
  /// first dimensions are the same.
  /// Note that: broadcasting direction is important.
  bool broadcastLastDim(mlir::Value tensor1, mlir::Value tensor2) const;

  /// Dumps the analysis information.
  void dump() const;

private:
  /// Initializes the internal mappings.
  /// Each dynamic dimension is initially assigned to a singleton set.
  void build(mlir::Value val);

  /// Initializes the internal mappings for a single dynamic dimension.
  /// The dynamic dimension is initially assigned to a newly-created set or an
  /// existing set depending on `setID` is -1 or not.
  /// This method returns the set ID that contains the dimension.
  int64_t build(DimT d, int64_t setID = -1);

  /// Initializes the internal mappings for function arguments and resutls.
  void buildFunctionArgsRes(mlir::func::FuncOp funcOp);

  // Create dims for function arguments.
  /// Update each set of dynamic dimensions to include the same dynamic
  /// dimensions. This is a local update in the sense that the search space
  /// includes dynamic dimensions that directly link to the dimensions in the
  /// set via defining operations.
  bool updateDimSets();

  /// Merge sets of dynamic dimensions. Two sets with a common dimension will
  /// be merged into a single set consisting of elements from each set.
  void mergeDimSets();

  /// Visit a dynamic dimension and find new same dynamic dimensions.
  void visitDim(DimT &dim, DimSetT &sameDims) const;

  /// Get onnx.dim_params value from a function argument/result and put it into
  /// a map.
  /// TODO: find a new home for this function.
  void getONNXDimParams(std::map<unsigned, std::string> &indexParamMap,
      mlir::ArrayAttr argResAttr, unsigned index);

private:
  int64_t setCounter = 0;
  int64_t numOfDynamicDims = 0;
  /// This mapping maps each dynamic dimension in the tensor to a set of same
  /// dynamic dimensions.
  DimSetMapT dimSetMap;
};

} // namespace onnx_mlir
#endif
