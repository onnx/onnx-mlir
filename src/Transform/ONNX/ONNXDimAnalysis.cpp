/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ONNXDimAnalysis.cpp - ONNX Dimension Analysis ---------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements an analysis on unknown dimensions in ONNX ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/TypeUtilities.hpp"
#include "src/Transform/ONNX/ONNXDimAnalysis.hpp"

#define DEBUG_TYPE "dim_analysis"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper functions for dimension analysis.
//===----------------------------------------------------------------------===//

/// Check if two sets are overlapping or not.
bool areOverlapping(const onnx_mlir::DimAnalysis::DimSetT &lhs,
    const onnx_mlir::DimAnalysis::DimSetT &rhs) {
  // Call `contains` on the smaller set for better performance.
  if (lhs.size() < rhs.size()) {
    for (auto &ti : lhs) {
      if (rhs.contains(ti)) {
        return true;
      }
    }
  } else {
    for (auto &ti : rhs) {
      if (lhs.contains(ti)) {
        return true;
      }
    }
  }
  return false;
}

/// Given a QuestionMarkIndexExpr representing an unknown dimension, find the
/// same unknown dimensions in the inputs.
void findAndAddSameDim(const onnx_mlir::QuestionmarkIndexExpr &qmOuputIE,
    mlir::ValueRange operands, onnx_mlir::DimAnalysis::DimSetT &sameDims) {
  // Cannot process if the question mark is not a specific one.
  if (!qmOuputIE.specificQuestionmark())
    return;
  // Find the same unknown dimension in the inputs.
  for (Value v : operands) {
    if (onnx_mlir::isFromNone(v))
      continue;
    int64_t rank = onnx_mlir::getRank(v.getType());
    onnx_mlir::MemRefBoundsIndexCapture vDims(v);
    for (int64_t i = 0; i < rank; ++i) {
      if (qmOuputIE.sameQuestionmark(vDims.getDim(i)))
        sameDims.insert(onnx_mlir::DimAnalysis::DimT(v, i));
    }
  }
}

/// Given an unknown dimension, find the same unknown dimensions in the inputs.
/// This function uses ShapeHelper to explore the same unknown dimensions.
/// Use this function for operations that use adaptor to compute shape.
template <typename ONNX_OP, typename SHAPE_HELPER>
void exploreSameInputDims(const onnx_mlir::DimAnalysis::DimT &dim, ONNX_OP op,
    onnx_mlir::DimAnalysis::DimSetT &sameDims) {
  SHAPE_HELPER shapeHelper(op.getOperation(), {});
  shapeHelper.computeShapeAndAssertOnFailure();
  // The operation may have multiple outputs, find the index of the processing
  // output.
  Value outputTensor = dim.first;
  uint64_t tensorIndex = 0;
  for (uint64_t i = 0; i < op->getNumResults(); ++i) {
    if (op->getResult(i) == outputTensor) {
      tensorIndex = i;
      break;
    }
  }
  // Find the unknown input dimensions that were transferred to the unknown
  // output dimension.
  uint64_t dimIndex = dim.second;
  onnx_mlir::QuestionmarkIndexExpr qmOuputIE =
      shapeHelper.getOutputDims(tensorIndex)[dimIndex];
  findAndAddSameDim(qmOuputIE, op.getOperation()->getOperands(), sameDims);
}

/// Given an unknown dimension, find the same unknown dimensions in the inputs.
/// This function uses ShapeHelper to explore the same unknown dimensions.
/// Use this function for unary operations.
void exploreSameInputDimsUnaryOp(const onnx_mlir::DimAnalysis::DimT &dim,
    mlir::Operation *op, onnx_mlir::DimAnalysis::DimSetT &sameDims) {
  onnx_mlir::ONNXUnaryOpShapeHelper shapeHelper(op, {});
  shapeHelper.computeShapeAndAssertOnFailure();
  // Find the unknown input dimensions that were transferred to the unknown
  // output dimension.
  onnx_mlir::QuestionmarkIndexExpr qmOuputIE =
      shapeHelper.getOutputDims()[dim.second];
  findAndAddSameDim(qmOuputIE, op->getOperands(), sameDims);
}

/// Given an unknown dimension, find the same unknown dimensions in the inputs.
/// This function uses ShapeHelper to explore the same unknown dimensions.
/// Use this function for binary operations.
void exploreSameInputDimsBinaryOp(const onnx_mlir::DimAnalysis::DimT &dim,
    mlir::Operation *op, onnx_mlir::DimAnalysis::DimSetT &sameDims) {
  Value A = op->getOperands()[0];
  Value B = op->getOperands()[1];

  // Build shape helper
  onnx_mlir::ONNXBroadcastOpShapeHelper shapeHelper(
      op, ArrayRef<Value>({A, B}));
  shapeHelper.computeShapeAndAssertOnFailure();
  // Find the unknown input dimensions that were transferred to the unknown
  // output dimension.
  onnx_mlir::QuestionmarkIndexExpr qmOuputIE =
      shapeHelper.getOutputDims()[dim.second];
  findAndAddSameDim(qmOuputIE, op->getOperands(), sameDims);
}

} // namespace

namespace onnx_mlir {

DimAnalysis::DimAnalysis(ArrayRef<Value> vals) {
  for (Value val : vals)
    if (!isFromNone(val))
      build(val);

  LLVM_DEBUG(llvm::dbgs() << "The number of unknown dims in the IR: "
                          << numOfUnknownDims << "\n");
}

DimAnalysis::DimAnalysis(ModuleOp moduleOp) {
  moduleOp.walk([&](Operation *op) {
    for (Value output : op->getResults())
      build(output);
  });
  LLVM_DEBUG(llvm::dbgs() << "The number of unknown dims in the IR: "
                          << numOfUnknownDims << "\n");
}

void DimAnalysis::build(Value val) {
  if (auto tensorType = val.getType().dyn_cast<RankedTensorType>()) {
    for (unsigned i = 0; i < tensorType.getRank(); ++i) {
      // Only care about unknown dimensions.
      if (tensorType.isDynamicDim(i)) {
        DimT ti(val, i);
        DimSetT dimSet;
        dimSet.insert(ti);
        dimSetMap[setCounter++] = dimSet;
        numOfUnknownDims++;
      }
    }
  }
}

bool DimAnalysis::sameUnknownDim(mlir::Value tensor1, uint64_t dimAxis1,
    mlir::Value tensor2, uint64_t dimAxis2) const {
  if ((tensor1 == tensor2) && (dimAxis1 == dimAxis2))
    return true;

  DimT dim1(tensor1, dimAxis1);
  DimT dim2(tensor2, dimAxis2);
  // Two dims are the same if they are in the same set.
  for (auto &entry : dimSetMap) {
    DimSetT dims = entry.second;
    if (dims.contains(dim1) && dims.contains(dim2))
      return true;
  }
  return false;
}

bool DimAnalysis::sameShape(Value tensor1, Value tensor2) const {
  ShapedType tensor1Type = tensor1.getType().cast<ShapedType>();
  ShapedType tensor2Type = tensor2.getType().cast<ShapedType>();
  if (!tensor1Type.hasRank() || !tensor2Type.hasRank())
    return false;
  // Different rank, return false.
  if (tensor1Type.getRank() != tensor2Type.getRank())
    return false;
  // Both tensors have static dimensions.
  if (tensor1Type.hasStaticShape() && tensor2Type.hasStaticShape())
    return (tensor1Type.getShape() == tensor2Type.getShape());
  // There are unknown dimensions, use DimAnalysis to check equality.
  for (unsigned i = 0; i < tensor1Type.getRank(); ++i) {
    int64_t dim1 = tensor1Type.getShape()[i];
    int64_t dim2 = tensor2Type.getShape()[i];
    if (dim1 != dim2)
      return false;
    // Same dimensions but can be unknown (-1).
    if (ShapedType::isDynamic(dim1)) {
      // Two unknown dimensions are NOT the same at compile time.
      if (!sameUnknownDim(tensor1, i, tensor2, i))
        return false;
    }
  }
  return true;
}

void DimAnalysis::dump() const {
  llvm::outs() << numOfUnknownDims
               << " unknown dimensions (not including block arguments) are "
                  "classified into "
               << dimSetMap.size() << " sets.\n";
  for (auto &entry : dimSetMap) {
    uint64_t i = entry.first;
    DimSetT dimSet = entry.second;
    llvm::outs() << "- Set " << i << " (size: " << dimSet.size() << "):\n";
    for (auto &ti : dimSet)
      llvm::outs() << "  - Dim " << ti.second << " of " << ti.first << "\n";
  }
}

void DimAnalysis::analyze() {
  // Build sets of the same unknown dimensions and merge them until a fixed
  // point where there is no update on each set.
  bool continued = true;
  while (continued) {
    // Local search and update each set of unknown dimensions.
    continued = updateDimSets();

    // Merge sets if there is update.
    if (continued)
      // Two sets with a common dimension will be merged into a single set
      // consisting of elements from each set.
      mergeDimSets();
  }

  LLVM_DEBUG(
      llvm::dbgs() << "The number of sets of same unknown dims in the IR: "
                   << dimSetMap.size() << "\n");
}

bool DimAnalysis::updateDimSets() {
  bool updated = false;
  for (auto &entry : dimSetMap) {
    DimSetT &dimSet = entry.getSecond();
    // Explore new dims.
    DimSetT newSameDims;
    for (auto &d : dimSet) {
      visitDim(d, newSameDims);
    }
    // Update the dim set.
    for (auto &d : newSameDims) {
      if (!dimSet.contains(d)) {
        // Found new unknown dims.
        dimSet.insert(d);
        updated = true;
      }
    }
  }
  return updated;
}

void DimAnalysis::mergeDimSets() {
  bool continued = true;
  while (continued) {
    // Get keys.
    SmallVector<uint64_t, 4> keys;
    for (auto &ds : dimSetMap)
      keys.emplace_back(ds.first);

    // Check and merge sets.
    llvm::SmallDenseSet<uint64_t, 4> erasedKeys;
    for (uint64_t i = 0; i < keys.size() - 1; ++i) {
      int64_t lhsKey = keys[i];
      if (erasedKeys.contains(lhsKey))
        continue;
      for (uint64_t k = i + 1; k < keys.size(); ++k) {
        uint64_t rhsKey = keys[k];
        if (erasedKeys.contains(rhsKey))
          continue;
        // Two sets that have a common dim can be merged.
        DimSetT &lhs = dimSetMap[lhsKey];
        DimSetT &rhs = dimSetMap[rhsKey];
        if (areOverlapping(lhs, rhs)) {
          /// Merge the rhs set into the lhs set.
          for (auto &ti : rhs)
            lhs.insert(ti);
          erasedKeys.insert(rhsKey);
        }
      }
    }

    // Erase merged sets.
    for (uint64_t key : erasedKeys)
      dimSetMap.erase(key);
    continued = (erasedKeys.size() > 0);
  }
}

void DimAnalysis::visitDim(
    DimAnalysis::DimT &dim, DimAnalysis::DimSetT &sameDims) const {
  Value tensor = dim.first;
  uint64_t dimIndex = dim.second;

  // Tensor is a block argument. Nothing to do further.
  if (tensor.isa<BlockArgument>())
    return;

  // Find where a dimension comes from, depending on operation semantics.
  // We utilize the operation's shape helper for this purpose as much as
  // possible.
  Operation *op = tensor.getDefiningOp();

  // UnaryOp
  if (isa<ONNXCastOp>(op) || isa<ONNXPowOp>(op) || isa<ONNXReluOp>(op) ||
      isa<ONNXReciprocalOp>(op) || isa<ONNXTanhOp>(op) ||
      isa<ONNXSigmoidOp>(op) || isa<ONNXSoftmaxOp>(op) || isa<ONNXSqrtOp>(op)) {
    exploreSameInputDimsUnaryOp(dim, op, sameDims);
    return;
  }

  // BinaryOp
  if (isa<ONNXAddOp>(op) || isa<ONNXDivOp>(op) || isa<ONNXMulOp>(op) ||
      isa<ONNXSubOp>(op)) {
    exploreSameInputDimsBinaryOp(dim, op, sameDims);
    // If we know by this analysis that two unknown dims at the same index are
    // the same, then the output dim must be the same too.
    Value A = op->getOperands()[0];
    Value B = op->getOperands()[1];
    Type aType = A.getType();
    Type bType = B.getType();
    uint64_t aRank = onnx_mlir::getRank(aType);
    uint64_t bRank = onnx_mlir::getRank(bType);
    uint64_t maxRank = std::max(aRank, bRank);
    if ((aRank != 0) && (bRank != 0)) {
      ArrayRef<int64_t> aShape = onnx_mlir::getShape(aType);
      ArrayRef<int64_t> bShape = onnx_mlir::getShape(bType);
      // aDim == bDim (unknown), there is no broadcasting and aDim == outputDim.
      int64_t aDimIndex = dimIndex - (maxRank - aRank);
      int64_t bDimIndex = dimIndex - (maxRank - bRank);
      if ((aDimIndex >= 0) && (bDimIndex >= 0) &&
          ShapedType::isDynamic(aShape[aDimIndex]) &&
          ShapedType::isDynamic(bShape[bDimIndex]) &&
          onnx_mlir::DimAnalysis::sameUnknownDim(A, aDimIndex, B, bDimIndex))
        sameDims.insert(onnx_mlir::DimAnalysis::DimT(A, aDimIndex));
    }
    return;
  }

  // AveragePoolOp
  if (auto poolOp = dyn_cast<ONNXAveragePoolOp>(op)) {
    exploreSameInputDims<ONNXAveragePoolOp, ONNXAveragePoolOpShapeHelper>(
        dim, poolOp, sameDims);
    return;
  }

  // ArgMaxOp
  if (auto argmaxOp = dyn_cast<ONNXArgMaxOp>(op)) {
    exploreSameInputDims<ONNXArgMaxOp, ONNXArgMaxOpShapeHelper>(
        dim, argmaxOp, sameDims);
    return;
  }

  // ConstantOfShapeOp
  if (auto constOp = dyn_cast<ONNXConstantOfShapeOp>(op)) {
    if (!areDimsFromConcat(constOp.input()))
      return;
    SmallVector<Value, 4> inputs;
    getDims(constOp.input(), inputs);
    DimT newSameDim(inputs[dimIndex], dimIndex);
    sameDims.insert(newSameDim);
    return;
  }

  // ConvOp
  if (auto convOp = dyn_cast<ONNXConvOp>(op)) {
    exploreSameInputDims<ONNXConvOp, ONNXConvOpShapeHelper>(
        dim, convOp, sameDims);
    return;
  }

  // DimOp
  if (auto dimOp = dyn_cast<ONNXDimOp>(op)) {
    DimT newSameDim(dimOp.data(), dimOp.axis());
    sameDims.insert(newSameDim);
    return;
  }

  // GemmOp
  if (auto gemmOp = dyn_cast<ONNXGemmOp>(op)) {
    exploreSameInputDims<ONNXGemmOp, ONNXGemmOpShapeHelper>(
        dim, gemmOp, sameDims);
    return;
  }

  // MatMulOp
  if (auto matmulOp = dyn_cast<ONNXMatMulOp>(op)) {
    exploreSameInputDims<ONNXMatMulOp, ONNXMatMulOpShapeHelper>(
        dim, matmulOp, sameDims);
    // If we know by this analysis that two unknown dims at the same index in
    // the batchsize space are the same, then the output dim must be the same
    // too.
    Value A = matmulOp.A();
    Value B = matmulOp.B();
    Type aType = A.getType();
    Type bType = B.getType();
    uint64_t aRank = getRank(aType);
    uint64_t bRank = getRank(bType);
    uint64_t maxRank = std::max(aRank, bRank);
    if (dimIndex <= maxRank - 2) { // In the batchsize space.
      ArrayRef<int64_t> aShape = getShape(aType);
      ArrayRef<int64_t> bShape = getShape(bType);
      // aDim == bDim (unknown), there is no broadcasting and aDim == outputDim.
      int64_t aDimIndex = dimIndex - (maxRank - aRank);
      int64_t bDimIndex = dimIndex - (maxRank - bRank);
      if ((aDimIndex >= 0) && (bDimIndex >= 0) &&
          ShapedType::isDynamic(aShape[aDimIndex]) &&
          ShapedType::isDynamic(bShape[bDimIndex]) &&
          sameUnknownDim(A, aDimIndex, B, bDimIndex))
        sameDims.insert(DimT(A, aDimIndex));
    }
    return;
  }

  // MaxPoolSingleOutOp
  if (auto poolOp = dyn_cast<ONNXMaxPoolSingleOutOp>(op)) {
    exploreSameInputDims<ONNXMaxPoolSingleOutOp,
        ONNXMaxPoolSingleOutOpShapeHelper>(dim, poolOp, sameDims);
    return;
  }

  // PadOp
  if (auto padOp = dyn_cast<ONNXPadOp>(op)) {
    exploreSameInputDims<ONNXPadOp, ONNXPadOpShapeHelper>(dim, padOp, sameDims);
    return;
  }

  // ReshapeOp
  if (auto reshapeOp = dyn_cast<ONNXReshapeOp>(op)) {
    if (reshapeOp.allowzero() != 0)
      return;

    // The output dimension i can be from
    // - shape[i] or,
    // - data[j] if
    //    - dim j and i are the only unknown dimension in data and
    // output, respectively, and
    //    - the products of the static dimensions in data and output are
    //    equal.
    //
    // It's interesting that if shape[i] is arg0[ii] and data[j] is arg1[jj],
    // we can say that arg0[ii] == arg1[jj], that can be used to verify user
    // inputs.

    // Get the unknown dimension from shape. Use this to update the current
    // dimension.
    if (areDimsFromConcat(reshapeOp.shape())) {
      SmallVector<Value, 4> shapeDims;
      getDims(reshapeOp.shape(), shapeDims);
      Value dimFromShape = shapeDims[dimIndex];
      DimT newSameDim(dimFromShape, dimIndex);
      sameDims.insert(newSameDim);
    }

    // Get the unknown dimension from data.
    RankedTensorType dataType =
        reshapeOp.data().getType().dyn_cast<RankedTensorType>();
    RankedTensorType outputType =
        reshapeOp.reshaped().getType().dyn_cast<RankedTensorType>();
    // Check if there is only one unknown dimension in the data and output.
    bool isDataOK = (llvm::count(dataType.getShape(), -1) == 1);
    bool isOutputOK = (llvm::count(outputType.getShape(), -1) == 1);
    // Check if the products of static sizes in the data and output are equal.
    // It's ok to count -1 (unknown dimension) in the size.
    int64_t dataSize = 1, outputSize = 1;
    for (int64_t i = 0; i < dataType.getRank(); ++i)
      dataSize *= dataType.getShape()[i];
    for (int64_t i = 0; i < outputType.getRank(); ++i)
      outputSize *= outputType.getShape()[i];
    // Conditions hold, the unknown dimension can be from the data.
    if (isDataOK && isOutputOK && (dataSize == outputSize)) {
      // Find the index of the unknown dimension in the data.
      int64_t unknownDimIndexInData = -1;
      for (int64_t i = 0; i < dataType.getRank(); ++i)
        if (dataType.isDynamicDim(i)) {
          unknownDimIndexInData = i;
          break;
        }
      assert(unknownDimIndexInData != -1 &&
             "Failed to obtain the index of the unknown dimension in the data");
      DimT newSameDim(reshapeOp.data(), unknownDimIndexInData);
      sameDims.insert(newSameDim);
    }
    return;
  }

  // ReduceMeanOp
  if (auto reduceMeanOp = dyn_cast<ONNXReduceMeanOp>(op)) {
    // TODO: replace the code here by the following code once ReduceMean uses
    // IndexExpr for its shape inference.
    // ```c
    // exploreSameInputDims<ONNXReduceMeanOp, ONNXReduceMeanOpShapeHelper>(
    //    dim, reduceMeanOp, sameDims);
    // ```

    // Only support keepdims at this moment.
    if (reduceMeanOp.keepdims() != 1)
      return;
    // Reduction dims in output are always 1. So an unknown dim in the output
    // is at the same index as the one in the input.
    llvm::Optional<ArrayAttr> axesAttr = reduceMeanOp.axes();
    if (axesAttr.has_value()) {
      // Do nothing if the target dim is the reduction dim.
      for (size_t i = 0; i < ArrayAttrSize(axesAttr); ++i) {
        if (ArrayAttrIntVal(axesAttr, i) == (int64_t)dim.second)
          return;
      }
      // The target dim is not the reduction dim, it would be the same as the
      // input dim.
      sameDims.insert(DimT(reduceMeanOp.data(), dim.second));
    }
    return;
  }

  // SliceOp
  if (auto sliceOp = dyn_cast<ONNXSliceOp>(op)) {
    exploreSameInputDims<ONNXSliceOp, ONNXSliceOpShapeHelper>(
        dim, sliceOp, sameDims);
    return;
  }

  // SplitOp
  if (auto splitOp = dyn_cast<ONNXSplitOp>(op)) {
    exploreSameInputDims<ONNXSplitOp, ONNXSplitOpShapeHelper>(
        dim, splitOp, sameDims);
    return;
  }

  // SqueezeOp
  if (auto squeezeOp = dyn_cast<ONNXSqueezeOp>(op)) {
    exploreSameInputDims<ONNXSqueezeOp, ONNXSqueezeOpShapeHelper>(
        dim, squeezeOp, sameDims);
    return;
  }

  // TransposeOp
  if (auto transposeOp = dyn_cast<ONNXTransposeOp>(op)) {
    exploreSameInputDims<ONNXTransposeOp, ONNXTransposeOpShapeHelper>(
        dim, transposeOp, sameDims);
    return;
  }

  // Unsqueeze
  if (auto unsqueezeOp = dyn_cast<ONNXUnsqueezeOp>(op)) {
    exploreSameInputDims<ONNXUnsqueezeOp, ONNXUnsqueezeOpShapeHelper>(
        dim, unsqueezeOp, sameDims);
    return;
  }

  // Unsupported operations, just stop the analysis.
  return;
}

} // namespace onnx_mlir

namespace {

/// This pass is for testing purpose. It introduces onnx.DimGroup to the IR to
/// show group IDs. Unknown dimensions with the same group ID are supposed to be
/// equal.
struct ONNXDimAnalysisPass
    : public PassWrapper<ONNXDimAnalysisPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXDimAnalysisPass)

  StringRef getArgument() const override { return "onnx-dim-analysis"; }

  StringRef getDescription() const override {
    return "Perform an analysis on unknown dimensions in ONNX ops";
  }

  void runOnOperation() final;
};

void ONNXDimAnalysisPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  OpBuilder b(moduleOp.getContext());

  onnx_mlir::DimAnalysis testOp(moduleOp);
  testOp.analyze();
  // testOp.dump();

  // Add onnx.DimGroup into the IR for LIT tests.
  onnx_mlir::DimAnalysis::DimSetMapT mapping = testOp.getGroupingResult();
  for (auto &entry : mapping) {
    uint64_t groupID = entry.first;
    onnx_mlir::DimAnalysis::DimSetT dimSet = entry.second;
    for (auto &ti : dimSet) {
      Value val = ti.first;
      uint64_t dimAxis = ti.second;
      Location loc = val.getLoc();
      if (auto arg = val.dyn_cast<BlockArgument>()) {
        Block *owner = arg.getOwner();
        b = OpBuilder::atBlockBegin(owner);
      } else {
        Operation *op = val.getDefiningOp();
        b.setInsertionPointAfter(op);
        if (auto dimOp = dyn_cast<ONNXDimOp>(op))
          val = dimOp.data();
      }
      onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(b, loc);
      create.onnx.dimGroup(val, dimAxis, groupID);
    }
  }
}

} // namespace

namespace onnx_mlir {

/*!
 * Create a ONNXDimAnalysis pass.
 * This pass is used to test the DimAnalysis class.
 */
std::unique_ptr<mlir::Pass> createONNXDimAnalysisPass() {
  return std::make_unique<ONNXDimAnalysisPass>();
}

} // namespace onnx_mlir
