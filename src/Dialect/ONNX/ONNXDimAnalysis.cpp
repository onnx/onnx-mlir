/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ONNXDimAnalysis.cpp - ONNX Dimension Analysis ---------------===//
//
// Copyright 2022-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file implements an analysis on dynamic dimensions in ONNX ops.
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
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/TypeUtilities.hpp"

#define DEBUG_TYPE "dim_analysis"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Helper functions for dimension analysis.
//===----------------------------------------------------------------------===//

/// Check if two sets are overlapping or not.
static bool areOverlapping(
    const DimAnalysis::DimSetT &lhs, const DimAnalysis::DimSetT &rhs) {
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

/// Insert a dynamic dimension into the analysis sets.
/// It is expected that the shape-related operations were simplified by
/// `simplify-shape-related-ops-onnx` pass before this analysis pass. Thus,
/// operations that are related to dynamic dimensions include DimOp, ConstantOp,
/// and CastOp.
/// Also, by `simplify-shape-related-ops-onnx`, static dimensions would be
/// propagated well, so we care about dynamic dimensions in this analysis only.
static std::optional<DimAnalysis::DimT> insertDimWhenUseful(const Value tensor,
    const uint64_t dimIndex, DimAnalysis::DimSetT &sameDims) {
  auto tensorType = cast<ShapedType>(tensor.getType());

  bool okToInsert = false;
  if (tensor.isa<BlockArgument>()) {
    okToInsert = true;
  } else {
    Operation *op = tensor.getDefiningOp();
    if (isa<ONNXConstantOp, ONNXCastOp, ONNXDimOp>(op) ||
        tensorType.isDynamicDim(dimIndex))
      okToInsert = true;
  }

  if (!okToInsert)
    return std::nullopt;

  DimAnalysis::DimT dim(tensor, dimIndex);
  sameDims.insert(dim);
  return dim;
}

static bool handleAndTestInBound(int64_t &axis, ShapedType type) {
  int64_t rank = type.getRank();
  if (axis < 0)
    axis += rank;
  return axis >= 0 && axis < rank;
}

/// Given a QuestionMarkIndexExpr representing a dynamic dimension, find the
/// same dynamic dimensions in the inputs.
static void findAndAddSameDim(const QuestionmarkIndexExpr &qmOuputIE,
    mlir::Operation *op, mlir::ValueRange operands,
    DimAnalysis::DimSetT &sameDims) {
  mlir::Location loc = op->getLoc();
  IndexExprBuilderForAnalysis createIE(loc);

  // Cannot process if the question mark is not a specific one.
  if (!qmOuputIE.specificQuestionmark())
    return;
  // Find the same dynamic dimension in the inputs.
  for (Value v : operands) {
    if (isNoneValue(v))
      continue;
    int64_t rank = getRank(v.getType());
    DimsExpr vDims;
    createIE.getShapeAsDims(v, vDims);
    for (int64_t i = 0; i < rank; ++i) {
      if (qmOuputIE.sameQuestionmark(vDims[i])) {
        if (auto d = insertDimWhenUseful(v, i, sameDims))
          LLVM_DEBUG(llvm::dbgs() << "  - Added a new dim(" << d.value().first
                                  << ", " << d.value().second << ")\n");
      }
    }
  }
}

/// Given a dynamic dimension, find the same dynamic dimensions in the inputs.
/// This function uses ShapeHelper to explore the same dynamic dimensions.
/// Use this function for operations that use adaptor to compute shape.
static bool exploreSameDimsUsingShapeHelper(const DimAnalysis::DimT &dim,
    mlir::Operation *op, DimAnalysis::DimSetT &sameDims) {
  LLVM_DEBUG(llvm::dbgs() << "Explore using shape helper\n");
  // Has this op a ShapeHelper interface?
  auto shape_op = llvm::dyn_cast<ShapeHelperOpInterface>(*op);
  if (!shape_op)
    return false;

  // Get its shape interface.
  ONNXOpShapeHelper *shapeHelper =
      shape_op.getShapeHelper(op, {}, nullptr, nullptr);
  // If no shape helper, or unimplemented, just abort.
  if (!shapeHelper || !shapeHelper->isImplemented())
    return false;

  // Compute shape.
  if (failed(shapeHelper->computeShape())) {
    delete shapeHelper;
    return false;
  }

  // The operation may have multiple outputs, find the index of the processing
  // output.
  Value outputTensor = dim.first;
  int64_t tensorIndex = -1;
  for (int64_t i = 0; i < op->getNumResults(); ++i) {
    if (op->getResult(i) == outputTensor) {
      tensorIndex = i;
      break;
    }
  }
  assert(tensorIndex != -1 && "Value does not exist");

  // Find the dynamic input dimensions that were transferred to the dynamic
  // output dimension.
  uint64_t dimIndex = dim.second;
  QuestionmarkIndexExpr qmOuputIE =
      shapeHelper->getOutputDims(tensorIndex)[dimIndex];
  findAndAddSameDim(qmOuputIE, op, op->getOperands(), sameDims);
  delete shapeHelper;
  return true;
}

// clang-format off
/// Given a dynamic dimension in the output, find the same dynamic dimensions in
/// the inputs. This function is used for operations whose one of the input
/// tensors defines the output shape. For example, ConstantOfShape:
/// ```
//  %d1    = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
//  %d2    = "onnx.Dim"(%arg1) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
//  %d3    = "onnx.Dim"(%arg2) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
//  %shape = "onnx.Concat"(%d1, %d2, %d3) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
/// %c     = "onnx.ConstantOfShape"(%shape) {value = dense<[1.0]> : tensor<1xf32>} : (tensor<3xi64>) -> tensor<?x?x?xf32>
/// ```
/// It is expected that the shape tensor is produced by concatenation of
/// dimensions (done by `simplify-shape-related-ops-onnx` pass in advance).
/// Otherwise, there is no information for analyzing further.
///
/// By this analyis, we know that the 1st, 2nd, and 3rd dim of `%c` is `%d1`,
/// `%d2`, and `%d3` respectively.
///
// clang-format on
static bool exploreSameDimsUsingShapeInput(const DimAnalysis::DimT &dim,
    mlir::Operation *op, DimAnalysis::DimSetT &sameDims) {
  LLVM_DEBUG(llvm::dbgs() << "Explore using shape input\n");
  uint64_t outputDimIndex = dim.second;
  // It's often the case the input and output dim indices are the same.
  // Otherwise, the input dim index will be refined depending on the operation.
  uint64_t inputDimIndex = outputDimIndex;

  // If an operation has an operand that stores the output shape, use the
  // operand to explore same dimensions.
  // Below are ONNX operations we know that specify the output shape via an
  // operand. Sorted in the alphabetical order.
  Value shapeInput = nullptr;
  if (auto onnxOp = dyn_cast<ONNXCenterCropPadOp>(op)) {
    // `shape` stores shape information for dimensions specified by `axes`.
    // `outputDimIndex` must be in `axes` in order to get dim from `shape`.
    auto outputType = cast<ShapedType>(onnxOp.getResult().getType());
    SmallVector<int64_t, 4> axesInt;
    ArrayAttr axes = onnxOp.getAxesAttr();
    if (axes) {
      ArrayAttrIntVals(axes, axesInt);
    } else {
      for (int64_t i = 0; i < outputType.getRank(); ++i)
        axesInt.emplace_back(i);
    }
    bool found = false;
    for (size_t i = 0; i < axesInt.size(); ++i) {
      int64_t axis = axesInt[i];
      if (!handleAndTestInBound(axis, outputType))
        continue;
      if ((uint64_t)axis == outputDimIndex) {
        inputDimIndex = i;
        found = true;
        break;
      }
    }
    if (found)
      shapeInput = onnxOp.getShape();
  } else if (auto onnxOp = dyn_cast<ONNXConstantOfShapeOp>(op)) {
    // `input` stores shape information.
    shapeInput = onnxOp.getInput();
  } else if (auto onnxOp = dyn_cast<ONNXExpandOp>(op)) {
    // `shape` stores shape information.
    shapeInput = onnxOp.getShape();
  } else if (auto onnxOp = dyn_cast<ONNXMaxUnpoolOp>(op)) {
    // Optional `output_shape` stores shape information.
    if (!isNoneValue(onnxOp.getOutputShape()))
      shapeInput = onnxOp.getOutputShape();
  } else if (auto onnxOp = dyn_cast<ONNXReshapeOp>(op)) {
    // `shape` stores shape information. Only support `allow_zero == 0`.
    if (onnxOp.getAllowzero() == 0)
      shapeInput = onnxOp.getShape();
  } else if (auto onnxOp = dyn_cast<ONNXTileOp>(op)) {
    // If input dimension i is 1, `repeats` i stores shape information.
    Type inputType = onnxOp.getInput().getType();
    ArrayRef<int64_t> inputShape = getShape(inputType);
    if (inputShape[inputDimIndex] == 1)
      shapeInput = onnxOp.getRepeats();
  }
  if (!shapeInput)
    return false;

  // If it is not from Concat (e.g. shape operations are not simplified to
  // Concat), we would not have enough information.
  if (!areDimsFromConcat(shapeInput))
    return false;

  SmallVector<Value, 4> dims;
  getDims(shapeInput, dims);
  if (auto d =
          insertDimWhenUseful(dims[inputDimIndex], inputDimIndex, sameDims))
    LLVM_DEBUG(llvm::dbgs() << "  - Added a new dim(" << d.value().first << ", "
                            << d.value().second << ")\n");
  return true;
}

//===----------------------------------------------------------------------===//
// DimAnalysis class.
//===----------------------------------------------------------------------===//

DimAnalysis::DimAnalysis(ArrayRef<Value> vals) {
  for (Value val : vals)
    if (!isNoneValue(val))
      build(val);

  LLVM_DEBUG(llvm::dbgs() << "The number of dynamic dims in the IR: "
                          << numOfDynamicDims << "\n");
}

DimAnalysis::DimAnalysis(ModuleOp moduleOp) {
  moduleOp.walk([&](Operation *op) {
    for (Value output : op->getResults())
      build(output);
  });
  LLVM_DEBUG(llvm::dbgs() << "The number of dynamic dims in the IR: "
                          << numOfDynamicDims << "\n");
}

void DimAnalysis::build(Value val) {
  if (auto tensorType = val.getType().dyn_cast<RankedTensorType>()) {
    for (unsigned i = 0; i < tensorType.getRank(); ++i) {
      // Only care about dynamic dimensions.
      if (tensorType.isDynamicDim(i)) {
        DimT ti(val, i);
        DimSetT dimSet;
        dimSet.insert(ti);
        dimSetMap[setCounter++] = dimSet;
        numOfDynamicDims++;
        LLVM_DEBUG(llvm::dbgs() << "Build a new dim(" << ti.first << ", "
                                << ti.second << ")\n");
      }
    }
  }
}

bool DimAnalysis::sameDim(
    Value tensor1, int64_t dimAxis1, Value tensor2, int64_t dimAxis2) const {
  // Handle negative axis and test if in bound.
  ShapedType tensor1Type = tensor1.getType().cast<ShapedType>();
  ShapedType tensor2Type = tensor2.getType().cast<ShapedType>();
  if (!handleAndTestInBound(dimAxis1, tensor1Type) ||
      !handleAndTestInBound(dimAxis2, tensor2Type))
    return false;
  // Same tensor, same axis.
  if ((tensor1 == tensor2) && (dimAxis1 == dimAxis2))
    return true;
  int64_t dim1 = tensor1Type.getShape()[(uint64_t)dimAxis1];
  int64_t dim2 = tensor2Type.getShape()[(uint64_t)dimAxis2];
  // Both dims are static.
  if (!ShapedType::isDynamic(dim1) && !ShapedType::isDynamic(dim2))
    return (dim1 == dim2);
  // One is static, other is dynamic.
  if (dim1 != dim2)
    return false;
  // Both are dynamic.
  return sameDynDim(tensor1, dimAxis1, tensor2, dimAxis2);
}

bool DimAnalysis::sameDynDim(
    Value tensor1, int64_t dimAxis1, Value tensor2, int64_t dimAxis2) const {
  // Handle negative axis and test if in bound.
  ShapedType tensor1Type = tensor1.getType().cast<ShapedType>();
  ShapedType tensor2Type = tensor2.getType().cast<ShapedType>();
  if (!handleAndTestInBound(dimAxis1, tensor1Type) ||
      !handleAndTestInBound(dimAxis2, tensor2Type))
    return false;
  // Same tensor, same axis.
  if ((tensor1 == tensor2) && (dimAxis1 == dimAxis2))
    return true;
  DimT dim1(tensor1, (uint64_t)dimAxis1);
  DimT dim2(tensor2, (uint64_t)dimAxis2);
  // Two dims are the same if they are in the same set.
  for (auto &entry : dimSetMap) {
    DimSetT dims = entry.second;
    if (dims.contains(dim1) && dims.contains(dim2))
      return true;
  }
  return false;
}

bool DimAnalysis::sameShape(Value tensor1, Value tensor2) const {
  if (!sameRank(tensor1, tensor2))
    return false;
  unsigned rank = tensor1.getType().cast<ShapedType>().getRank();
  // Check each dimension.
  for (unsigned i = 0; i < rank; ++i) {
    if (!sameDim(tensor1, i, tensor2, i))
      return false;
  }
  return true;
}

bool DimAnalysis::sameDynShape(Value tensor1, Value tensor2) const {
  if (!sameRank(tensor1, tensor2))
    return false;
  ArrayRef<int64_t> shape1 = tensor1.getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> shape2 = tensor2.getType().cast<ShapedType>().getShape();
  // Check each dimension.
  for (unsigned i = 0; i < shape1.size(); ++i) {
    int64_t dim1 = shape1[i];
    int64_t dim2 = shape2[i];
    // Both dims are static, ignore this case. Only care about dynamic dims.
    if (!ShapedType::isDynamic(dim1) && !ShapedType::isDynamic(dim2))
      continue;
    // At least one is dynamic.
    if (!sameDim(tensor1, i, tensor2, i))
      return false;
  }
  return true;
}

bool DimAnalysis::broadcastLastDim(Value tensor1, Value tensor2) const {
  if (!sameRank(tensor1, tensor2))
    return false;
  ArrayRef<int64_t> shape1 = tensor1.getType().cast<ShapedType>().getShape();
  unsigned rank = shape1.size();
  // The last dimension of tensor1 must be 1, so that tensor1 is broadcasting
  // to tensor2.
  if (shape1[rank - 1] != 1)
    return false;
  // Other dimensions except the last one must be the same.
  for (unsigned i = 0; i < rank - 1; ++i) {
    if (!sameDim(tensor1, i, tensor2, i))
      return false;
  }
  return true;
}

void DimAnalysis::dump() const {
  llvm::outs() << numOfDynamicDims
               << " dynamic dimensions (not including block arguments) are "
                  "classified into "
               << dimSetMap.size() << " sets.\n";
  for (auto &entry : dimSetMap) {
    uint64_t i = entry.first;
    DimSetT dimSet = entry.second;
    llvm::outs() << "\n- Set " << i << " (size: " << dimSet.size() << "):\n";
    for (auto &ti : dimSet)
      llvm::outs() << "  - Dim " << ti.second << " of " << ti.first << "\n";
  }
}

void DimAnalysis::analyze() {
  // Build sets of the same dynamic dimensions and merge them until a fixed
  // point where there is no update on each set.
  bool continued = true;
  while (continued) {
    // Local search and update each set of dynamic dimensions.
    continued = updateDimSets();

    // Merge sets if there is update.
    if (continued)
      // Two sets with a common dimension will be merged into a single set
      // consisting of elements from each set.
      mergeDimSets();
  }

  LLVM_DEBUG(
      llvm::dbgs() << "\nThe number of sets of same dynamic dims in the IR: "
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
        // Found new dynamic dims.
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
  LLVM_DEBUG(
      llvm::dbgs() << "\nVisiting dim(" << tensor << ", " << dimIndex << ")\n");

  // Tensor is from a constant. Nothing to do further.
  if (isa<ONNXConstantOp>(op))
    return;

  // DimOp
  if (auto dimOp = dyn_cast<ONNXDimOp>(op)) {
    DimAnalysis::DimT newSameDim(dimOp.getData(), dimOp.getAxis());
    sameDims.insert(newSameDim);
    return;
  }

  // CastOp
  if (auto castOp = dyn_cast<ONNXCastOp>(op)) {
    DimAnalysis::DimT newSameDim(castOp.getInput(), dimIndex);
    sameDims.insert(newSameDim);
    return;
  }

  // All dimensions in the analysis must be dynamic. If not, something really
  // wrong happened.
  ShapedType ty = tensor.getType().cast<ShapedType>();
  assert(ty.isDynamicDim(dimIndex) && "There is a static dim in the analysis. "
                                      "Something really wrong happened.");

  ////////////////////////////////////////////////////
  // Using ShapeHelper to find out where the output dim comes from.
  exploreSameDimsUsingShapeHelper(dim, op, sameDims);

  ////////////////////////////////////////////////////
  // For operations that have an input specifying the output shape, the output
  // dim can come from the shape input.
  // For example: ConstantOfShape, Expand, Reshape.
  exploreSameDimsUsingShapeInput(dim, op, sameDims);

  ////////////////////////////////////////////////////
  // Special/additional cases.

  // Variadic/non-variadic BinaryOp (alphabetical list, same as under
  // ONNXBroadcastOpShapeHelper).
  // MatMul uses the same broadcasting rule as BinaryOp, include it here. If we
  // know by this analysis that two dynamic dims at the same index (counting
  // from the innermost dim back) are the same, then the output dim must be the
  // same too.
  if (isa<ONNXAddOp, ONNXAndOp, ONNXBitShiftOp, ONNXBitwiseAndOp,
          ONNXBitwiseOrOp, ONNXBitwiseXorOp, ONNXBitShiftOp, ONNXDivOp,
          ONNXEqualOp, ONNXGreaterOp, ONNXGreaterOrEqualOp, ONNXLessOp,
          ONNXLessOrEqualOp, ONNXMatMulOp, ONNXMeanOp, ONNXMinOp, ONNXModOp,
          ONNXMulOp, ONNXOrOp, ONNXPowOp, ONNXSubOp, ONNXSumOp, ONNXWhereOp,
          ONNXXorOp>(op)) {
    LLVM_DEBUG(llvm::dbgs() << "Explore same inputs\n");
    OperandRange operands = op->getOperands();
    for (size_t i = 0; i < operands.size() - 1; ++i) {
      Value A = op->getOperands()[i];
      for (size_t k = i + 1; k < operands.size(); ++k) {
        Value B = op->getOperands()[k];
        uint64_t aRank = getRank(A.getType());
        uint64_t bRank = getRank(B.getType());
        if ((aRank != 0) && (bRank != 0)) {
          // aDim == bDim (dynamic), there is no broadcasting and aDim ==
          // outputDim.
          uint64_t maxRank = std::max(aRank, bRank);
          int64_t negativeIndex = dimIndex - maxRank;
          if (sameDim(A, negativeIndex, B, negativeIndex)) {
            if (auto d =
                    insertDimWhenUseful(A, aRank + negativeIndex, sameDims))
              LLVM_DEBUG(llvm::dbgs()
                         << "  - Added a new dim(" << d.value().first << ", "
                         << d.value().second << ")\n");
          }
        }
      }
    }
  }

  // ReshapeOp has some additional cases.
  if (auto reshapeOp = dyn_cast<ONNXReshapeOp>(op)) {
    if (reshapeOp.getAllowzero() != 0)
      return;

    LLVM_DEBUG(llvm::dbgs() << "Special case for Reshape\n");

    Value data = reshapeOp.getData();
    Value output = reshapeOp.getReshaped();

    // Special case 1: the output dimension i can be from
    // - data[j] if
    //    - dim j and i are the only dynamic dimension in data and output,
    //    respectively, and
    //    - the products of the remaining static dimensions in data and output
    //    are equal.
    //
    // It's interesting that if shape[i] is arg0[ii] and data[j] is arg1[jj],
    // we can say that arg0[ii] == arg1[jj], that can be used to verify user
    // inputs.

    // Get the dynamic dimension from data.
    auto dataType = cast<RankedTensorType>(data.getType());
    auto outputType = cast<RankedTensorType>(output.getType());
    // Check if there is only one dynamic dimension in the data and output.
    bool dataHasOneDynamicDim =
        (llvm::count(dataType.getShape(), ShapedType::kDynamic) == 1);
    bool outputHasOneDynamicDim =
        (llvm::count(outputType.getShape(), ShapedType::kDynamic) == 1);
    // Check if the products of static sizes in the data and output are equal.
    // It's ok to count ShapedType::kDynamic (dynamic dimension) in the size.
    int64_t dataStaticSize = 1, outputStaticSize = 1;
    for (int64_t i = 0; i < dataType.getRank(); ++i)
      dataStaticSize *= dataType.getShape()[i];
    for (int64_t i = 0; i < outputType.getRank(); ++i)
      outputStaticSize *= outputType.getShape()[i];
    // Conditions hold, the dynamic dimension can be from the data.
    if (dataHasOneDynamicDim && outputHasOneDynamicDim &&
        (dataStaticSize == outputStaticSize)) {
      // Find the index of the dynamic dimension in the data.
      std::optional<int64_t> dynamicDimIndexInData = std::nullopt;
      for (int64_t i = 0; i < dataType.getRank(); ++i)
        if (dataType.isDynamicDim(i)) {
          dynamicDimIndexInData = i;
          break;
        }
      assert(dynamicDimIndexInData.has_value() &&
             "Failed to obtain the index of the dynamic dimension in the data");
      if (auto d = insertDimWhenUseful(
              reshapeOp.getData(), *dynamicDimIndexInData, sameDims))
        LLVM_DEBUG(llvm::dbgs()
                   << "  - Case 1: Added a new dim(" << d.value().first << ", "
                   << d.value().second << ")\n");
    }

    // Special case 2: input and output have the same rank of 2, if one output
    // dim is from an input dim, the other output dim must be from the remaining
    // input dim.
    //
    // clang-format off
    // ```mlir
    // %cst_minus1 = onnx.Constant dense<-1> : tensor<1xi64>
    // %0 = "onnx.Dim"(%arg0) {axis = 1 : si64} : (tensor<?x?xi64>) -> tensor<1xi64>
    // %1 = "onnx.Concat"(%cst_minus1, %0) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    // %2 = "onnx.Reshape"(%arg0, %1) {allowzero = 0 : si64} : (tensor<?x?xi64>, tensor<2xi64>) -> tensor<?x?xi64>
    // ```
    // clang-format on
    int64_t dataRank = dataType.getRank();
    int64_t outputRank = outputType.getRank();
    if ((dataRank == 2) && (outputRank == 2)) {
      // Find if the output dim is from an input dim.
      int64_t iDim = -1;
      for (int64_t i = 0; i < dataRank; ++i) {
        if (sameDynDim(data, i, output, 1 - dimIndex)) {
          iDim = i;
          // The other output dim must be the same as the other input dim.
          if (auto d = insertDimWhenUseful(data, 1 - iDim, sameDims))
            LLVM_DEBUG(llvm::dbgs()
                       << "  - Case 2: Added a new dim(" << d.value().first
                       << ", " << d.value().second << ")\n");
        }
        break;
      }
    }
  }
}

} // namespace onnx_mlir

namespace {

/// This pass is for testing purpose. It introduces onnx.DimGroup to the IR to
/// show group IDs. Dynamic dimensions with the same group ID are supposed to be
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

  using namespace onnx_mlir;

  DimAnalysis testOp(moduleOp);
  testOp.analyze();
  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    testOp.dump();
  });

  // Add onnx.DimGroup into the IR for LIT tests.
  DimAnalysis::DimSetMapT mapping = testOp.getGroupingResult();
  DimAnalysis::DimSetT processed;
  for (auto &entry : mapping) {
    uint64_t groupID = entry.first;
    DimAnalysis::DimSetT dimSet = entry.second;
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
          val = dimOp.getData();
      }
      DimAnalysis::DimT dim(val, dimAxis);
      // Ignore if a DimGroup was created for it.
      if (processed.contains(dim))
        continue;
      MultiDialectBuilder<OnnxBuilder> create(b, loc);
      create.onnx.dimGroup(val, dimAxis, groupID);
      processed.insert(dim);
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
