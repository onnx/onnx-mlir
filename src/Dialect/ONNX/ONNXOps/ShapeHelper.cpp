/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------ONNXShapeHelper.cpp - help for shapes----------------=== //
//
// Copyright 2020-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file has the computations to compute the shapes using the new index expr
// approach.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/BitVector.h"
#include "llvm/Support/Debug.h"

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

#include <algorithm>

#define DEBUG_TYPE "shape-helper"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Support functions
//===----------------------------------------------------------------------===//

// Check if axis is in [-rank, rank), or [-rank, rank] when includeRank is true.
// Return false when not in range; set axis to positive value when in range.
bool isAxisInRange(int64_t &axis, int64_t rank, bool includeRank) {
  int64_t ub = includeRank ? rank + 1 : rank;
  if (axis < -rank || axis >= ub)
    return false;
  if (axis < 0)
    axis += rank;
  return true;
}

bool isAxisInRange(int64_t &axis, Value val, bool includeRank) {
  ShapedType shapedType = mlir::cast<ShapedType>(val.getType());
  assert(shapedType && "expected a shaped type to determine the rank for axis");
  return isAxisInRange(axis, shapedType.getRank(), includeRank);
}

// Check if axis is in [-rank, rank), or [-rank, rank] when includeRank is
// true.  Assert when not in range. Return positive axis.
int64_t getAxisInRange(int64_t axis, int64_t rank, bool includeRank) {
  assert(isAxisInRange(axis, rank, includeRank) && "expected axis in range");
  return axis;
}

int64_t getAxisInRange(int64_t axis, Value val, bool includeRank) {
  assert(isAxisInRange(axis, val, includeRank) && "expected axis in range");
  return axis;
}

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper
//===----------------------------------------------------------------------===//

/// Refine `inferredDims` using the output's shape if possible. For example,
/// replacing a dynamic dim in `inferredDims` by a static dim in the output's
/// shape.
static void refineDims(Operation *op, DimsExpr &inferredDims, Value output) {
  // Nothing to do if the output is unranked.
  if (!isRankedShapedType(output.getType()))
    return;

  llvm::ArrayRef<int64_t> existingDims = getShape(output.getType());
  // Do not handle the case of scalar tensor whose type can be tensor<f32>
  // or tensor<1xf32>. Just use the inferredShape in this case.
  if (existingDims.size() < 1 || inferredDims.size() < 1)
    return;

  assert((existingDims.size() == inferredDims.size()) &&
         "Inferred shape and existing shape are inconsistent in the number "
         "of elements");

  // Try to update inferredDim if existingDim is static.
  for (unsigned i = 0; i < existingDims.size(); ++i) {
    // Safety checks for old convention of using -1 for dynamic.
    assert(existingDims[i] != -1 && "dynamic use kDynamic now");
    if (inferredDims[i].isLiteral()) {
      // Index expressions should not use the ShapedType::kDynamic ever to
      // signal dynamic shape. Questionmarks are used for that.
      assert(inferredDims[i].getLiteral() != -1 && "dynamic use questionmark");
      assert(inferredDims[i].getLiteral() != ShapedType::kDynamic &&
             "dynamic use questionmark");
    }
    // ExistingDim is dynamic, nothing to learn from.
    if (existingDims[i] == ShapedType::kDynamic)
      continue;

    // InferredDim is unknown at shape inference: update it.
    if (inferredDims[i].isQuestionmark()) {
      inferredDims[i] = LitIE(existingDims[i]);
      continue;
    }
    // inferredDim is unknown at lowering: use existing dim for efficiency.
    if (!inferredDims[i].isLiteral()) {
      inferredDims[i] = LitIE(existingDims[i]);
      continue;
    }
    // inferredDim is different from existingDim. Believe in existingDim.
    assert(inferredDims[i].isLiteral() && "isLiteral failed");
    if (existingDims[i] != inferredDims[i].getLiteral()) {
      if (op) {
        llvm::outs() << "\nWarning for operation " << op->getName()
                     << ": [Shape inference, dim " << i
                     << "] the inferred dim (" << inferredDims[i].getLiteral()
                     << ") is different from the existing dim ("
                     << existingDims[i]
                     << "). Use the existing dim instead.\n\n";
      } else {
        llvm::outs() << "\nWarning: [Shape inference, dim " << i
                     << "] the inferred dim (" << inferredDims[i].getLiteral()
                     << ") is different from the existing dim ("
                     << existingDims[i]
                     << "). Use the existing dim instead.\n\n";
      }
      inferredDims[i] = LitIE(existingDims[i]);
    }
  }
}

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper::ONNXOpShapeHelper(Operation *inputOp,
    ValueRange inputOperands, IndexExprBuilder *inputIeBuilder,
    IndexExprScope *inputScope)
    : op(inputOp), operands(inputOperands), createIE(inputIeBuilder),
      scope(inputScope), privateOutputsDims(), ownScope(inputScope == nullptr),
      ownBuilder(inputIeBuilder == nullptr) {
  assert(op && "Expecting a valid operation pointer");
  if (ownBuilder) {
    createIE = new IndexExprBuilderForAnalysis(op->getLoc());
    assert(createIE && "failed to create a new builder");
  }
  if (ownScope) {
    scope = new IndexExprScope(createIE->getBuilderPtr(), createIE->getLoc());
    assert(scope && "failed to create a new scope");
  }
  privateOutputsDims.resize(op->getNumResults());
  // When we have no inputOperands, get them from the operation.
  if (inputOperands.size() == 0) {
    // A operand cache is used here, as we don't want to rely on the live range
    // of the passed parameter. Possibly a more elegant solution can be used, I
    // could not find one at this time.
    privateOperandsCache = llvm::SmallVector<Value, 4>(
        op->getOperands().begin(), op->getOperands().end());
    operands = ValueRange(privateOperandsCache);
  }
}

ONNXOpShapeHelper::~ONNXOpShapeHelper() {
  if (ownScope)
    delete scope;
  if (ownBuilder)
    delete createIE;
}

void ONNXOpShapeHelper::computeShapeAndAssertOnFailure() {
  // Invoke virtual compute shape.
  LogicalResult res = computeShape();
  assert(succeeded(res) && "Failed to compute shape");
}

void ONNXOpShapeHelper::setOutputDims(
    const DimsExpr &inferredDims, int n, bool refineShape) {
  privateOutputsDims[n] = inferredDims;
  if (refineShape) {
    Value output = getOutput(n);
    refineDims(op, privateOutputsDims[n], output);
  }
}

LogicalResult ONNXOpShapeHelper::setOutputDimsFromOperand(
    Value operand, int n, bool refineShape) {
  // Output and operand have the same shape. Just pass the operand shape to the
  // output.
  DimsExpr outputDims;
  createIE->getShapeAsDims(operand, outputDims);
  setOutputDims(outputDims, n, refineShape);
  return success();
}

LogicalResult ONNXOpShapeHelper::setOutputDimsFromLiterals(
    SmallVector<int64_t, 4> shape, int n, bool refineShape) {
  // Output has the shape given by the vector of integer numbers. Number
  // ShapedType::kDynamic is transformed into a questionmark.
  DimsExpr outputDims;
  getIndexExprListFromShape(shape, outputDims);
  setOutputDims(outputDims, n, refineShape);
  return success();
}

LogicalResult ONNXOpShapeHelper::setOutputDimsFromTypeWithConstantShape(
    Type type, int n, bool refineShape) {
  RankedTensorType rankedType = mlir::dyn_cast<RankedTensorType>(type);
  if (!rankedType)
    return failure();
  DimsExpr outputDims;
  getIndexExprListFromShape(rankedType.getShape(), outputDims);
  if (!IndexExpr::isNonNegativeLiteral(outputDims))
    return failure();
  setOutputDims(outputDims, n, refineShape);
  return success();
}

// Reuse the same type for each of the outputs.
LogicalResult ONNXOpShapeHelper::computeShapeAndUpdateType(
    Type elementType, Attribute encoding) {
  // Invoke virtual compute shape.
  if (failed(computeShape()))
    return op->emitError("Failed to scan parameters successfully");
  assert((mlir::isa<VectorType>(elementType) ||
             !mlir::isa<ShapedType>(elementType)) &&
         "element type cannot be a shaped type other than vector type");
  uint64_t resNum = op->getNumResults();
  for (uint64_t i = 0; i < resNum; ++i) {
    // If we have an optional type, leave it as is.
    if (mlir::isa<NoneType>(op->getResults()[i].getType()))
      continue;
    llvm::SmallVector<int64_t, 4> shapeVect;
    IndexExpr::getShape(getOutputDims(i), shapeVect);
    // Set refineShape to false here because we refine it (or not) when setting
    // the output shape. So there is no need to perform this again here.
    updateType(op, op->getResults()[i], shapeVect, elementType, encoding,
        /*refineShape*/ false);
  }
  return success();
}

// Use a distinct type for each of the output.
LogicalResult ONNXOpShapeHelper::computeShapeAndUpdateTypes(
    TypeRange elementTypeRange, ArrayRef<Attribute> encodingList) {
  uint64_t resNum = op->getNumResults();
  assert((elementTypeRange.size() == resNum) &&
         "Incorrect number of elementTypes");
  bool hasEncoding = encodingList.size() > 0;
  assert((!hasEncoding || encodingList.size() == resNum) &&
         "Incorrect number of encoding");
  // Invoke virtual compute.
  if (failed(computeShape()))
    return op->emitError("Failed to scan " + op->getName().getStringRef() +
                         " parameters successfully");
  for (uint64_t i = 0; i < resNum; ++i) {
    // If we have an optional type, leave it as is.
    if (mlir::isa<NoneType>(op->getResults()[i].getType()))
      continue;
    llvm::SmallVector<int64_t, 4> shapeVect;
    IndexExpr::getShape(getOutputDims(i), shapeVect);
    Type currElementType = elementTypeRange[i];
    // Set refineShape to false here because we refine it (or not) when setting
    // the output shape. So there is no need to perform this again here.
    updateType(op, op->getResults()[i], shapeVect, currElementType,
        hasEncoding ? encodingList[i] : nullptr, /*refineShape*/ false);
  }
  return success();
}

void ONNXOpShapeHelper::setOperands(ValueRange inputs) {
  // Note: do not use operands until it is re-assigned
  privateOperandsCache =
      llvm::SmallVector<Value, 4>(inputs.begin(), inputs.end());
  operands = ValueRange(privateOperandsCache);
}

//===----------------------------------------------------------------------===//
// ONNX Broadcast Op Shape Helper
//===----------------------------------------------------------------------===//

LogicalResult ONNXBroadcastOpShapeHelper::customComputeShape(
    ValueRange initialOperands, DimsExpr *additionalOperand) {
  // if additionalOperand is not used, we expect a zero-sized vector.
  // A temporary IndexExpr vector for the output.
  DimsExpr dimsExpr;
  uint64_t numOfInputs = initialOperands.size();

  if (!llvm::all_of(initialOperands,
          [](Value initalOperand) { return hasShapeAndRank(initalOperand); })) {
    return failure();
  }

  // Compute rank of the output. Rank of the output is the maximum rank of all
  // initial operands.
  uint64_t additionalOperRank =
      additionalOperand ? additionalOperand->size() : 0;
  outputRank = additionalOperRank;
  for (uint64_t i = 0; i < numOfInputs; ++i)
    outputRank =
        std::max(outputRank, createIE->getShapedTypeRank(initialOperands[i]));
  dimsExpr.resize(outputRank);

  // Prepare dims for every input. Prepend 1s if the input's shape has smaller
  // rank, so that all the shapes have the same rank.
  LiteralIndexExpr one(1);
  for (uint64_t i = 0; i < numOfInputs; ++i) {
    uint64_t r = createIE->getShapedTypeRank(initialOperands[i]);
    // Prepend 1s.
    DimsExpr dims(outputRank - r, one);
    // Get from the input.
    for (uint64_t k = 0; k < r; ++k)
      dims.emplace_back(createIE->getShapeAsDim(initialOperands[i], k));
    inputsDims.emplace_back(dims);
  }

  // Handle the additional operand here.
  if (additionalOperRank > 0) {
    DimsExpr dims(outputRank - additionalOperRank, one);
    for (uint64_t k = 0; k < additionalOperRank; ++k)
      dims.emplace_back((*additionalOperand)[k]);
    inputsDims.emplace_back(dims);
    numOfInputs++;
  }

  // Initialize the output with the first operand.
  dimsExpr = inputsDims[0];

  // Note on IndexExpr. When we are not allowed to generate code, QuestionMark
  // stands for anything but a literal. When we are allowed to generate code,
  // there should be no more QuestionMarks as we are allowed to generate
  // affine/symbols/dims/non-affine expressions. Since this code predominantly
  // runs when we can gen code (as it actually does gen max ops), we should
  // use !isLiteral() for anything that is runtime. The comments were left
  // unchanged.

  //  Now compute each broadcasted dimension for the output. Folding over the
  //  other operands along the current dimension index.
  for (uint64_t i = 1; i < numOfInputs; ++i) {
    for (uint64_t j = 0; j < outputRank; ++j) {
      // Set the output dimension based on the two dimension values.
      // Dimension value can be one of 1, QuestionMark, LiteralNot1.
      IndexExpr currentDimExpr = dimsExpr[j];
      IndexExpr nextDimExpr = inputsDims[i][j];
      // Case: 1 - *.
      if (currentDimExpr.isLiteralAndIdenticalTo(1)) {
        if (!hasUniBroadcasting) {
          dimsExpr[j] = nextDimExpr;
        }
        continue;
      }
      // Case: LiteralNot1 - *.
      if (currentDimExpr.isLiteralAndDifferentThan(1)) {
        // LiteralNot1 - LiteralNot1 => keep unchanged with verifying.
        if (nextDimExpr.isLiteralAndDifferentThan(1) &&
            !currentDimExpr.isLiteralAndIdenticalTo(nextDimExpr))
          return op->emitOpError("Incompatible broadcast matching " +
                                 std::to_string(currentDimExpr.getLiteral()) +
                                 " with " +
                                 std::to_string(nextDimExpr.getLiteral()));
        // Case: LiteralNot1 - (QuestionMark or 1) => Keep unchanged without
        // verifying.
        continue;
      }
      // Case: QuestionMark - 1 => keep unchanged.
      if (!currentDimExpr.isLiteral() &&
          nextDimExpr.isLiteralAndIdenticalTo(1)) {
        continue;
      }
      // Case QuestionMark - LiteralNot1 => set to LiteralNot1 without
      // verifying.
      if (!currentDimExpr.isLiteral() &&
          nextDimExpr.isLiteralAndDifferentThan(1)) {
        dimsExpr[j] = nextDimExpr;
        continue;
      }
      // Case: QuestionMark - QuestionMark
      if (!hasUniBroadcasting) {
        dimsExpr[j] = IndexExpr::max(currentDimExpr, nextDimExpr);
      }
    }
  }
  // Set the final output.
  setOutputDims(dimsExpr);
  return success();
}

// Attempt to rule out broadcasting at compile time, using dim analysis when
// available (i.e. nonnull). Must be called after computeShape.
//
// Note that broadcasting handles tensors of different ranks by prepending `1x`
// to the shorter input shapes. When inputs `1x1x5xf32` and `5xf32` are analyzed
// for broadcasting patterns, the shorter `5xf32` is first expanded to
// `1x1x5xf32` before being compared to the other inputs. Comparing `1x1x5xf32`
// with `1x1x5xf32` determines that there is no broadcast; thus this call will
// return false in such situation. This make practical senses too as no values
// of either input will be used more than once with the value of the other
// input.
//
bool ONNXBroadcastOpShapeHelper::hasNoBroadcast(DimAnalysis *dimAnalysis) {
  // First use static analysis to rule out broadcast. If we cannot rule out
  // broadcasting for any reasons, hasNoBroadcast is set to false.
  bool hasNoBroadcast = true;
  for (uint64_t r = 0; r < outputRank && hasNoBroadcast; ++r) {
    bool hasOne, hasOtherThanOne;
    hasOne = hasOtherThanOne = false;
    for (DimsExpr dims : inputsDims) {
      if (!dims[r].isLiteral()) {
        // Has dynamic values.. possible broadcast, assume the worst.
        hasNoBroadcast = false;
        break;
      }
      int64_t lit = dims[r].getLiteral();
      if (lit == 1)
        hasOne = true;
      else
        hasOtherThanOne = true;
    }
    if (hasOne && hasOtherThanOne)
      // Has a known broadcast situation. No need for further analysis,
      // broadcasting has been detected.
      return false;
  }

  // Using the most conservative analysis, we did not detect any broadcasting,
  // we are good.
  if (hasNoBroadcast)
    return true;

  // We have dynamic dimensions that prevented us to rule out broadcasting, try
  // the more expensive dimAnalysis approach now, if available.
  if (!dimAnalysis)
    return false;
  // In some cases, we can have more inputDims than operands (custom broadcast
  // operators, e.g. ONNXExtendOp). Dismiss such cases as we need here the
  // values of each of the inputs.
  int64_t inputNum = operands.size();
  if ((int64_t)inputsDims.size() != inputNum)
    return false;
  // Check if we can prove that each operand has the same shape.
  for (int i = 1; i < inputNum; ++i)
    if (!dimAnalysis->sameShape(operands[0], operands[i]))
      return false;
  // All have the same shape.
  return true;
}

// Checks if the input operands need rank broadcasting.
bool ONNXBroadcastOpShapeHelper::hasRankBroadcast() {
  ValueRange operands = this->operands;
  for (Value operand : operands) {
    auto operandType = mlir::cast<ShapedType>(operand.getType());
    if (outputRank != (uint64_t)operandType.getRank())
      return true;
  }
  return false;
}

bool ONNXBroadcastOpShapeHelper::hasManageableBroadcastForInnerDims(
    int64_t &collapsedInnermostLoops, int64_t &collapsedLiteralSize,
    IndexExpr &collapsedDynamicSize, DimAnalysis *dimAnalysis) {
  int64_t dimNum = inputsDims.size();
  bool canUseDimAnalysis = dimAnalysis && (int64_t)operands.size() == dimNum;
  LLVM_DEBUG(llvm::dbgs() << "has manageable broadcast with"
                          << (canUseDimAnalysis ? "" : "out")
                          << " dim analysis\n");
  // Keep track of cumulative inner dim sizes.
  collapsedLiteralSize = 1;
  collapsedDynamicSize = LitIE(1);
  // Keep track of ones, scalar, and broadcast per input.
  llvm::SmallBitVector isOne(dimNum, true);
  llvm::SmallBitVector isScalar(dimNum, true);
  llvm::SmallBitVector hasBroadcast(dimNum, false);

  // Walk through the rank from innermost to outermost, using neg values.
  int64_t outputRankInt = outputRank;
  collapsedInnermostLoops = 0;
  for (int64_t r = -1; r >= -outputRankInt; --r) {
    // Analyze if we have manageable broadcast at rank r.
    int64_t rr =
        r + outputRankInt; // Use for inputDims, they are padded to outputRank.
    LLVM_DEBUG(llvm::dbgs() << "iter r " << r << ", rr " << rr << "\n");

    // 1) Iterate through all the inputs and survey which inputs have 1s at
    // rank r and which inputs continue to be fully scalar. If we detect an
    // input that was broadcasted and that just stopped being a scalar, then
    // this is a broadcast that cannot be managed.
    int64_t nonScalarID = -1; // Id of one non-scalar; -1 if none.
    int64_t numOfOnes = 0;
    for (int64_t d = 0; d < dimNum; ++d) {
      // Test if this input d has a 1 at position rr.
      isOne[d] = inputsDims[d][rr].isLiteralAndIdenticalTo(1);
      if (isOne[d]) {
        numOfOnes++;
      } else {
        LLVM_DEBUG({
          if (isScalar[d])
            llvm::dbgs() << "  lost scalar: " << d << "\n";
        });
        isScalar[d] = false;
        nonScalarID = d; // Keep the id of one non-scalar input.
      }
      // Test if we lost a scalar that was being broadcasted.
      if (!isScalar[d] && hasBroadcast[d]) {
        // We lost a scalar that was being broadcasted. We cannot let that
        // happen, thus we stop with the previous iteration of r.
        // Case 4x1 and 4x3: first was broadcast at r==-1, not scalar at
        // r==-2.
        LLVM_DEBUG(llvm::dbgs()
                   << "  lost scalar with broadcast: " << d << "; abort\n");
        return collapsedInnermostLoops > 0;
      }
    }

    // 2) When we only have 1s, there is no broadcast in this iteration r, and
    // we don't have to update the sizes as this r dim's contribution is *1,
    // just skip. Catches 1x3 and 1x1: first was broadcast at r==-1, no
    // broadcast at r==-2 as they are all 1s.
    //
    // In addition, we don't update collapsedInnermostLoops to this iter, as
    // we don't want to collapse leading dims that have only ones.
    // Case: 1x1x4x8 and 1x1x4x8. We can collapse r==-1 and r==-2, but there is
    // no need to collapse the 1 dimensions... it brings no advantages. So by
    // skipping the updating of collapsedInnermostLoops here, we will omit
    // these leading ones.

    // Revision: it is actually good to detects 1s everywhere as we can
    // collapse the loop and have less overhead.
#define REVISION_COLLAPSE_ALL_ONES 1
    bool allOnes = numOfOnes == dimNum;
    if (allOnes) {
#if REVISION_COLLAPSE_ALL_ONES
      // No need to update the sizes as dim is all ones.
      collapsedInnermostLoops = -r;
      LLVM_DEBUG(llvm::dbgs() << "  SUCCESS (all ones) at collapsing "
                              << collapsedInnermostLoops
                              << " inner loops with cumulative static size of "
                              << collapsedLiteralSize << "\n\n");

#else
      LLVM_DEBUG(llvm::dbgs() << "  all ones, done\n");
#endif
      continue;
    }

    // 3) If we have 2 or more non scalars, test that they are compatible.
    int64_t nonScalarNum = dimNum - numOfOnes;
    assert(nonScalarNum > 0 && "eliminated the all one scenario");
    assert(nonScalarID != -1 && "eliminated the all one scenario");
    if (nonScalarNum >= 2) {
      LLVM_DEBUG(llvm::dbgs() << "  check non-scalar compatibility\n");
      // For all non scalars...
      for (int64_t d = 0; d < dimNum; ++d) {
        // Consider only dims d that are not scalar, and skip d ==
        // nonScalarID.
        if (isOne[d] || d == nonScalarID)
          continue;
        // Compare nonScalarID with d
        if (inputsDims[nonScalarID][rr].isLiteral() &&
            inputsDims[d][rr].isLiteral()) {
          // Both literal, do a literal check.
          if (inputsDims[nonScalarID][rr].getLiteral() ==
              inputsDims[d][rr].getLiteral()) {
            // Same literal dims, nonScalarID and d are compatible.
            // Continue to the next non-scalar.
            LLVM_DEBUG(llvm::dbgs() << "    literal compatibility "
                                    << nonScalarID << " & " << d << "\n");
            continue;
          }
          // Different literal dims, nonScalarID and d are NOT compatible.
          // Abort at this rank r; thus stops at previous iteration of r.
          LLVM_DEBUG(llvm::dbgs() << "    literal incompatibility "
                                  << nonScalarID << " & " << d << "; abort\n");
          return collapsedInnermostLoops > 0;
        }
        // We could not determine compatibility with literals, try deducing
        // info with dim analysis, if available.
        if (canUseDimAnalysis &&
            /* Use negative index convention here as operands may have fewer
               than outputRank dimensions */
            dimAnalysis->sameDim(operands[nonScalarID], r, operands[d], r)) {
          // Analysis demonstrated them to be the same, we are fine.
          // Continue to the next non-scalar input.
          LLVM_DEBUG(llvm::dbgs() << "    dyn compatibility " << nonScalarID
                                  << " & " << d << "\n");
          continue;
        }
        // Analysis could not prove operands's r dims to be identical.
        // Abort at this rank r; thus stops at previous iteration of r.
        LLVM_DEBUG(llvm::dbgs() << "    dyn incompatibility " << nonScalarID
                                << " & " << d << "; abort\n");
        return collapsedInnermostLoops > 0;
      } // End for all non-scalars,
    }   // End testing non-scalar compatibility.

    // 4) Since we have at least one non-scalar
    //   4.1) all the scalar inputs are now marked as having a broadcast.
    //   4.2) any inputs with a one that is not a scalar has a new broadcast,
    //        which is not allowed as only scalars can be broadcast to be
    //        manageable.
    for (int64_t d = 0; d < dimNum; ++d) {
      if (isScalar[d]) {
        // Case 1x1 and 2x1; the first is broadcast at r==-2 since it is a
        // scalar and there is one or more non-scalars.
        LLVM_DEBUG(llvm::dbgs() << "  broadcast for " << d << "\n");
        hasBroadcast[d] = true;
      } else if (isOne[d]) { // Is one but is not a scalar.
        // Case 1x4x1, 2x4x1, and 1x1x1: no broadcast at r==-1, broadcast at
        // r==-2 for last entry, no broadcast for the others. At r==-3,
        // continued broadcast for last entry, but first entry has new
        // broadcast to size 2 (i.e. isOne[0] is true, and isScalar[0] is
        // false). We cannot manage this. Abort at this rank r; thus stops at
        // previous iteration of r.
        LLVM_DEBUG(llvm::dbgs() << "  one and no scalar" << d << "; abort\n");
        return collapsedInnermostLoops > 0;
      }
    }

    // 5) This dim is fine for manageable broadcasting. Account for the
    // cumulative size of the inner dimensions.
    // If all scalar, then dim is 1, and there is nothing to do. Otherwise
    // accumulate in literal or dynamic sizes.
    if (inputsDims[nonScalarID][rr].isLiteral()) {
      collapsedLiteralSize *= inputsDims[nonScalarID][rr].getLiteral();
    } else {
      collapsedDynamicSize = collapsedDynamicSize * inputsDims[nonScalarID][rr];
    }
    collapsedInnermostLoops = -r;
    LLVM_DEBUG(llvm::dbgs()
               << "  SUCCESS at collapsing " << collapsedInnermostLoops
               << " inner loops with cumulative static size of "
               << collapsedLiteralSize << "\n\n");
  } // For rank r.

  // Came up to here, we are able to collapse them all.
  return collapsedInnermostLoops > 0;
}

LogicalResult ONNXBroadcastOpShapeHelper::getAccessExprs(Value operand,
    int64_t operandIndex, const SmallVectorImpl<IndexExpr> &loopAccessExprs,
    SmallVectorImpl<IndexExpr> &operandAccessExprs, bool flattenedInnerDims,
    bool hasNoBroadcast) {
  // Get info.
  int64_t loopDepth = loopAccessExprs.size();
  int64_t inputSize = inputsDims.size();
  int64_t operandRank = mlir::cast<ShapedType>(operand.getType()).getRank();
  // Flattened? no more than one loop per dim in output (aka output rank).
  // Not flattened? one loop per dim in output (aka output rank).
  if (flattenedInnerDims)
    assert(loopDepth <= (int64_t)outputRank &&
           "with flattening, expect no more than one loop iter variable per "
           "output rank");
  else
    assert(loopDepth == (int64_t)outputRank &&
           "without flattening, expect one loop iter variable per output rank");

  // Emtpy the access expr, just in case.
  operandAccessExprs.clear();
  // There is this case where we have no broadcast per se, but we have
  // mixtures of 1xTYPE vs TYPE scalars. Handle this case properly here.
  if (operandRank == 0)
    return success();

  // The hasNoBroadcast pattern can be established by shape inference using
  // DimAnalysis. If that is available, and broadcasting was ruled out, then
  // more efficient code can be generated.
  bool noBroadcasting =
      hasNoBroadcast || (hasUniBroadcasting && operandIndex == 0);

  for (int64_t r = 0; r < operandRank; ++r) {
    // Shape helper may pretend 1s, thus adjust dimension index accordingly.
    // Take loopDepth instead of outputRank as loopDepth reflect the
    // (possible) flattening of the loops.
    int64_t dimIndex = loopDepth - operandRank + r;
    SymbolIndexExpr operandDim(inputsDims[operandIndex][dimIndex]);

    bool useLoopIndexNoMatterWhat = false;
    if (noBroadcasting) {
      // Broadcasting is already ruled out, no need for further analysis
    } else {
      // If it turns out that all of the other input operands (and for this
      // dim index) have values 1, then we know we can use the loop variable
      // index without worry as either (1) it also has a dim of 1 (and thus
      // the loop variable index is 0) or (2) it has a dim of X (compile or
      // runtime) and this will be a broadcasted dimensions (and thus the loop
      // variable index can also safely be used).
      //
      // For example we have 5x?x3xf32 and 5x1x3xf32. We don't know at compile
      // time if the `?` will be 1 or (assuming without loss of generality) 10.
      // Normal code is to generate for `?` for that access dimension:
      //   select(dim==1 ? 0 : loop-var)
      // so that if this access needs to be broadcasted, we will only access the
      // `0` value of this broadcasted value; and if it is not broadcasted, we
      // will access each `loop-var` value for this access.
      //
      // But since all of the other dimensions are '1', they cannot generate
      // a broadcasting situation. Thus, for this access function, we can
      // generate the access `loop-var` regardless of whether the `?` will
      // evaluate to 1 or 10 at runtime. It will be correct no matter which
      // situation we will be in.
      bool allOtherInputDimsAreOne = true;
      for (int64_t i = 0; i < inputSize; ++i) {
        if (i == operandIndex)
          continue;
        IndexExpr otherInputDim = inputsDims[i][dimIndex];
        if (!otherInputDim.isLiteralAndIdenticalTo(1)) {
          allOtherInputDimsAreOne = false;
          break;
        }
      }
      useLoopIndexNoMatterWhat = allOtherInputDimsAreOne;
    }

    // Compute access index based on broadcasting rules.
    if (operandDim.isLiteralAndIdenticalTo(1)) {
      // Dim of size 1: access is always 0.
      operandAccessExprs.emplace_back(LitIE(0));
    } else if (noBroadcasting || useLoopIndexNoMatterWhat) {
      // No broadcasting or we can use the loop index no matter what -> just use
      // the index.
      //
      // useLoopIndexNoMatterWhat -> use loop index without worries.
      // Our dim may be [*, dim, *] where all the others are [*, 1, *];
      // Regardless of the value of dim (constant, `?`) we can use the loop
      // index variable without reservation as if dim is 1, then its 0 by def,
      // and if dim>1, then its 0...dim-1 without issue.
      operandAccessExprs.emplace_back(loopAccessExprs[dimIndex]);
    } else if (flattenedInnerDims && r == operandRank - 1) {
      // Flattened dims: we only have manageable broadcast; either scalar
      // (access is 0) or non-broadcast (access is loop index).
      assert(!operandDim.isLiteralAndIdenticalTo(1) && "treated before");
      operandAccessExprs.emplace_back(loopAccessExprs[dimIndex]);
    } else {
      // If dim is a compile time constant, then the test below will resolve
      // at compile time. If dim is dynamic (i.e. only known at runtime), then
      // we will issue code for the compare and select and the right value
      // will be used at runtime.
      operandAccessExprs.emplace_back(
          IndexExpr::select(operandDim > 1, loopAccessExprs[dimIndex], 0));
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Unary Op Shape Helper
//===----------------------------------------------------------------------===//

LogicalResult ONNXUnaryOpShapeHelper::computeShape() {
  // Set the variables that belong to superclass ONNXBroadcastOpShapeHelper
  // (inputsDims, outputRank) to valid values. The hasUniBroadcast flag is
  // already set to default false in the constructor.
  outputRank = createIE->getShapedTypeRank(operands[0]);
  DimsExpr dims;
  createIE->getShapeAsDims(operands[0], dims);
  inputsDims.emplace_back(dims);
  return setOutputDimsFromOperand(operands[0]);
}

bool ONNXUnaryOpShapeHelper::hasNoBroadcast(DimAnalysis *dimAnalysis) {
  // Unary op have no broadcast.
  return true;
}

bool ONNXUnaryOpShapeHelper::hasManageableBroadcastForInnerDims(
    int64_t &collapsedInnermostLoops, int64_t &collapsedLiteralSize,
    IndexExpr &collapsedDynamicSize, DimAnalysis *dimAnalysis) {
  // Unary op have no broadcast; simply states that all dims can be collapsed.
  DimsExpr output = getOutputDims();
  int64_t outputRank = output.size();
  // Keep track of cumulative inner dim sizes.
  collapsedLiteralSize = 1;
  collapsedDynamicSize = LitIE(1);
  for (int64_t r = 0; r < outputRank; ++r) {
    if (output[r].isLiteral())
      collapsedLiteralSize *= output[r].getLiteral();
    else
      collapsedDynamicSize = collapsedDynamicSize * output[r];
  }
  // every input dim can be collapsed.
  collapsedInnermostLoops = outputRank;
  return true;
}

LogicalResult ONNXUnaryOpShapeHelper::getAccessExprs(Value operand,
    int64_t operandIndex, const SmallVectorImpl<IndexExpr> &loopAccessExprs,
    SmallVectorImpl<IndexExpr> &operandAccessExprs, bool flattenedInnerDims,
    bool hasNoBroadcast) {
  operandAccessExprs.clear();
  for (IndexExpr l : loopAccessExprs)
    operandAccessExprs.emplace_back(l);
  return success();
}

//===----------------------------------------------------------------------===//
// Setting a new constant or attribute value.
//===----------------------------------------------------------------------===//

void SaveOnnxConstInOp(
    Operation *op, const llvm::SmallVectorImpl<int64_t> &vals, int operandId) {
  OpBuilder builder(op->getContext());
  builder.setInsertionPoint(op);
  OnnxBuilder createONNX(builder, op->getLoc());
  Value constVal = createONNX.constantInt64(vals);
  op->setOperand(operandId, constVal);
}

void SaveOnnxConstInOp(Operation *op, MutableOperandRange operand,
    const llvm::SmallVectorImpl<int64_t> &vals) {
  OpBuilder builder(op->getContext());
  builder.setInsertionPoint(op);
  OnnxBuilder createONNX(builder, op->getLoc());
  Value constVal = createONNX.constantInt64(vals);
  operand.assign(constVal);
}

//===----------------------------------------------------------------------===//
// Setting the type of the output using explict element type and shape.
//===----------------------------------------------------------------------===//

/// Update a tensor type by using the given shape, elementType and encoding.
void updateType(Operation *op, Value val, ArrayRef<int64_t> shape,
    Type elementType, Attribute encoding, bool refineShape) {
  // Try to combine the given shape and the output's shape if possible.
  SmallVector<int64_t, 4> inferredShape;
  if (refineShape) {
    IndexExprScope scope(nullptr, val.getLoc());
    DimsExpr inferredDims;
    for (int64_t d : shape) {
      // TODO: "-1" may be used if "shape" is coming from e.g. the parameters
      // of an `onnx.Reshape` op?
      if (ShapedType::isDynamic(d) || d == -1)
        inferredDims.emplace_back(QuestionmarkIndexExpr(/*isFloat*/ false));
      else
        inferredDims.emplace_back(LitIE(d));
    }
    refineDims(op, inferredDims, val);
    IndexExpr::getShape(inferredDims, inferredShape);
  } else {
    // TODO: "-1" may be used if "shape" is coming from e.g. the parameters of
    // an `onnx.Reshape` op?
    for (size_t i = 0; i < shape.size(); ++i)
      inferredShape.emplace_back(
          shape[i] != -1 ? shape[i] : ShapedType::kDynamic);
  }

  // Get element type.
  if (!elementType)
    elementType = getElementType(val.getType());

  // Get encoding.
  if (auto valType = mlir::dyn_cast<RankedTensorType>(val.getType()))
    if (!encoding)
      encoding = valType.getEncoding();

  // Build result type.
  RankedTensorType resType;
  if (encoding)
    resType = RankedTensorType::get(inferredShape, elementType, encoding);
  else
    resType = RankedTensorType::get(inferredShape, elementType);
  // Reset type
  val.setType(resType);
}

static void resetTypeShapeToQuestionmarks(Value val) {
  // Only deal with ranked tensor types here.
  RankedTensorType valType = mlir::dyn_cast<RankedTensorType>(val.getType());
  if (!valType)
    return;
  // Reset any compile time literal to unknown (aka question marks).
  SmallVector<int64_t, 4> newShape(valType.getRank(), ShapedType::kDynamic);
  auto resType = RankedTensorType::Builder(valType).setShape(newShape);
  // Reset type
  val.setType(resType);
}

void resetTypesShapeToQuestionmarks(Operation *op) {
  int numRes = op->getNumResults();
  for (int i = 0; i < numRes; ++i)
    resetTypeShapeToQuestionmarks(op->getResult(i));
}

//===----------------------------------------------------------------------===//
// ONNX Custom Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXCustomOpShapeHelper::ONNXCustomOpShapeHelper(Operation *op,
    ValueRange operands, IndexExprBuilder *ieBuilder, IndexExprScope *scope,
    bool hasUniBroadcasting)
    : ONNXUnaryOpShapeHelper(op, operands, ieBuilder, scope) {
  ONNXCustomOp customOp = cast<ONNXCustomOp>(op);
  if (!customOp.getShapeInferPattern().has_value()) {
    pattern = 0;
    return;
  }

  if (customOp.getShapeInferPattern() == "SameAs") {
    pattern = 1;
  } else if (customOp.getShapeInferPattern() == "MDBroadcast") {
    pattern = 2;
  } else {
    // ToFix: move the check into verifier
    llvm_unreachable("The specified shape_infer_pattern is not supported"
                     "Error encountered in shape inference.");
  }

  std::optional<ArrayAttr> inputIndexAttrs = customOp.getInputsForInfer();
  ValueRange inputs =
      operands.empty() ? ValueRange(customOp.getInputs()) : operands;
  if (!inputIndexAttrs.has_value()) {
    return;
  }

  std::vector<Value> operandsVector;
  for (auto indexAttr : inputIndexAttrs.value()) {
    operandsVector.push_back(
        inputs[mlir::cast<IntegerAttr>(indexAttr).getInt()]);
  }
  setOperands(ValueRange(operandsVector));
}

bool ONNXCustomOpShapeHelper::isImplemented() { return pattern != 0; }

LogicalResult ONNXCustomOpShapeHelper::computeShape() {
  if (pattern == 1) {
    return ONNXUnaryOpShapeHelper::computeShape();
  } else if (pattern == 2) {
    return ONNXBroadcastOpShapeHelper::computeShape();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Template instantiation (last).
//===----------------------------------------------------------------------===//

// Ideally should be in more specific files

} // namespace onnx_mlir
