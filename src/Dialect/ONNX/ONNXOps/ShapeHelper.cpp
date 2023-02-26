/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------ONNXShapeHelper.cpp - help for shapes----------------=== //
//
// Copyright 2020-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file has the computations to compute the shapes using the new index expr
// approach.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

#include <algorithm>

#define DEBUG_TYPE "shape-helper"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper
//===----------------------------------------------------------------------===//

/// Refine `inferredDims` using the output's shape if possible. For example,
/// replacing a dynamic dim in `inferredDims` by a static dim in the output's
/// shape.
static void refineDims(DimsExpr &inferredDims, Value output) {
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
    // Sanity checks for old convention of using -1 for dynamic.
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
      inferredDims[i] = LiteralIndexExpr(existingDims[i]);
      continue;
    }
    // inferredDim is unknown at lowering: use existing dim for efficiency.
    if (!inferredDims[i].isLiteral()) {
      inferredDims[i] = LiteralIndexExpr(existingDims[i]);
      continue;
    }
    // inferredDim is different from existingDim. Believe in existingDim.
    assert(inferredDims[i].isLiteral() && "sanity");
    if (existingDims[i] != inferredDims[i].getLiteral()) {
      // Warning for users.
      llvm::outs() << "Warning: [Shape inference, dim " << i
                   << "] the inferred dim (" << inferredDims[i].getLiteral()
                   << ") is different from the existing dim ("
                   << existingDims[i] << "). Use the existing dim instead.\n";
      inferredDims[i] = LiteralIndexExpr(existingDims[i]);
    }
  }
}

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper::ONNXOpShapeHelper(Operation *inputOp,
    ArrayRef<Value> inputOperands, IndexExprBuilder *inputIeBuilder,
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
    operands = ArrayRef<Value>(privateOperandsCache);
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
    refineDims(privateOutputsDims[n], output);
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
  RankedTensorType rankedType = type.dyn_cast<RankedTensorType>();
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
  uint64_t resNum = op->getNumResults();
  for (uint64_t i = 0; i < resNum; ++i) {
    // If we have an optional type, leave it as is.
    if (op->getResults()[i].getType().isa<NoneType>())
      continue;
    llvm::SmallVector<int64_t, 4> shapeVect;
    IndexExpr::getShape(getOutputDims(i), shapeVect);
    // Set refineShape to false here because we refine it (or not) when setting
    // the output shape. So there is no need to perform this again here.
    updateType(op->getResults()[i], shapeVect, elementType, encoding,
        /*refineShape*/ false);
  }
  return success();
}

// Use a distinct type for each of the output.
LogicalResult ONNXOpShapeHelper::computeShapeAndUpdateTypes(
    TypeRange elementTypeRange, mlir::ArrayRef<mlir::Attribute> encodingList) {
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
    if (op->getResults()[i].getType().isa<NoneType>())
      continue;
    llvm::SmallVector<int64_t, 4> shapeVect;
    IndexExpr::getShape(getOutputDims(i), shapeVect);
    Type currElementType = elementTypeRange[i];
    // Set refineShape to false here because we refine it (or not) when setting
    // the output shape. So there is no need to perform this again here.
    updateType(op->getResults()[i], shapeVect, currElementType,
        hasEncoding ? encodingList[i] : nullptr, /*refineShape*/ false);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Broadcast Op Shape Helper
//===----------------------------------------------------------------------===//

LogicalResult ONNXBroadcastOpShapeHelper::customComputeShape(
    ArrayRef<Value> initialOperands, DimsExpr *additionalOperand) {
  // if additionalOperand is not used, we expect a zero-sized vector.
  // A temporary IndexExpr vector for the output.
  DimsExpr dimsExpr;
  uint64_t numOfInputs = initialOperands.size();

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
        if (!hasUniBroadcasting && !hasNoBroadcasting)
          dimsExpr[j] = nextDimExpr;
        continue;
      }
      // Case: LiteralNot1 - *.
      if (currentDimExpr.isLiteralAndDifferentThan(1)) {
        // LiteralNot1 - LiteralNot1 => keep unchanged with verifying.
        if (nextDimExpr.isLiteralAndDifferentThan(1) &&
            !currentDimExpr.isLiteralAndIdenticalTo(nextDimExpr))
          return op->emitError("Incompatible broadcast matching " +
                               std::to_string(currentDimExpr.getLiteral()) +
                               " with " +
                               std::to_string(currentDimExpr.getLiteral()));
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

LogicalResult ONNXBroadcastOpShapeHelper::getAccessExprs(Value operand,
    uint64_t operandIndex, const SmallVectorImpl<IndexExpr> &outputAccessExprs,
    SmallVectorImpl<IndexExpr> &operandAccessExprs) {
  if (hasNoBroadcasting || (hasUniBroadcasting && operandIndex == 0)) {
    for (IndexExpr ie : outputAccessExprs)
      operandAccessExprs.emplace_back(ie);
    return success();
  }

  uint64_t operandRank = operand.getType().cast<ShapedType>().getRank();
  for (uint64_t i = 0; i < operandRank; ++i) {
    // Shape helper may pretend 1s, thus adjust dimension index accordingly.
    uint64_t dimIndex = outputRank - operandRank + i;
    SymbolIndexExpr dim(inputsDims[operandIndex][dimIndex]);

    // Compute access index based on broadcasting rules.
    // If all other operand dims are 1, just use the output access index.
    // Otherwise, emit a select op.
    bool allOtherInputDimsAreOne = true;
    for (uint64_t i = 0; i < inputsDims.size(); ++i) {
      if (i == operandIndex)
        continue;
      IndexExpr dim = inputsDims[i][dimIndex];
      if (!dim.isLiteralAndIdenticalTo(1)) {
        allOtherInputDimsAreOne = false;
        break;
      }
    }
    if (allOtherInputDimsAreOne) {
      operandAccessExprs.emplace_back(outputAccessExprs[dimIndex]);
    } else {
      operandAccessExprs.emplace_back(
          IndexExpr::select(dim > 1, outputAccessExprs[dimIndex], 0));
    }
  }
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
void updateType(Value val, ArrayRef<int64_t> shape, Type elementType,
    Attribute encoding, bool refineShape) {
  // Try to combine the given shape and the output's shape if possible.
  SmallVector<int64_t, 4> inferredShape;
  if (refineShape) {
    IndexExprScope scope(nullptr, val.getLoc());
    DimsExpr inferredDims;
    for (int64_t d : shape) {
      // TODO: "-1" may be used if "shape" is coming from e.g. the parameters of
      // an `onnx.Reshape` op?
      if (ShapedType::isDynamic(d) || d == -1)
        inferredDims.emplace_back(QuestionmarkIndexExpr(/*isFloat*/ false));
      else
        inferredDims.emplace_back(LiteralIndexExpr(d));
    }
    refineDims(inferredDims, val);
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
  if (auto valType = val.getType().dyn_cast<RankedTensorType>())
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
  RankedTensorType valType = val.getType().dyn_cast<RankedTensorType>();
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
// Template instantiation (last).
//===----------------------------------------------------------------------===//

// Ideally should be in more specific files

} // namespace onnx_mlir
