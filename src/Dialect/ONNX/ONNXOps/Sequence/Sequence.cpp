/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Sequence.cpp - ONNX Operations -------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Sequence operations.
//
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

// Sequence related operations
// The general form for seq is seq<tensor<*xT>>
// Tensors will be add to or removed from a seq dynamically.
// The tensor type in a seq should be a summary of all the tensor type in
// the seq.
// It is possible seq<tensor<*xT>> can be refined into seq<RankedTensor>,
// or even seq<StaticShapedTensor> if all the tensors have common shape info
// It is important to refine the type for seq in onnx-mlir because static
// type is used. If seq of unranked tensor remains, onnx-mlir can not handle
// the unranked tensor retrieved from the seq.
// Here is the rules for shape inferences of seq-related ops:
// * A seq is started empty as the result of SequenceEmpty. We can track this
//   property with a tag in seq type or along dataflow.
// * When the an element is added, we can merge its shape with that in seq.
// * when an element is removed from seq, the seq becomes empty if it is the
//   last tenor in the seq (known statically).
// Since the seq is usually used as a parameter of a graph (e.g. for LoopOp),
// shape inference for region may need improvement.

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace {

// Helper function used in Sequence ops shape inference
ShapedType sequenceAddType(
    ShapedType accumulatedType, ShapedType additionalType) {
  Type elementType = accumulatedType.getElementType();
  assert(elementType == additionalType.getElementType() &&
         "types to merge must have the same data type");
  // Pick the weaker attr: known dim > unknown dim > unranked
  if (!accumulatedType.hasRank())
    return accumulatedType;
  if (!additionalType.hasRank())
    return additionalType;
  int64_t rank = accumulatedType.getRank();
  if (rank != additionalType.getRank())
    return UnrankedTensorType::get(elementType);
  ArrayRef<int64_t> acc = accumulatedType.getShape();
  ArrayRef<int64_t> add = additionalType.getShape();
  SmallVector<int64_t, 4> dims;
  for (int64_t i = 0; i < rank; i++) {
    dims.push_back(acc[i] != add[i] ? ShapedType::kDynamic : add[i]);
  }
  return RankedTensorType::get(dims, elementType);
}

} // namespace

//===----------------------------------------------------------------------===//
// SequenceAtOp
//===----------------------------------------------------------------------===//

std::vector<Type> ONNXSequenceAtOp::resultTypeInference() {
  // The output is the element type of the input sequence.  If the input
  // sequence type is not yet resolved (e.g. NoneType at import time), fall
  // back to tensor<*xf32> so that downstream code (e.g. the Decompose pass
  // which runs before shape inference) always sees a valid ShapedType.
  auto seqType = mlir::dyn_cast<SeqType>(getInputSequence().getType());
  if (!seqType)
    return {UnrankedTensorType::get(Builder(getContext()).getF32Type())};
  auto shapedElem = mlir::dyn_cast<ShapedType>(seqType.getElementType());
  if (!shapedElem)
    return {UnrankedTensorType::get(Builder(getContext()).getF32Type())};
  return {UnrankedTensorType::get(shapedElem.getElementType())};
}

LogicalResult ONNXSequenceAtOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  auto outputType = getResult().getType();
  auto inputElementType =
      mlir::cast<SeqType>(getInputSequence().getType()).getElementType();
  if (!mlir::isa<UnrankedTensorType>(inputElementType) &&
      mlir::isa<UnrankedTensorType>(outputType)) {
    getResult().setType(inputElementType);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SequenceConstructOp
//===----------------------------------------------------------------------===//

std::vector<Type> ONNXSequenceConstructOp::resultTypeInference() {
  // Derive the sequence element type from the first input.  If inputs are not
  // yet typed (NoneType), fall back to f32.
  auto types = getInputs().getTypes();
  Type scalarType = Builder(getContext()).getF32Type();
  if (!types.empty()) {
    if (auto shapedType = mlir::dyn_cast<ShapedType>(types.front()))
      scalarType = shapedType.getElementType();
  }
  ShapedType elemType = UnrankedTensorType::get(scalarType);
  return {SeqType::get(elemType, static_cast<int64_t>(types.size()))};
}

LogicalResult ONNXSequenceConstructOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  auto types = getInputs().getTypes();
  ShapedType seqTensorType = mlir::cast<ShapedType>(types[0]);
  for (size_t i = 1; i < types.size(); ++i) {
    seqTensorType =
        sequenceAddType(seqTensorType, mlir::cast<ShapedType>(types[i]));
  }
  getResult().setType(SeqType::get(seqTensorType, types.size()));
  return success();
}

//===----------------------------------------------------------------------===//
// SequenceEmptyOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSequenceEmptyOp::verify() {
  // For the Optional dtypeAttr, the default type is F32
  Type elementType = getResultElementTypeFromDtypeDefaultingToF32(*this);

  // Get element type for seq from the output
  auto outputSeqElementType = mlir::cast<ShapedType>(
      mlir::cast<SeqType>(getResult().getType()).getElementType());
  if (outputSeqElementType.getElementType() != elementType)
    return emitError("SequenceEmpty getDtype() does not match the output type");
  return success();
}

LogicalResult ONNXSequenceEmptyOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  auto originTy = mlir::cast<SeqType>(getResult().getType());
  auto elementTy = originTy.getElementType();
  auto returnTy = SeqType::get(elementTy, 0);
  getResult().setType(returnTy);
  return success();
}

std::vector<Type> ONNXSequenceEmptyOp::resultTypeInference() {
  Type scalarType = getResultElementTypeFromDtypeDefaultingToF32(*this);
  ShapedType elementType = UnrankedTensorType::get(scalarType);
  return {SeqType::get(elementType, 0)};
}

//===----------------------------------------------------------------------===//
// SequenceEraseOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSequenceEraseOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  auto inputTy = mlir::cast<SeqType>(getInputSequence().getType());
  int64_t length = inputTy.getLength();

  if (length == 0)
    return emitError("SequenceErase from an empty seq");
  getResult().setType(SeqType::get(inputTy.getElementType(),
      length == ShapedType::kDynamic ? ShapedType::kDynamic : length - 1));
  return success();
}

//===----------------------------------------------------------------------===//
// SequenceInsertOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSequenceInsertOp::verify() {
  ONNXSequenceInsertOpAdaptor operandAdaptor =
      ONNXSequenceInsertOpAdaptor(*this);

  // These cast should be guaranteed by default verifier
  Type seqElementType =
      mlir::dyn_cast<SeqType>(operandAdaptor.getInputSequence().getType())
          .getElementType();
  Type elementType1 =
      mlir::dyn_cast<ShapedType>(seqElementType).getElementType();
  ShapedType insertType =
      mlir::dyn_cast<ShapedType>(operandAdaptor.getTensor().getType());
  Type elementType2 = insertType.getElementType();

  if (elementType1 != elementType2) {
    return emitError("Element types of the tensor in seqence and input "
                     "have to be the same");
  }
  return success();
}

LogicalResult ONNXSequenceInsertOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Merge the tensor type for the seq and the inserted tensor
  SeqType seqType = mlir::cast<SeqType>(getInputSequence().getType());
  ShapedType tensorType = mlir::cast<ShapedType>(getTensor().getType());
  int64_t length = seqType.getLength();
  if (length == 0) {
    // When the input seq is empty, inherit the tensor type
    getResult().setType(SeqType::get(tensorType, 1));
  } else {
    int64_t newLength =
        length == ShapedType::kDynamic ? ShapedType::kDynamic : length + 1;
    ShapedType seqTensorType = mlir::cast<ShapedType>(seqType.getElementType());
    seqTensorType = sequenceAddType(seqTensorType, tensorType);
    getResult().setType(SeqType::get(seqTensorType, newLength));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SequenceMapOp
//===----------------------------------------------------------------------===//

std::vector<Type> ONNXSequenceMapOp::resultTypeInference() {
  // The body region outputs are tensors; each becomes the element type of the
  // corresponding output sequence.  The output sequence length is unknown at
  // import time (it equals the input sequence length, resolved during shape
  // inference), so we use kDynamic.
  Operation *terminator = getBody().back().getTerminator();
  std::vector<Type> resultTypes;
  for (Type ty : terminator->getOperandTypes())
    resultTypes.push_back(
        SeqType::get(mlir::cast<ShapedType>(ty), ShapedType::kDynamic));
  return resultTypes;
}

//===----------------------------------------------------------------------===//
// SequenceLengthOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSequenceLengthOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Type outputTy = getResult().getType();
  if (!mlir::isa<RankedTensorType>(outputTy) ||
      mlir::cast<RankedTensorType>(outputTy).getRank() != 0) {
    SmallVector<int64_t, 1> dims;
    auto builder = Builder(getContext());
    Type scalarTy = RankedTensorType::get(dims, builder.getIntegerType(64));
    getResult().setType(cast<TensorType>(scalarTy));
  }
  // ElementType of I64 will be checked by verifier
  return success();
}
