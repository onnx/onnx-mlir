/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ If.cpp - ONNX Operations --------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect IF operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace {
bool areCompatibleIfTypes(Type ifResultType, Type branchResultType) {
  // ifResultType must be tensor/seq/opt type because that's checked in
  // ONNXIfOp::verifyInvariantsImpl()
  if (ShapedType ifShapedType = ifResultType.dyn_cast<ShapedType>()) {
    if (ShapedType branchShapedType = branchResultType.dyn_cast<ShapedType>()) {
      return ifShapedType.getElementType() == branchShapedType.getElementType();
    } else {
      return false;
    }
  }
  if (SeqType ifSeqType = ifResultType.dyn_cast<SeqType>()) {
    if (SeqType branchSeqType = branchResultType.dyn_cast<SeqType>()) {
      return areCompatibleIfTypes(
          ifSeqType.getElementType(), branchSeqType.getElementType());
    } else {
      return false;
    }
  }
  if (OptType ifOptType = ifResultType.dyn_cast<OptType>()) {
    if (OptType branchOptType = branchResultType.dyn_cast<OptType>()) {
      return areCompatibleIfTypes(
          ifOptType.getElementType(), branchOptType.getElementType());
    } else {
      return false;
    }
  }
  llvm_unreachable("areCompatibleIfTypes called with non tensor/seq/opt type");
}

// Pre-condition: areCompatibleIfTypes(ifTy, lhs) && areCompatibleIfTypes(ifTy,
// rhs)
Type unionOfIfTypes(Type lhs, Type rhs) {
  // All asserts below are checked in areCompatibleIfTypes().
  if (ShapedType lhsShapedType = lhs.dyn_cast<ShapedType>()) {
    ShapedType rhsShapedType = rhs.cast<ShapedType>();
    Type elementType = lhsShapedType.getElementType();
    assert(elementType == rhsShapedType.getElementType() &&
           "tensor element types mismatch");
    if (lhsShapedType.hasRank() && rhsShapedType.hasRank() &&
        lhsShapedType.getRank() == rhsShapedType.getRank()) {
      int64_t rank = lhsShapedType.getRank();
      auto lhsShape = lhsShapedType.getShape();
      auto rhsShape = rhsShapedType.getShape();
      SmallVector<int64_t, 4> shape;
      for (int64_t i = 0; i < rank; ++i) {
        shape.push_back(lhsShape[i] == rhsShape[i] ? lhsShape[i] : -1);
      }
      return RankedTensorType::get(shape, elementType);
    } else {
      return UnrankedTensorType::get(elementType);
    }
  }
  if (SeqType lhsSeqType = lhs.dyn_cast<SeqType>()) {
    SeqType rhsSeqType = rhs.cast<SeqType>();
    int64_t length = lhsSeqType.getLength() == rhsSeqType.getLength()
                         ? lhsSeqType.getLength()
                         : -1;
    return SeqType::get(unionOfIfTypes(lhsSeqType.getElementType(),
                            rhsSeqType.getElementType()),
        length);
  }
  if (OptType lhsOptType = lhs.dyn_cast<OptType>()) {
    OptType rhsOptType = rhs.cast<OptType>();
    return OptType::get(unionOfIfTypes(
        lhsOptType.getElementType(), rhsOptType.getElementType()));
  }
  llvm_unreachable("unionOfIfTypes called with non tensor/seq/opt type");
}
} // namespace

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXIfOp::verify() {
  size_t ifNumResults = getNumResults();
  assert(ifNumResults == outputs().size() && "outputs() != all results");
  auto thenResults = then_branch().back().getTerminator()->getOperands();
  if (ifNumResults != thenResults.size())
    return emitOpError() << "then branch #results=" << thenResults.size()
                         << " differ from if #results=" << ifNumResults;
  auto elseResults = else_branch().back().getTerminator()->getOperands();
  if (ifNumResults != elseResults.size())
    return emitOpError() << "else branch #results=" << elseResults.size()
                         << " differ from if #results=" << ifNumResults;
  auto thenResultTypes = thenResults.getTypes();
  auto elseResultTypes = elseResults.getTypes();
  for (size_t i = 0; i < ifNumResults; ++i) {
    Type ifResultType = getResultTypes()[i];
    if (!areCompatibleIfTypes(ifResultType, thenResultTypes[i]))
      emitOpError() << "then branch disagrees on result type #" << (i + 1)
                    << " of " << ifNumResults;
    if (!areCompatibleIfTypes(ifResultType, elseResultTypes[i]))
      emitOpError() << "else branch disagrees on result type #" << (i + 1)
                    << " of " << ifNumResults;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXIfOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  doShapeInference(then_branch());
  doShapeInference(else_branch());
  size_t ifNumResults = getNumResults();
  auto thenResultTypes =
      then_branch().back().getTerminator()->getOperandTypes();
  auto elseResultTypes =
      else_branch().back().getTerminator()->getOperandTypes();
  // assert is checked in verify()
  assert(ifNumResults == thenResultTypes.size() &&
         ifNumResults == elseResultTypes.size() &&
         "if #results and branches #results differ");
  for (size_t i = 0; i < ifNumResults; ++i) {
    getResult(i).setType(
        unionOfIfTypes(thenResultTypes[i], elseResultTypes[i]));
  }
  return success();
}
