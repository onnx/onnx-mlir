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
  if (ShapedType ifShapedType = mlir::dyn_cast<ShapedType>(ifResultType)) {
    if (ShapedType branchShapedType =
            mlir::dyn_cast<ShapedType>(branchResultType)) {
      return ifShapedType.getElementType() == branchShapedType.getElementType();
    } else {
      return false;
    }
  }
  if (SeqType ifSeqType = mlir::dyn_cast<SeqType>(ifResultType)) {
    if (SeqType branchSeqType = mlir::dyn_cast<SeqType>(branchResultType)) {
      return areCompatibleIfTypes(
          ifSeqType.getElementType(), branchSeqType.getElementType());
    } else {
      return false;
    }
  }
  if (OptType ifOptType = mlir::dyn_cast<OptType>(ifResultType)) {
    if (OptType branchOptType = mlir::dyn_cast<OptType>(branchResultType)) {
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
  if (ShapedType lhsShapedType = mlir::dyn_cast<ShapedType>(lhs)) {
    ShapedType rhsShapedType = mlir::cast<ShapedType>(rhs);
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
        shape.push_back(
            lhsShape[i] == rhsShape[i] ? lhsShape[i] : ShapedType::kDynamic);
      }
      return RankedTensorType::get(shape, elementType);
    } else {
      return UnrankedTensorType::get(elementType);
    }
  }
  if (SeqType lhsSeqType = mlir::dyn_cast<SeqType>(lhs)) {
    SeqType rhsSeqType = mlir::cast<SeqType>(rhs);
    int64_t length = lhsSeqType.getLength() == rhsSeqType.getLength()
                         ? lhsSeqType.getLength()
                         : -1;
    return SeqType::get(unionOfIfTypes(lhsSeqType.getElementType(),
                            rhsSeqType.getElementType()),
        length);
  }
  if (OptType lhsOptType = mlir::dyn_cast<OptType>(lhs)) {
    OptType rhsOptType = mlir::cast<OptType>(rhs);
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
  assert(ifNumResults == getOutputs().size() && "outputs() != all results");
  auto thenResults = getThenBranch().back().getTerminator()->getOperands();
  if (ifNumResults != thenResults.size())
    return emitOpError() << "then branch #results=" << thenResults.size()
                         << " differ from if #results=" << ifNumResults;
  auto elseResults = getElseBranch().back().getTerminator()->getOperands();
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
// Type Inference
//===----------------------------------------------------------------------===//

std::vector<Type> ONNXIfOp::resultTypeInference() {
  auto thenResultTypes =
      getThenBranch().back().getTerminator()->getOperandTypes();
  auto elseResultTypes =
      getElseBranch().back().getTerminator()->getOperandTypes();
  // assert is checked in verify()
  assert(getNumResults() == thenResultTypes.size() &&
         getNumResults() == elseResultTypes.size() &&
         "if #results and branches #results differ");
  std::vector<Type> resultTypes;
  for (auto [thenTy, elseTy] : llvm::zip(thenResultTypes, elseResultTypes))
    resultTypes.push_back(unionOfIfTypes(thenTy, elseTy));
  return resultTypes;
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXIfOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  doShapeInference(getThenBranch());
  doShapeInference(getElseBranch());
  for (auto [i, ty] : llvm::enumerate(resultTypeInference()))
    getResult(i).setType(ty);
  return success();
}
