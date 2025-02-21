/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====----- ONNXToZHighCommon.cpp - Common functions to ZHigh lowering ----===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file includes utility functions for lowering ONNX operations to a
// combination of ONNX and ZHigh operations.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHighCommon.hpp"
#include "src/Accelerators/NNPA/Compiler/NNPACompilerOptions.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"

using namespace mlir;
namespace onnx_mlir {

bool isEnableScalarBcastBinary() { return nnpaEnableScalarBcastBinary; }

/// Get transposed tensor by using a permutation array.
Value emitONNXTranspose(
    Location loc, PatternRewriter &rewriter, Value x, ArrayRef<int64_t> perms) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  Value result = create.onnx.transposeInt64(x, perms);
  return result;
}

/// Get transposed tensor by using a permutation array and a result type.
Value emitONNXTransposeWithType(Location loc, PatternRewriter &rewriter,
    Type transposedType, Value x, ArrayRef<int64_t> perms) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  Value result =
      create.onnx.transpose(transposedType, x, rewriter.getI64ArrayAttr(perms));
  return result;
}

/// Split a tensor along an axis in which each chunk has a size of
/// NNPAGetMaxForDim and the last chunk can be smaller.
ValueRange splitAlongAxis(
    MultiDialectBuilder<OnnxBuilder> &create, Value X, int64_t axis) {
  Type xType = X.getType();
  ArrayRef<int64_t> xShape = getShape(xType);
  int64_t xRank = xShape.size();
  Type elementTy = getElementType(xType);

  // Compute split sizes.
  SmallVector<Type> splitTy;
  SmallVector<int64_t> splitSizesI64;
  SmallVector<int64_t> splitShape(xShape);
  int64_t dimSize = xShape[axis];
  // First splits have the same size of NNPAGetMaxForDim.
  int64_t maxSize = NNPAGetMaxForDim(axis, xRank);
  while (dimSize > maxSize) {
    splitShape[axis] = maxSize;
    auto ty = RankedTensorType::get(splitShape, elementTy);
    splitTy.emplace_back(ty);
    splitSizesI64.emplace_back(maxSize);
    dimSize -= maxSize;
  }
  // The last split.
  splitShape[axis] = dimSize;
  auto ty = RankedTensorType::get(splitShape, elementTy);
  splitTy.emplace_back(ty);
  splitSizesI64.emplace_back(dimSize);

  Value splitSizes = create.onnx.constantInt64(splitSizesI64);
  ValueRange splits = create.onnx.split(splitTy, X, splitSizes, axis);
  return splits;
}

bool isF32ScalarConstantTensor(Value v) {
  if (!isScalarConstantTensor(v))
    return false;
  auto t = mlir::dyn_cast<ShapedType>(v.getType());
  return t.getElementType().isF32();
}

FloatAttr getScalarF32AttrFromConstant(Value v) {
  if (!isF32ScalarConstantTensor(v))
    return nullptr;
  ElementsAttr constElements = getElementAttributeFromONNXValue(v);
  return constElements.getSplatValue<FloatAttr>();
}

Value getDynShape(Location loc, PatternRewriter &rewriter, Value x) {
  if (!hasShapeAndRank(x))
    llvm_unreachable("The input must have shape and rank");

  OnnxBuilder create(rewriter, loc);
  auto t = mlir::dyn_cast<ShapedType>(x.getType());
  int64_t r = t.getRank();
  SmallVector<Value> dims;
  for (int64_t i = 0; i < r; ++i) {
    Value d = create.dim(x, i);
    dims.emplace_back(d);
  }
  return create.concat(
      RankedTensorType::get({r}, rewriter.getI64Type()), dims, 0);
}

int ONNXToZHighLoweringConfiguration::optReportNNPAUnsupportedOps =
    0; // 0: Compile option (--opt-report=NNPAUnsupportedOps) not specified.
int ONNXToZHighLoweringConfiguration::reportOnNNPAUnsupportedOps =
    0; // 0: no reporting.
bool ONNXToZHighLoweringConfiguration::isDynQuant = false;
bool ONNXToZHighLoweringConfiguration::Quant::isActivationSym = false;
bool ONNXToZHighLoweringConfiguration::Quant::isWeightSym = true;
llvm::SmallVector<std::string>
    ONNXToZHighLoweringConfiguration::Quant::opTypes = {};

void configureONNXToZHighLoweringPass(bool optReportNNPAUnsupportedOps,
    bool isDynQuant, bool quantIsActivationSym, bool quantIsWeightSym,
    llvm::ArrayRef<std::string> quantOpTypes) {
  ONNXToZHighLoweringConfiguration::optReportNNPAUnsupportedOps =
      optReportNNPAUnsupportedOps;
  ONNXToZHighLoweringConfiguration::isDynQuant = isDynQuant;
  if (isDynQuant) {
    ONNXToZHighLoweringConfiguration::Quant::isActivationSym =
        quantIsActivationSym;
    ONNXToZHighLoweringConfiguration::Quant::isWeightSym = quantIsWeightSym;
    ONNXToZHighLoweringConfiguration::Quant::opTypes.insert(
        ONNXToZHighLoweringConfiguration::Quant::opTypes.begin(),
        quantOpTypes.begin(), quantOpTypes.end());
  }
}

} // namespace onnx_mlir
