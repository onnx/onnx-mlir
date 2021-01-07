//===------- ONNXOpsHelper.cpp - Helper functions for ONNX dialects -------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for lowering ONNX ops to Krnl Dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

// Identity affine
using namespace mlir;
using namespace mlir::onnxmlir;

AffineMap getIdentityDimMap(Builder &builder) {
  return AffineMap::get(1, 0, {builder.getAffineDimExpr(0)});
}

// Pool/conv affine
// dim =
//   let numerator = (input + pad - (kernel - 1) * dilation - 1)
//   in let denominator = stride
//      in
//        if (ceilMode)
//          ceil(numerator / denominator) + 1
//        else
//          floor(numerator / denominator) + 1
AffineMap getConvDimMap(Builder &builder, bool ceilMode) {
  AffineExpr input = builder.getAffineDimExpr(0);
  AffineExpr kernel = builder.getAffineSymbolExpr(0);
  AffineExpr pad = builder.getAffineSymbolExpr(1);
  AffineExpr stride = builder.getAffineSymbolExpr(2);
  AffineExpr dilation = builder.getAffineSymbolExpr(3);

  AffineExpr dimExp;
  if (ceilMode)
    dimExp = (input + pad - (kernel - 1) * dilation - 1).ceilDiv(stride) + 1;
  else
    dimExp = (input + pad - (kernel - 1) * dilation - 1).floorDiv(stride) + 1;

  return AffineMap::get(1, 4, {dimExp});
}

//===----------------------------------------------------------------------===//
// ONNX Helper functions
//===----------------------------------------------------------------------===//

size_t ArrayAttrSize(ArrayAttr a) { return a.size(); }

size_t ArrayAttrSize(Optional<ArrayAttr> a) { return a.getValue().size(); }

int64_t ArrayAttrIntVal(ArrayAttr a, int i) {
  return (a.getValue()[i]).cast<IntegerAttr>().getInt();
}

int64_t ArrayAttrIntVal(Optional<ArrayAttr> a, int i) {
  return (a.getValue().getValue()[i]).cast<IntegerAttr>().getInt();
}

DenseElementsAttr getDenseElementAttributeFromValue(Value value) {
  auto definingOp = value.getDefiningOp();
  if (auto constantOp = dyn_cast_or_null<mlir::ONNXConstantOp>(definingOp))
    return constantOp.valueAttr().dyn_cast<DenseElementsAttr>();
  else if (auto globalOp = dyn_cast_or_null<mlir::KrnlGlobalOp>(definingOp))
    if (globalOp.value().hasValue())
      return globalOp.valueAttr().dyn_cast<DenseElementsAttr>();
  return nullptr;
}

bool getIntegerLiteralFromValue(Value value, int64_t &intLit) {
  // From lib/Dialect/LinAlg/Transform/Promotion.cpp
  if (auto constantOp = value.getDefiningOp<ConstantOp>()) {
    if (constantOp.getType().isa<IndexType>())
      intLit = constantOp.value().cast<IntegerAttr>().getInt();
    return true;
  }
  // Since ConsantIndexOp is a subclass of ConstantOp, not sure if this one is
  // useful.
  if (auto constantOp = value.getDefiningOp<ConstantIndexOp>()) {
    if (constantOp.getType().isa<IndexType>())
      intLit = constantOp.value().cast<IntegerAttr>().getInt();
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Get a broadcasted type for RankedTensorType and MemRefType.
//===----------------------------------------------------------------------===//
Type getBroadcastedRankedType(Type type1, Type type2) {
  if (type1.isa<RankedTensorType>() && type2.isa<RankedTensorType>())
    return OpTrait::util::getBroadcastedType(type1, type2);
  if (type1.isa<MemRefType>() && type2.isa<MemRefType>()) {
    // Contruct RankedTensorType(s).
    auto elementType = type1.cast<MemRefType>().getElementType();
    auto ty1 =
        RankedTensorType::get(type1.cast<MemRefType>().getShape(), elementType);
    auto ty2 =
        RankedTensorType::get(type2.cast<MemRefType>().getShape(), elementType);
    // Compute a broadcasted type.
    Type outputType = OpTrait::util::getBroadcastedType(ty1, ty2);
    // Construct a MemRefType.
    return MemRefType::get(
        outputType.cast<RankedTensorType>().getShape(), elementType);
  } else
    return (Type)NULL;
}
