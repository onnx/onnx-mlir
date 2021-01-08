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

/// Pool/conv helper affine expressions

std::vector<AffineMap> getAffineMapsForConvWindow(
    Builder &builder, bool ceilMode, bool isDilated) {
  // Affine maps for the pooling window.
  AffineMap poolStartMap, poolEndMap, poolDimMap;
  { // Construct poolStartMap, poolEndMap and poolDimMap.
    // AffineExpr(s) to obtain the dimensions and symbols.
    AffineExpr outputIndex = builder.getAffineDimExpr(0);
    AffineExpr inputDim = builder.getAffineSymbolExpr(0);
    AffineExpr kernelDim = builder.getAffineSymbolExpr(1);
    AffineExpr padTopDim = builder.getAffineSymbolExpr(2);
    AffineExpr strideDim = builder.getAffineSymbolExpr(3);
    AffineExpr dilationDim = builder.getAffineSymbolExpr(4);
    AffineExpr start1 =
        (padTopDim).ceilDiv(dilationDim) * dilationDim - padTopDim;
    AffineExpr start2 = outputIndex * strideDim - padTopDim;
    AffineExpr end1 = inputDim;
    AffineExpr end2 =
        outputIndex * strideDim + (kernelDim - 1) * dilationDim + 1 - padTopDim;

    // poolDimMap
    SmallVector<AffineExpr, 4> dimExpr;
    // Upperbound for an affine.for is `min AffineMap`, where `min` is
    // automatically inserted when an affine.for is constructed from
    // an AffineMap, thus we rewrite `endH - startH` as follows:
    //   endH - start H
    //     = min(end1, end2) - max(start1, start2)
    //     = min(end1 - start1, end1 - start2, end2 - start1, end2 - start2)
    AffineExpr dimExpr1 = end1 - start1;
    AffineExpr dimExpr2 = end1 - start2;
    AffineExpr dimExpr3 = end2 - start1;
    AffineExpr dimExpr4 = end2 - start2;
    for (AffineExpr de : {dimExpr1, dimExpr2, dimExpr3, dimExpr4}) {
      if (isDilated) {
        de = de + 1;
        de = (ceilMode) ? de.ceilDiv(dilationDim) : de.floorDiv(dilationDim);
      }
      dimExpr.emplace_back(de);
    }
    poolDimMap = AffineMap::get(1, 5, dimExpr, builder.getContext());

    // poolStartMap and poolEndMap
    poolStartMap =
        AffineMap::get(1, 5, {start1, start2}, builder.getContext());
    poolEndMap = AffineMap::get(1, 5, {end1, end2}, builder.getContext());
  }

  return std::vector<AffineMap>{poolStartMap, poolEndMap, poolDimMap};
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
