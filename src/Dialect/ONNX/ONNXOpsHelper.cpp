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

/// Affine Maps to compute the convolution/pooling window.
///
/// The conv/pooling window can be smaller than the kernel when slicing it over
/// the border edges. Thus, we will compute the start and end indices for
/// each window dimension as follows.
///   firstValidH = ceil(float(ptH / dH)) * dH - ptH
///   startH = max(firstValidH, ho * sH - ptH)
///   endH = min(H, ho * sH + (kH - 1) * dH  + 1 - pbH)
///   hDim = round(float(endH - startH) / float(dH))
/// We also want to compute how the window is smaller than the kernel.
///   kernelOffset = min(0, ho * sH - ptH)
///
/// How to derive 'firstValidH':
///   When dilation is non-unit, the first valid pixel to apply conv/pooling on
///   will not be the 0-th pixel, but rather the smallest integer n to make
///   '-pH + n * dH' greater than or equal to 0, where pH and dH are pad
///   and dilation along axis H. We derive what is this smallest n:
///   -pH + n * dH >= 0
///         n * dH >= pH
///              n >= pH/dH
///   thus n = ceil(pH/dH)
///   thus the first valid pixel location is 'ceil(pH / dH) * dH- pH'.
///
/// This function returns {startH, endH, hDim, kernelOffset}.

std::vector<AffineMap> getAffineMapsForConvWindow(
    Builder &builder, bool ceilMode, bool isDilated) {
  // Affine maps for the conv/pooling window.
  AffineMap windowStartMap, windowEndMap, windowDimMap, kernelOffsetMap;
  { // Construct windowStartMap, windowEndMap and windowDimMap.
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

    // windowDimMap
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
    windowDimMap = AffineMap::get(1, 5, dimExpr, builder.getContext());

    // windowStartMap, windowEndMap, and kernelOffsetMap.
    windowStartMap =
        AffineMap::get(1, 5, {start1, start2}, builder.getContext());
    windowEndMap = AffineMap::get(1, 5, {end1, end2}, builder.getContext());
    kernelOffsetMap = AffineMap::get(1, 5,
        {mlir::getAffineConstantExpr(0, builder.getContext()), start2},
        builder.getContext());
  }

  return std::vector<AffineMap>{
      windowStartMap, windowEndMap, windowDimMap, kernelOffsetMap};
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
    Type elementType = type1.cast<MemRefType>().getElementType();
    RankedTensorType ty1 =
        RankedTensorType::get(type1.cast<MemRefType>().getShape(), elementType);
    RankedTensorType ty2 =
        RankedTensorType::get(type2.cast<MemRefType>().getShape(), elementType);
    // Compute a broadcasted type.
    Type outputType = OpTrait::util::getBroadcastedType(ty1, ty2);
    // Construct a MemRefType.
    return MemRefType::get(
        outputType.cast<RankedTensorType>().getShape(), elementType);
  } else
    return {};
}
