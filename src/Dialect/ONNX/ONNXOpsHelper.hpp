//===------- ONNXOpsHelper.hpp - Helper functions for ONNX dialects -------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for lowering ONNX ops to Krnl Dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

using namespace mlir;

// Identity affine map:
// #map = affine_map<(d0)[] -> d0>
AffineMap getIdentityDimMap(Builder &builder);

// Pool/conv affine map:
// #map0 = affine_map<(d0)[s0, s1, s2, s3]
//                    -> (d0 + s1 - (s0 - 1) * s3 - 1) floordiv s2 + 1>
// In the case of `ceilMode = true`:
// #map0 = affine_map<(d0)[s0, s1, s2, s3]
//                    -> (d0 + s1 - (s0 - 1) * s3 - 1) ceildiv s2 + 1>
// where:
// - d0: input dim
// - s0: kernel
// - s1: pad
// - s2: stride
// - s3: dilation
AffineMap getConvDimMap(Builder &builder, bool ceilMode);

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
/// How 'firstValidH' was derived:
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
    Builder &builder, bool ceilMode, bool isDilated);

// Helper functions to get values from attribute arrays.
size_t ArrayAttrSize(ArrayAttr a);
size_t ArrayAttrSize(Optional<ArrayAttr> a);
int64_t ArrayAttrIntVal(ArrayAttr a, int i);
int64_t ArrayAttrIntVal(Optional<ArrayAttr> a, int i);

DenseElementsAttr getDenseElementAttributeFromValue(Value value);
bool getIntegerLiteralFromValue(Value value, int64_t &intLit);
Type getBroadcastedRankedType(Type type1, Type type2);
