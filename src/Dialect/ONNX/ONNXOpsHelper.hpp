/*
 * SPDX-License-Identifier: Apache-2.0
 */

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
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "src/Dialect/ONNX/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

//====---------------- EDSC Support with Value ---------------------------===//
namespace mlir {
using onnx_add = mlir::edsc::ValueBuilder<ONNXAddOp>;
using onnx_sub = mlir::edsc::ValueBuilder<ONNXSubOp>;
using onnx_mul = mlir::edsc::ValueBuilder<ONNXMulOp>;
using onnx_div = mlir::edsc::ValueBuilder<ONNXDivOp>;
using onnx_matmul = mlir::edsc::ValueBuilder<ONNXMatMulOp>;
using onnx_gemm = mlir::edsc::ValueBuilder<ONNXGemmOp>;
} // namespace mlir

// Identity affine map:
// #map = affine_map<(d0)[] -> d0>
mlir::AffineMap getIdentityDimMap(mlir::Builder &builder);

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
mlir::AffineMap getConvDimMap(mlir::Builder &builder, bool ceilMode);

/// IndexExprs to compute the start and end indices of the convolution/pooling
/// window.
///
/// The conv/pooling window can be smaller than the kernel when slicing it over
/// the border edges. Thus, we will compute the start and end indices for
/// each window dimension as follows.
///   firstValidH = ceil(float(ptH / dH)) * dH - ptH
///   startH = max(firstValidH, ho * sH - ptH)
///   endH = min(H, ho * sH + (kH - 1) * dH  + 1 - pbH)
///
/// Full conv/pooling window can be reconstructed by:
///   hDim = round(float(endH - startH) / float(dH))
//
/// We also want to compute the relative position of the window w.r.t. the
/// kernel.
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
/// This function returns {startH, endH, kernelOffset}.
std::vector<mlir::IndexExpr> getIndexExprsForConvWindow(
    llvm::SmallVectorImpl<mlir::IndexExpr> &inputExprs, bool ceilMode,
    bool isDilated);

/// The conv/pooling window can be smaller than the kernel when slicing it over
/// the border edges. This function returns an AffineMap to compute the size of
/// one edge of the window.
mlir::AffineMap getWindowAffineMap(
    mlir::Builder &builder, bool ceilMode, bool isDilated);

// Helper functions to get values from attribute arrays.
size_t ArrayAttrSize(mlir::ArrayAttr a);
size_t ArrayAttrSize(llvm::Optional<mlir::ArrayAttr> a);
int64_t ArrayAttrIntVal(mlir::ArrayAttr a, int i);
int64_t ArrayAttrIntVal(llvm::Optional<mlir::ArrayAttr> a, int i);

// This function satisfies the ArrayValueIndexCapture::DenseElementsAttr lambda
// type, using ONNX operations only.
mlir::DenseElementsAttr getDenseElementAttributeFromONNXValue(
    mlir::Value value);

mlir::ONNXConstantOp getONNXConstantOp(mlir::Value value);
mlir::Value getONNXConstantOpFromDenseAttr(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Attribute dense);
bool isFromNone(mlir::Value value);
mlir::Type getBroadcastedRankedType(mlir::Type type1, mlir::Type type2);

//===----------------------------------------------------------------------===//
// Support for transpose patterns.
//===----------------------------------------------------------------------===//

/// Compute the combined permute pattern from a pair of permute patterns.
mlir::ArrayAttr CombinedTransposePattern(mlir::PatternRewriter &rewriter,
    mlir::ArrayAttr firstPermAttr, mlir::ArrayAttr secondPermAttr);

/// Test if the permute pattern correspond to an identity pattern.
/// Identity patterns are {0, 1, 2, ... , rank -1}.
bool IsIdentityPermuteVector(mlir::ArrayAttr permAttr);

/// Test if two axis arrays contain the same values or not.
bool AreTheSameAxisArray(
    int64_t rank, mlir::ArrayAttr lhsAttr, mlir::ArrayAttr rhsAttr);

//===----------------------------------------------------------------------===//
// Support for Rewrite.
//===----------------------------------------------------------------------===//

// Create a DenseElementsAttr from a float attribute.
mlir::DenseElementsAttr createDenseElementsAttrFromFloatAttr(
    mlir::PatternRewriter &rewriter, mlir::Type elementType,
    mlir::FloatAttr attr);

mlir::DenseElementsAttr createDenseElementsAttrFromFloatAttrs(
    mlir::PatternRewriter &rewriter, mlir::Type elementType,
    llvm::SmallVector<mlir::Attribute> attrs);

// Create a DenseElementsAttr from a integer attribute.
// The attribute is assumed to be SingedInteger.
mlir::DenseElementsAttr createDenseElementsAttrFromIntegerAttr(
    mlir::PatternRewriter &rewriter, mlir::Type elementType,
    mlir::IntegerAttr attr);

mlir::DenseElementsAttr createDenseElementsAttrFromFloatAttrs(
    mlir::PatternRewriter &rewriter, mlir::Type elementType,
    llvm::SmallVector<mlir::Attribute> attrs);

// Integer attribute is assumed to be Signedless
mlir::DenseElementsAttr createDenseElementsAttrFromIntegerAttrs(
    mlir::PatternRewriter &rewriter, mlir::Type elementType,
    llvm::SmallVector<mlir::Attribute> attrs);

// Create a DenseElementsAttr from a String attribute.
mlir::DenseElementsAttr createDenseElementsAttrFromStringAttrs(
    mlir::PatternRewriter &rewriter, mlir::Type elementType,
    llvm::SmallVector<mlir::Attribute> attrs);

mlir::Value normalizeConstantOp(
    mlir::PatternRewriter &rewriter, mlir::Value output, mlir::Attribute attr);

// Create a DenseElementsAttr based on the shape of type.
mlir::DenseElementsAttr createDenseElementsAttrFromShape(
    mlir::PatternRewriter &rewriter, mlir::Value value);

// Create a DenseElementsAttr based on the size of type.
mlir::DenseElementsAttr createDenseElementsAttrFromSize(
    mlir::PatternRewriter &rewriter, mlir::Value value);

// Check whether a value is produced by a dense ONNXConstantOp.
bool isDenseONNXConstant(mlir::Value result);
