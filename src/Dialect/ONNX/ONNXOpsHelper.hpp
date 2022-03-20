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
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "onnx/onnx_pb.h"
#include "src/Dialect/ONNX/IndexExpr.hpp"
#include "src/Dialect/ONNX/MLIRDialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

namespace mlir {

//====-------------------------- ONNX Builder ---------------------------===//

struct OnnxBuilder : DialectBuilder {
  OnnxBuilder(OpBuilder &b, Location loc) : DialectBuilder(b, loc) {}
  OnnxBuilder(DialectBuilder &db) : DialectBuilder(db) {}

  Value add(Value A, Value B) const;
  Value sub(Value A, Value B) const;
  Value mul(Value A, Value B) const;
  Value div(Value A, Value B) const;
  Value matmul(Type Y, Value A, Value B) const;

  Value reshape(Type outputType, Value input, Value shape) const;
  Value transpose(Type outputType, Value input, ArrayAttr perm) const;

  Value constant(Attribute denseAttr) const;
};

// Recursive class specialized for OnnxBuilder refereed to as onnx.
template <class... Ts>
struct MultiDialectBuilder<OnnxBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(OpBuilder &b, Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), onnx(b, loc) {}
  MultiDialectBuilder(DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), onnx(db) {}
  OnnxBuilder onnx;
};

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
mlir::Value createONNXConstantOpWithDenseAttr(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Attribute dense);
mlir::Value createNoneIntegerConstant(
    mlir::PatternRewriter &rewriter, mlir::Location loc);
mlir::Value createNoneFloatConstant(
    mlir::PatternRewriter &rewriter, mlir::Location loc);

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

/// Test if the value has the specified constant shape
bool HasSpecifiedConstantShape(mlir::Value value, mlir::Value shape);

/// Test if two constant ops contain the same values or not.
bool AreTheSameConstantOpDenseAttr(
    mlir::Builder &builder, int64_t rank, mlir::Value lhsOp, mlir::Value rhsOp);

/// Test if 'val' has shape and rank or not.
bool hasShapeAndRank(mlir::Value val);

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

// Create an ArrayAttr from a dense ConstantOp
mlir::ArrayAttr createArrayAttrFromConstantOp(
    mlir::Builder &builder, mlir::Value constOp);

// Check whether a value is produced by a dense ONNXConstantOp.
bool isDenseONNXConstant(mlir::Value result);

// Get scalar value when it is a constant.
template <typename RESULT_TYPE>
RESULT_TYPE getScalarValue(mlir::DenseElementsAttr &denseAttr, mlir::Type type);

template <typename RESULT_TYPE>
RESULT_TYPE getScalarValue(mlir::ONNXConstantOp constantOp, mlir::Type type);

mlir::Type convertONNXTypeToMLIRType(
    mlir::OpBuilder &builder_, onnx::TensorProto_DataType onnxType);
