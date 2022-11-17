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

#pragma once

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include "onnx/onnx_pb.h"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXLayoutHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/Diagnostic.hpp"
#include "src/Support/TypeUtilities.hpp"

#include <algorithm>
#include <string>

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// ONNX Tensor support.

/// Get a ONNX Custom Tensor data layout by StringRef. If layout string is a
/// standard layout or any other unrecognized string, just return false.
bool convertStringToONNXCustomTensorDataLayout(mlir::StringAttr layoutAttr,
    mlir::ONNXTensorEncodingAttr::DataLayout &layout, int64_t &xFactor,
    int64_t &yFactor);

/// Convert a data layout to StringRef, assert on error. Default yFactor value
/// is undef, namely 0.
llvm::StringRef convertONNXTensorDataLayoutToString(
    mlir::ONNXTensorEncodingAttr::DataLayout layout, int64_t xFactor,
    int64_t yFactor = 0);

// Add ONNX tensor encoding to ranked & shaped types. Return type only has the
// encoding if the layout is custom, Currently assert for non ranked/shaped
// type.
mlir::Type convertTensorTypeToTensorTypeWithONNXTensorEncoding(
    mlir::Builder &builder, const mlir::Type inputType,
    mlir::StringAttr layoutAttr);

/// Return true if the tensor is a ONNX tensor (having ONNXTensorEncodingAttr).
bool isONNXTensor(const mlir::Type type);

/// Get a ONNX tensor encoding attribute from a type.Returns null-attribute for
/// any type without an encoding.
mlir::ONNXTensorEncodingAttr getONNXTensorEncoding(mlir::Type type);

/// Get the layout of a ONNX tensor.
mlir::ONNXTensorEncodingAttr::DataLayout getONNXTensorLayout(mlir::Type type);

// Return true if both types have the same ONNX Tensor Data Layout (does not
// check for dimensions, elementary types...).
bool identicalONNXTensorDataLayout(
    const mlir::Type type1, const mlir::Type type2);

// Return true if the type has a layout associated with convolution
// optimizations.
bool hasConvONNXTensorDataLayout(const mlir::Type type);

// Return true if the type has a layout, and that layout is not STANDARD.
bool hasCustomONNXTensorDataLayout(const mlir::Type type);

//===----------------------------------------------------------------------===//
// Identity map

// Identity affine map:
// #map = affine_map<(d0)[] -> d0>
mlir::AffineMap getIdentityDimMap(mlir::Builder &builder);

//===----------------------------------------------------------------------===//
// Support for pool/convolutions

// Pool/conv affine map:
// #map0 = affine_map<(d0)[s0, s1, s2, s3]
//                    -> (d0 + s1 - (s0 - 1) * s3 - 1) floorDiv s2 + 1>
// In the case of `ceilMode = true`:
// #map0 = affine_map<(d0)[s0, s1, s2, s3]
//                    -> (d0 + s1 - (s0 - 1) * s3 - 1) ceilDiv s2 + 1>
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
/// The conv/pooling window can be smaller than the kernel when slicing it
/// over the border edges. Thus, we will compute the start and end indices for
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
///   When dilation is non-unit, the first valid pixel to apply conv/pooling
///   on will not be the 0-th pixel, but rather the smallest integer n to make
///   '-pH + n * dH' greater than or equal to 0, where pH and dH are pad
///   and dilation along axis H. We derive what is this smallest n:
///   -pH + n * dH >= 0
///         n * dH >= pH
///              n >= pH/dH
///   thus n = ceil(pH/dH)
///   thus the first valid pixel location is 'ceil(pH / dH) * dH- pH'.
///
/// This function returns {startH, endH, kernelOffset}.
std::vector<onnx_mlir::IndexExpr> getIndexExprsForConvWindow(
    llvm::SmallVectorImpl<onnx_mlir::IndexExpr> &inputExprs, bool ceilMode,
    bool isDilated);

/// The conv/pooling window can be smaller than the kernel when slicing it
/// over the border edges. This function returns an AffineMap to compute the
/// size of one edge of the window.
mlir::AffineMap getWindowAffineMap(
    mlir::Builder &builder, bool ceilMode, bool isDilated);

// Helper functions to get values from attribute arrays.
size_t ArrayAttrSize(mlir::ArrayAttr a);
size_t ArrayAttrSize(llvm::Optional<mlir::ArrayAttr> a);
int64_t ArrayAttrIntVal(mlir::ArrayAttr a, int i);
int64_t ArrayAttrIntVal(llvm::Optional<mlir::ArrayAttr> a, int i);

// This function satisfies the ArrayValueIndexCapture::GetDenseVal
// lambda type, using ONNX operations only.
mlir::ElementsAttr getElementAttributeFromONNXValue(mlir::Value value);

mlir::ONNXConstantOp getONNXConstantOp(mlir::Value value);
mlir::ONNXConstantOp createONNXConstantOpWithDenseAttr(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Attribute dense);
mlir::Value createNoneIntegerConstant(
    mlir::PatternRewriter &rewriter, mlir::Location loc);
mlir::Value createNoneFloatConstant(
    mlir::PatternRewriter &rewriter, mlir::Location loc);

bool isFromNone(mlir::Value value);

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

/// Create a DenseElementsAttr from a raw buffer.
mlir::DenseElementsAttr createDenseElementsAttrFromRawBuffer(
    mlir::Type resType, char *buf);

mlir::Value normalizeConstantOp(
    mlir::PatternRewriter &rewriter, mlir::Value output, mlir::Attribute attr);

// Create a DenseElementsAttr based on the shape of type at the given index.
mlir::DenseElementsAttr createDenseElementsAttrFromShapeAtIndex(
    mlir::PatternRewriter &rewriter, mlir::Value value,
    mlir::IntegerAttr indexAttr);

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
RESULT_TYPE getScalarValue(mlir::ElementsAttr denseAttr, mlir::Type type);

template <typename RESULT_TYPE>
RESULT_TYPE getScalarValue(mlir::ONNXConstantOp constantOp, mlir::Type type);

mlir::Type convertONNXTypeToMLIRType(
    mlir::OpBuilder &builder_, onnx::TensorProto_DataType onnxType);

/// Get the ONNX type corresponding to an MLIR type.
int64_t mlirTypeToOnnxType(mlir::Type elemType);

//===----------------------------------------------------------------------===//
// Support for dim operations.
//===----------------------------------------------------------------------===//

/// Check the defining operation of a value.
template <typename OP>
bool definedBy(mlir::Value v);

/// Check if a value is to store dimensions, meaning it is defined by
/// Dim/Constant/Cast/Concat.
bool areDims(mlir::Value val);

/// Check if a value is defined by Concat to store dimensions.
bool areDimsFromConcat(mlir::Value val);

/// Get all dimensions that are stored by the value.
void getDims(mlir::Value val, llvm::SmallVectorImpl<mlir::Value> &dims);

} // namespace onnx_mlir
