/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- OpHelper.hpp - Helper functions for ONNX dialects ---------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for lowering ONNX ops to Krnl Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_OPS_HELPER_H
#define ONNX_MLIR_OPS_HELPER_H

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
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"
#include "src/Support/Diagnostic.hpp"
#include "src/Support/TypeUtilities.hpp"

#include <algorithm>
#include <string>

namespace onnx_mlir {

/// This function returns a location with the corresponding ONNX operator name
/// inside. This is useful when tracing what expanded MLIR instructions
/// correspond to what ONNX operator.
///
template <typename OP_TYPE>
mlir::Location ONNXLoc(mlir::Operation *op);

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
mlir::Type convertTensorTypeToTensorTypeWithEncoding(
    const mlir::Type inputType, mlir::Attribute encodingAttr);

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

/// Return true if two tensors or memrefs have the same rank.
bool sameRank(mlir::Value tensorOrMemref1, mlir::Value tensorOrMemref2);

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
size_t ArrayAttrSize(std::optional<mlir::ArrayAttr> a);
int64_t ArrayAttrIntVal(mlir::ArrayAttr a, int i);
int64_t ArrayAttrIntVal(std::optional<mlir::ArrayAttr> a, int i);
void ArrayAttrIntVals(mlir::ArrayAttr a, mlir::SmallVectorImpl<int64_t> &i);

mlir::ElementsAttr getElementAttributeFromONNXValue(mlir::Value value);

mlir::ONNXConstantOp getONNXConstantOp(mlir::Value value);

// Obtain an array of int64_t values stored in ONNXConstantOp and append it to
// the given SmallVector iRes.
// Return true if successfully obtaining the array. Otherwise, false.
bool getI64ValuesFromONNXConstantOp(
    mlir::Value val, mlir::SmallVectorImpl<int64_t> &iRes);

// Test if the value is none. Since none is a unit value it never makes a
// difference whether it's a constant (the result of ONNXNoneOp) or the
// optional result of some other op (e.g. ONNXDropoutOp mask result).
// Note: It's ok to inline the isa<NoneType> test and not call this function.
inline bool isNoneValue(mlir::Value value);

//===----------------------------------------------------------------------===//
// Support for transpose patterns.
//===----------------------------------------------------------------------===//

/// Compute the combined permute pattern from a pair of permute patterns.
mlir::ArrayAttr CombinedTransposePattern(mlir::PatternRewriter &rewriter,
    mlir::ArrayAttr firstPermAttr, mlir::ArrayAttr secondPermAttr);

/// Test if the permute pattern correspond to an identity pattern.
/// Identity patterns are {0, 1, 2, ... , rank -1}.
bool IsIdentityPermuteVector(mlir::ArrayAttr permAttr);

/// Test if the value has the specified constant shape
bool HasSpecifiedConstantShape(mlir::Value value, mlir::Value shape);

/// Test if a value is a scalar constant tensor or not, i.e. tensor<dtype> or
/// tensor<1xdtype>.
bool isScalarConstantTensor(mlir::Value v);

/// Test if 'val' has shape and rank or not.
bool hasShapeAndRank(mlir::Value val);
bool hasShapeAndRank(mlir::Operation *op);

/// Test if a value has only one use except ONNXDimOp.
bool hasOneUseExceptDimOp(mlir::Value val);

//===----------------------------------------------------------------------===//
// Support for Rewrite.
//===----------------------------------------------------------------------===//

// Create a (rank 1) DenseElementsAttr from a float attribute.
mlir::DenseElementsAttr createDenseElementsAttrFromFloatAttr(
    mlir::PatternRewriter &rewriter, mlir::Type elementType,
    mlir::FloatAttr attr);

// Create a DenseElementsAttr based on the shape of type at the given index.
mlir::DenseElementsAttr createDenseElementsAttrFromShapeAtIndex(
    mlir::PatternRewriter &rewriter, mlir::Value value,
    mlir::IntegerAttr indexAttr);

// Create a DenseElementsAttr based on the size of type.
mlir::DenseElementsAttr createDenseElementsAttrFromSize(
    mlir::PatternRewriter &rewriter, mlir::Value value);

// Create an ArrayAttr from a dense ConstantOp
mlir::ArrayAttr createArrayAttrFromConstantOp(mlir::ONNXConstantOp constOp);

// Check whether a value is produced by a dense ONNXConstantOp.
bool isDenseONNXConstant(mlir::Value result);

// Get scalar value when it is a constant.
template <typename RESULT_TYPE>
RESULT_TYPE getScalarValue(mlir::ElementsAttr denseAttr, mlir::Type type);

template <typename RESULT_TYPE>
RESULT_TYPE getScalarValue(mlir::ONNXConstantOp constantOp);

/// Return the wide type of a value.
WideNum asWideNum(double n, mlir::Type elemType);

/// Checks whether a constant tensor's elements are all equal to a given scalar.
bool isConstOf(mlir::Value constValue, double n);

mlir::Type convertONNXTypeToMLIRType(
    mlir::Builder &builder, onnx::TensorProto_DataType onnxType);

/// Get the ONNX type corresponding to an MLIR type.
int64_t mlirTypeToOnnxType(mlir::Type elemType);

/// Check if a value is a scalar tensor.
bool isScalarTensor(mlir::Value v);

bool hasIntegerPowerExponent(mlir::ONNXPowOp *op, int64_t &exponentValue);

//===----------------------------------------------------------------------===//
// Support for dim operations.
//===----------------------------------------------------------------------===//

/// Check the defining operation of a value.
template <typename OP>
bool definedBy(mlir::Value v);

// This is to match if two values A and B are bijectively defined by OP1 and
// OP2. In other words,
// - if A is defined by OP1, then B would be defined by OP2.
// - if A is defined by OP2, then B would be defined by OP1.
//
// In both case, the output has two values,
// - the first one is the value defined by OP1,
// - the second one is the value defined by OP2.
//
// For example, to recognize BOTH A*B+C and C+A*B, where C is defined by
// ONNXConstant
// ```
// %C = onnx.Constant
// %AB = onnx.MatMul(A, B)
// onnx.Add(%AB, %C);
// ```
//
// We can use:
// Value lhs = addOp.getOperation(0);
// Value rhs = addOp.getOperation(1);
// ValueRange matchedValued;
//
// Value AB, C;
// areDefinedBy<ONNXMatMulOp, ONNXConstantOp>(lhs, rhs, AB, C);
//
// Note: The order of A and B are not important, they can be swapped.
template <typename OP1, typename OP2>
bool areDefinedBy(mlir::Value A, mlir::Value B, mlir::Value &matchedOP1,
    mlir::Value &matchedOP2);

// Check if the operation defining `op->operand[matchThisOperandIndex]` matches
// `OP`. If it does, set matchOperand to that operand, and matchOp to that
// defining op. Otherwise, don't change the match values.
// See operandOfOpDefinedBy comments in its implementation for suggested usages.
template <typename OP>
bool operandOfOpDefinedBy(mlir::Operation *&matchOp, mlir::Operation *op,
    mlir::Value &matchOperand, int64_t matchThisOperandIndex = 0);

// Same as above for binary operations, setting matchOperand0 and matchOperand1.
template <typename OP>
bool operandOfOpDefinedBy(mlir::Operation *&matchOp, mlir::Operation *op,
    mlir::Value &matchOperand0, mlir::Value &matchOperand1,
    int64_t matchThisOperandIndex);

// This is to recognize a binary op, e.g. A*B where one of A and B is a constant
// and the other one is defined by OP.
// Note: this function can handle the communitive property of the binary op.
//
// For example, to recognize this pattern:
// %x = "onnx.Tanh"()
// %y = 0.5 * %x    // or %x * 0.5
//
// we call
// ```
//   ONNXTanhOp tanhOp;
//   bool found = matchConstAndOp<ONNXTanhOp>(A, B, 0.5, tanhOp);
// ```
// where `A` and `B` are operands of ONNXMul that produces %y.
template <typename OP>
bool matchConstAndOp(mlir::Value A, mlir::Value B, double cst, OP &op);

// This is to recognize a binary op, e.g. A*B where one of A and B is the given
// value and the other one is defined by OP.
// Note: this function can handle the communitive property of the binary op.
//
// For example, to recognize this pattern where %z is one of the inputs of *,
// and the other input of * is defined by onnx.Tanh:
// %x = "onnx.Tanh"()
// %y = %z * %x    // or %x * %z
//
// we call
// ```
//   Value z;
//   ONNXTanhOp tanhOp;
//   bool found = matchConstAndOp<ONNXTanhOp>(A, B, z, tanhOp);
// ```
// where `A` and `B` are operands of ONNXMul that produces %y.
template <typename OP>
bool matchValueAndOp(
    mlir::Value A, mlir::Value B, mlir::Value matchValue, OP &matchOp);

/// Check if a value is to store dimensions, meaning it is a tensor of one
/// element or concatenation of one-element tensors.
bool areDims(mlir::Value val);

/// Check if a value is defined by Concat to store dimensions.
bool areDimsFromConcat(mlir::Value val);

/// Get all dimensions that are stored by the value.
void getDims(mlir::Value val, llvm::SmallVectorImpl<mlir::Value> &dims);

//===----------------------------------------------------------------------===//
// Support for ReshapeOp.
//===----------------------------------------------------------------------===//

// Return true if reshape does nothing, aka it returns the same as the input.
// Use dimAnalysis if provided.

bool isIdentityReshape(
    mlir::ONNXReshapeOp reshapeOp, const DimAnalysis *dimAnalysis = nullptr);

bool isIdentityReshape(mlir::Value input, mlir::Value output,
    const DimAnalysis *dimAnalysis = nullptr);

//===----------------------------------------------------------------------===//
// Support for location.
//===----------------------------------------------------------------------===//

std::string getNodeNameInPresenceOfOpt(
    mlir::Operation *op, bool useFileLine = true);

//===----------------------------------------------------------------------===//
// Support for DenseElementsAttr.
//===----------------------------------------------------------------------===//

/// Returns true if elementsAttr is a DenseResourceAttr with a blob that can not
/// be received
bool isElementAttrUninitializedDenseResource(mlir::ElementsAttr elementsAttr);

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp.inc"

} // namespace onnx_mlir
#endif
