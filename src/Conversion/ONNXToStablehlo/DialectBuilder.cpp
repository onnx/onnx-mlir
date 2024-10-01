/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------- DialectBuilder.cpp - Stablehlo dialect builder -----------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file contains dialect builder for Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/DialectBuilder.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Support/TypeUtilities.hpp"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace onnx_mlir {

// =============================================================================
// stablehlo Builder
// =============================================================================

Value StablehloBuilder::constant(Type type, double val) const {
  Value constant = nullptr;
  // Could be a vector type; look at the element type.
  Type elementType = type;
  VectorType vectorType = mlir::dyn_cast<VectorType>(type);
  if (vectorType)
    elementType = vectorType.getElementType();
  TypeSwitch<Type>(elementType)
      .Case<Float16Type>([&](Type) {
        constant =
            b().create<stablehlo::ConstantOp>(loc(), b().getF16FloatAttr(val));
      })
      .Case<Float32Type>([&](Type) {
        constant =
            b().create<stablehlo::ConstantOp>(loc(), b().getF32FloatAttr(val));
      })
      .Case<Float64Type>([&](Type) {
        constant =
            b().create<stablehlo::ConstantOp>(loc(), b().getF64FloatAttr(val));
      })
      .Case<IntegerType>([&](IntegerType elementType) {
        assert(val == static_cast<int64_t>(val) && "value is ambiguous");
        unsigned width = elementType.getWidth();

        if (width == 1)
          constant = b().create<stablehlo::ConstantOp>(
              loc(), b().getBoolAttr(val != 0));
        else {
          if (elementType.isUnsignedInteger()) {
            constant = b().create<stablehlo::ConstantOp>(
                loc(), b().getIntegerAttr(elementType,
                           APInt(width, static_cast<uint64_t>(val), false)));
          } else {
            constant = b().create<stablehlo::ConstantOp>(
                loc(), b().getIntegerAttr(elementType,
                           APInt(width, static_cast<int64_t>(val), true)));
          }
        }
      })
      .Case<IndexType>([&](Type elementType) {
        constant = b().create<stablehlo::ConstantOp>(
            loc(), b().getIntegerAttr(elementType, val));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });

  assert(constant != nullptr && "Expecting valid constant value");
  return constant;
}

Value StablehloBuilder::constantI64(int64_t val) const {
  IntegerAttr constantAttr = b().getIntegerAttr(b().getI64Type(), val);
  return b().create<stablehlo::ConstantOp>(loc(), constantAttr);
}

Value StablehloBuilder::shaped_zero(Type type) const {
  return b().create<stablehlo::ConstantOp>(loc(), b().getZeroAttr(type));
}

Value StablehloBuilder::reshape(Type resultType, Value operand) const {
  return b().create<stablehlo::ReshapeOp>(loc(), resultType, operand);
}

Value StablehloBuilder::dynamic_reshape(
    Type type, Value input, Value shape) const {
  return b().create<stablehlo::DynamicReshapeOp>(loc(), type, input, shape);
}

Value StablehloBuilder::real_dynamic_slice(Type type, Value operand,
    Value startIndices, Value limitIndices, Value strides) const {
  return b().create<stablehlo::RealDynamicSliceOp>(
      loc(), type, operand, startIndices, limitIndices, strides);
}

Value StablehloBuilder::dynamic_slice(Value operand,
    SmallVector<Value> startIndices, SmallVector<int64_t> sliceSizes) const {
  return b().create<stablehlo::DynamicSliceOp>(
      loc(), operand, startIndices, sliceSizes);
}

Value StablehloBuilder::dynamic_slice(Value operand,
    SmallVector<Value> startIndices, DenseI64ArrayAttr sliceSizes) const {
  return b().create<stablehlo::DynamicSliceOp>(
      loc(), operand, startIndices, sliceSizes);
}

Value StablehloBuilder::slice(Value operand, SmallVector<int64_t> startIndices,
    SmallVector<int64_t> limitIndices, SmallVector<int64_t> strides) const {
  return b().create<stablehlo::SliceOp>(
      loc(), operand, startIndices, limitIndices, strides);
}

Value StablehloBuilder::slice(Value operand, DenseI64ArrayAttr startIndices,
    DenseI64ArrayAttr limitIndices, DenseI64ArrayAttr strides) const {
  return b().create<stablehlo::SliceOp>(
      loc(), operand, startIndices, limitIndices, strides);
}

//===----------------------------------------------------------------------===//
// Extends OnnxBuilder with member functions that might generate stablehlo
// related dialect operations.
//===----------------------------------------------------------------------===//

Value OnnxToStablehloBuilder::reshape(
    const Value input, const ArrayRef<DimIndexExpr> shapeDims) const {
  assert(!shapeDims.empty() && "Shape dimensions should not be empty");

  ShapedType inputType = mlir::cast<ShapedType>(input.getType());
  Type elementType = inputType.getElementType();
  MultiDialectBuilder<StablehloBuilder, OnnxBuilder, ShapeBuilder> create(
      b(), loc());

  // If the output dimensions are all literals the 'onnx/Reshape' operation
  // can take the new shape via an 'onnx.Constant'.
  if (llvm::all_of(
          shapeDims, [](const DimIndexExpr &dim) { return dim.isLiteral(); })) {
    SmallVector<int64_t, 6> shape;
    for (const IndexExpr &dim : shapeDims)
      shape.push_back(dim.getLiteral());

    auto constantOp = create.onnx.constantInt64(shape);

    Value reshapeRes = create.onnx.reshape(
        RankedTensorType::get(shape, elementType), input, constantOp);

    return reshapeRes;
  }

  // When the output dimensions aren't all literals we need to generate code
  // to compute the shape.
  int64_t length = shapeDims.size();
  SmallVector<Value> dims;
  for (int64_t i = 0; i < length; ++i) {
    Value data = shapeDims[i].getValue();
    dims.push_back(data);
  }

  Value shapeExtent = create.shape.fromExtents(dims);
  Value shapeTensor = create.shape.toExtentTensor(
      RankedTensorType::get({length}, b().getIndexType()), shapeExtent);
  // result shape
  SmallVector<int64_t, 6> outputShape;
  for (const IndexExpr &dim : shapeDims)
    outputShape.push_back(
        dim.isLiteral() ? dim.getLiteral() : ShapedType::kDynamic);
  Value res = create.stablehlo.dynamic_reshape(
      RankedTensorType::get(outputShape, elementType), input, shapeTensor);
  return res;
}

Value OnnxToStablehloBuilder::transpose(const Value input,
    const ArrayRef<int64_t> perm,
    const ArrayRef<DimIndexExpr> outputDims) const {
  assert(!outputDims.empty() && "Output dimensions should not be empty");
  assert(!perm.empty() && perm.size() == outputDims.size() &&
         "Expecting valid permutation array");
  MultiDialectBuilder<OnnxBuilder> create(b(), loc());

  // Compute the shape of the 'onnx.Transpose' result.
  SmallVector<int64_t, 6> shape;
  for (const IndexExpr &dim : outputDims)
    shape.push_back(dim.isLiteral() ? dim.getLiteral() : ShapedType::kDynamic);

  // Create the "onnx.Transpose" operation.
  ShapedType inputType = mlir::cast<ShapedType>(input.getType());
  Value transposeRes = create.onnx.transpose(
      RankedTensorType::get(shape, inputType.getElementType()), input,
      b().getI64ArrayAttr(perm));

  return transposeRes;
}

// =============================================================================
// IndexExpr Builder for Lowering using Shape/Stablehlo Dialect.
// =============================================================================

// Return null if none is found.
ElementsAttr IndexExprBuilderForStablehlo::getConst(Value value) {
  auto definingOp = value.getDefiningOp();
  // If we have a cast between index/integer, skip it, i.e. get the defining op
  // that is the input to the cast.
  if (auto castOp = dyn_cast_or_null<arith::IndexCastOp>(definingOp)) {
    Value input = castOp.getIn();
    definingOp = input.getDefiningOp();
  }
  if (auto constOp = dyn_cast_or_null<stablehlo::ConstantOp>(definingOp)) {
    if (constOp.getValueAttr())
      return mlir::dyn_cast<ElementsAttr>(constOp.getValueAttr());
  } else if (auto constOp = dyn_cast_or_null<ONNXConstantOp>(definingOp)) {
    if (constOp.getValue().has_value())
      return mlir::dyn_cast<ElementsAttr>(constOp.getValueAttr());
  }
  return nullptr;
}

Value IndexExprBuilderForStablehlo::getVal(Value intArrayVal, uint64_t i) {
  Type elemType = getElementType(intArrayVal.getType());
  if (!mlir::isa<IndexType>(elemType)) {
    Type indexTensorType = RankedTensorType::get(
        mlir::cast<ShapedType>(intArrayVal.getType()).getShape(),
        b().getIndexType());
    intArrayVal =
        b().create<arith::IndexCastOp>(loc(), indexTensorType, intArrayVal);
  }
  ShapeBuilder createShape(*this);
  return createShape.getExtent(intArrayVal, i);
}

Value IndexExprBuilderForStablehlo::getShapeVal(
    Value tensorOrMemrefValue, uint64_t i) {
  ShapeBuilder createShape(*this);
  return createShape.dim(tensorOrMemrefValue, i);
}

} // namespace onnx_mlir
