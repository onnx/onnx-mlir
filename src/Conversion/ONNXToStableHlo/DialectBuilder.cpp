/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ DialectBuilder.cpp - StableHlo dialect builder
//--------------------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file contains dialect builder for StableHlo dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/Conversion/ONNXToStableHlo/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

Value StablehloBuilder::constant(mlir::Type type, double val) const {
  Value constant = nullptr;
  // Could be a vector type; look at the element type.
  Type elementType = type;
  VectorType vectorType = type.dyn_cast<VectorType>();
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
        assert(val == (int64_t)val && "value is ambiguous");
        unsigned width = elementType.getWidth();

        if (width == 1)
          constant = b().create<stablehlo::ConstantOp>(
              loc(), b().getBoolAttr(val != 0));
        else {
          if (elementType.isUnsignedInteger()) {
            constant = b().create<stablehlo::ConstantOp>(
                loc(), b().getIntegerAttr(
                           elementType, APInt(width, (uint64_t)val, false)));
          } else {
            constant = b().create<stablehlo::ConstantOp>(
                loc(), b().getIntegerAttr(
                           elementType, APInt(width, (int64_t)val, true)));
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

Value StablehloBuilder::constantIndex(int64_t val) const {
  IntegerAttr constantAttr = b().getIntegerAttr(b().getIndexType(), val);
  return b().create<stablehlo::ConstantOp>(loc(), constantAttr);
}

Value StablehloBuilder::shaped_zero(mlir::Type type) const {
  return b().create<stablehlo::ConstantOp>(loc(), b().getZeroAttr(type));
}

// =============================================================================
// IndexExpr Builder for Lowering using Shape/StableHlo Dialect.
// =============================================================================

// Return null if none is found.
ElementsAttr IndexExprBuilderForStableHlo::getConst(Value value) {
  auto definingOp = value.getDefiningOp();
  // If we have a cast between index/integer, skip it, i.e. get the defining op
  // that is the input to the cast.
  if (auto castOp = dyn_cast_or_null<arith::IndexCastOp>(definingOp)) {
    Value input = castOp.getIn();
    definingOp = input.getDefiningOp();
  }
  if (auto constOp = dyn_cast_or_null<stablehlo::ConstantOp>(definingOp)) {
    if (constOp.getValueAttr())
      return constOp.getValueAttr().dyn_cast<ElementsAttr>();
  } else if (auto constOp = dyn_cast_or_null<ONNXConstantOp>(definingOp)) {
    if (constOp.getValue().has_value())
      return constOp.getValueAttr().dyn_cast<ElementsAttr>();
  }
  return nullptr;
}

Value IndexExprBuilderForStableHlo::getVal(Value intArrayVal, uint64_t i) {
  Type elemType = getElementType(intArrayVal.getType());
  if (!elemType.isa<IndexType>()) {
    Type indexTensorType = RankedTensorType::get(
        intArrayVal.getType().cast<ShapedType>().getShape(),
        b().getIndexType());
    intArrayVal =
        b().create<arith::IndexCastOp>(loc(), indexTensorType, intArrayVal);
  }
  ShapeBuilder createShape(*this);
  return createShape.getExtent(intArrayVal, i);
}

Value IndexExprBuilderForStableHlo::getShapeVal(
    Value tensorOrMemrefValue, uint64_t i) {
  ShapeBuilder createShape(*this);
  return createShape.dim(tensorOrMemrefValue, i);
}

} // namespace onnx_mlir
