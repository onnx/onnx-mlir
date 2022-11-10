/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====----- ONNXToLinaglCommon.cpp - ONNX dialects to Linagl lowering ---------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the KRNL dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToLinagl/ONNXToLinaglCommon.hpp"
#include "src/Accelerators/Accelerator.hpp"
#include "src/Dialect/Linagl/DialectBuilder.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

bool ONNXToLinagl_gEmitDealloc = false;

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Type conversion from Onnx types to Linagl types.
//===----------------------------------------------------------------------===//

LinaglTypeConverter::LinaglTypeConverter() {
  // The order of type conversion is important: later ones are tried earlier.
  addConversion([](Type type) { return type; });

  addConversion([](ONNXStringType stringType) {
    return krnl::StringType::get(stringType.getContext());
  });

  addConversion([](TensorType tensorType) {
    assert(tensorType.hasRank() && "expected only ranked shapes");
    if (tensorType.getElementType().isa<ONNXStringType>()) {
      Type elementType = krnl::StringType::get(tensorType.getContext());
      return MemRefType::get(tensorType.getShape(), elementType);
    }
    // Accelerators may have special versions of TensorType. Call the
    // conversions of accelerators.
    for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators()) {
      MemRefType memRefType = accel->convertTensorTypeToMemRefType(tensorType);
      if (memRefType)
        return memRefType;
    }
    if (hasCustomONNXTensorDataLayout(tensorType))
      return convertTypeWithCustomONNXDataLayoutToMemRef(tensorType);
    return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  });

  addConversion([](SeqType seqType) {
    ShapedType seqElementType = seqType.getElementType();
    Type elementType = seqElementType.getElementType();
    Type seqElementConvertedType;
    if (seqElementType.hasRank()) {
      seqElementConvertedType =
          MemRefType::get(seqElementType.getShape(), elementType);
    } else {
      seqElementConvertedType = UnrankedMemRefType::get(elementType, 0);
    }
    SmallVector<int64_t, 1> dims;
    dims.emplace_back(seqType.getLength());
    llvm::ArrayRef<int64_t> shape(dims.data(), dims.size());
    return MemRefType::get(shape, seqElementConvertedType);
  });

  addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> Optional<Value> {
    if (inputs.size() != 1)
      return llvm::None;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });

  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> Optional<Value> {
    if (inputs.size() != 1)
      return llvm::None;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
}

} // namespace onnx_mlir
