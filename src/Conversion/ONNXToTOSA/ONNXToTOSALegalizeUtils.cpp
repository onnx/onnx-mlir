/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==== ONNXToTosaLegalizeUtils.cpp - ONNX dialects to TOSA lowering Utils-===//
//
// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file contains common utils shared by the functions performing the
// lowering to the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h" // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"          // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"                 // from @llvm-project
#include "mlir/IR/PatternMatch.h"                 // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h" // from @llvm-project
#include "mlir/Support/LLVM.h"

#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp" // from @llvm-project
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"
#include <cstdint>
#include <src/Dialect/Mlir/IndexExpr.hpp>

using namespace mlir;
namespace onnx_mlir {
namespace tosa {

mlir::RankedTensorType reduceAxisToOne(llvm::ArrayRef<int64_t> shape,
    mlir::Type elementType, mlir::Attribute encoding) {
  return mlir::RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(shape.size(), 1), elementType, encoding);
}

} // namespace tosa
} // namespace onnx_mlir
