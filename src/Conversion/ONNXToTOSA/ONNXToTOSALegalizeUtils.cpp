/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToTOSACommon.hpp - ONNX dialects to TOSA lowering Utils---===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file contains common utils shared by the functions performing the
// lowering to the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"   // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"            // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                 // from @llvm-project
#include "mlir/IR/PatternMatch.h"                 // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h" // from @llvm-project
#include "mlir/Support/LLVM.h"

#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp" // from @llvm-project
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"
#include <cstdint>

namespace mlir {
namespace tosa {

llvm::SmallVector<int64_t, 4> convertRankToShape(
    llvm::ArrayRef<int64_t> shapes) {
  llvm::SmallVector<int64_t, 4> vec(shapes.size(), 1);
  return vec;
};

mlir::RankedTensorType getTypeFromTensorShape(llvm::ArrayRef<int64_t> shape,
    mlir::Type elementType, mlir::Attribute encoding = {}) {
  return mlir::RankedTensorType::get(
      convertRankToShape(shape), elementType, encoding);
}

// Create a 32-bit float constant operator from a float
Value getTosaConstTensorSingleF32(PatternRewriter &rewriter, Operation *op,
    float val, llvm::ArrayRef<int64_t> shape) {
  auto constType = getTypeFromTensorShape(shape, rewriter.getF32Type());
  auto constAttr = DenseElementsAttr::get(constType, val);

  auto constOp = rewriter.create<ConstOp>(op->getLoc(), constType, constAttr);
  return constOp.getResult();
}
} // namespace tosa
} // namespace mlir