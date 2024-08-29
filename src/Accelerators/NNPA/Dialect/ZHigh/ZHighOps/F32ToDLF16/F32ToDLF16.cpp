/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- F32ToDLF16.cpp - ZHigh Operations -------------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/F32ToDLF16/ONNXZHighF32ToDLF16.inc"
} // end anonymous namespace

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// Custom builders
//===----------------------------------------------------------------------===//

void ZHighF32ToDLF16Op::build(OpBuilder &builder, OperationState &state,
    Value input, IntegerAttr saturation) {
  Type elementType = builder.getF16Type();
  Type resType = UnrankedTensorType::get(elementType);

  if (auto inType = mlir::dyn_cast<RankedTensorType>(input.getType()))
    resType = RankedTensorType::get(inType.getShape(), elementType);

  build(builder, state, resType, input, saturation);
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighF32ToDLF16Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Canonicalization patterns
//===----------------------------------------------------------------------===//

void ZHighF32ToDLF16Op::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<ConversionRemovalPattern>(context);
}

} // namespace zhigh
} // namespace onnx_mlir
