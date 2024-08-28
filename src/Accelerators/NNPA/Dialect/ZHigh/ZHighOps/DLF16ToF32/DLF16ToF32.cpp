/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- DLF16ToF32.cpp - ZHigh Operations -------------------===//
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
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/DLF16ToF32/ONNXZHighDLF16ToF32.inc"
} // end anonymous namespace

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// Custom builders
//===----------------------------------------------------------------------===//

void ZHighDLF16ToF32Op::build(
    OpBuilder &builder, OperationState &state, Value input) {
  Type elementType = builder.getF32Type();
  Type resType = UnrankedTensorType::get(elementType);

  if (auto inType = mlir::dyn_cast<RankedTensorType>(input.getType()))
    resType = RankedTensorType::get(inType.getShape(), elementType);

  build(builder, state, resType, input);
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighDLF16ToF32Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Canonicalization patterns
//===----------------------------------------------------------------------===//

void ZHighDLF16ToF32Op::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<ConversionRemovalPattern>(context);
  results.insert<DelayDLF16ToF32ViaReshapePattern>(context);
  results.insert<DelayDLF16ToF32ViaTransposePattern>(context);
  results.insert<DelayDLF16ToF32ViaSqueezePattern>(context);
  results.insert<DelayDLF16ToF32ViaUnsqueezePattern>(context);
  results.insert<DimDLF16ToF32RemovalPattern>(context);
}
} // namespace zhigh
} // namespace onnx_mlir
