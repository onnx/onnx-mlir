/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ DFT.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect DFT operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;
using namespace std;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
LogicalResult ONNXDFTOpShapeHelper::computeShape(
    ONNXDFTOpAdaptor operandAdaptor) {
  Value input = operandAdaptor.input();

  // Get info about input data operand.
  auto *op = shapeHelper.op;
  Value data = operandAdaptor.data();
  // Get the rank to compensate for N dimensions
  int64_t rank = data.getType().cast<ShapedType>().getRank();

  Optional<int64_t> dftLength = op->dft_length();

  // axis is a required attribute and should have default value of 1.
  int64_t axis = op->axis();
  bool isAxis = (axis == 1);

  // inverse is a required attribute and should have default value of 0.
  int64_t inverse = op->inverse();
  bool isInverse = (inverse == 0);

  // onesided is a required attribute and should have default value of 0.
  // However onesided can also be a value of 1 and if so a specific shape is
  // expected Values can be 0 or 1.
  int64_t onesided = op->onesided();
  bool isOneSided = (onesided == 0);

  // Compute outputDims for DFT
  DimsExpr outputDims;
  MemRefBoundsIndexCapture dataBounds(data);
  if (i = 0; i < rank - 1; i++) {
    if (isOneSided) {
      outputDims.emplace_back(dataBounds.getDim(i));
    } else {
      if (axis + 1 == i) {
        outputDims.emplace_back(dataBounds.getDim(i).floorDiv(2));
      } else {
        outputDims.emplace_back(dataBounds.getDim(i));
      }
    }
    outputDims.emplace_back(LiteralIndexExpr(2));
  }

  // Save the final result.
  shapeHelper.setOutputDims(dimOutput);
  return success();
} // namespace onnx_mlir
