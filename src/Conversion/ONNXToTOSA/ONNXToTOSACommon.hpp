/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToTOSACommon.hpp - ONNX dialects to TOSA lowering --------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Transform/ONNX/ConstPropHelper.hpp"

//===----------------------------------------------------------------------===//
// Functions to add lowering patterns for frontend operations.
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Check for valid TOSA types.
//===----------------------------------------------------------------------===//

inline bool isTOSASignedInt(Type type) {
  IntegerType intType = type.dyn_cast<IntegerType>();
  std::set<unsigned> intWidth{8, 16, 32, 48, 64};
  return intType && intType.isSigned() &&
         (intWidth.find(intType.getWidth()) != intWidth.end());
}

inline bool isTOSAFloat(Type type) {
  return type.isa<BFloat16Type, Float16Type, Float32Type>();
}

//===----------------------------------------------------------------------===//
// This is to get a TOSA operation of a given type for a specific operation.
//===----------------------------------------------------------------------===//
template <typename ONNXOp>
struct TOSADialectOp {
  using Op = void;
};

template <typename Op>
using TOSAOp = typename TOSADialectOp<Op>::Op;

// `Math` directory methods:
void populateLoweringONNXElementwiseOpToTOSAPattern(
    ConversionTarget &, RewritePatternSet &, TypeConverter &, MLIRContext *);
} // namespace onnx_mlir
