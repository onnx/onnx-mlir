//===- ONNXBufferizableOpInterface.cpp - ONNX Bufferizable Interface -----===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the BufferizableOpInterface for ONNX operations.
// This allows One-Shot Bufferization to work with mixed Linalg and ONNX
// operations in the same IR.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToLinalg/ONNXBufferizableOpInterface.hpp"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;
using namespace mlir::bufferization;
using namespace onnx_mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// ONNX BufferizableOpInterface External Model
//===----------------------------------------------------------------------===//

template <typename OpTy>
struct ONNXOpBufferizableInterface
    : public BufferizableOpInterface::ExternalModel<
          ONNXOpBufferizableInterface<OpTy>, OpTy> {

  // This operation reads from memory through its operands
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true; // ONNX operations typically read their inputs
  }

  // This operation writes to memory through its results
  bool bufferizesToMemoryWrite(Operation *op, OpResult opResult,
                               const AnalysisState &state) const {
    return true; // Results need to be written to memory
  }

  // This operation writes to memory through its operands (for in-place ops)
  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false; // ONNX ops don't write in-place through operands
  }

  // Conservative policy: no memory aliasing (outputs are newly allocated)
  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {}; // No aliasing - outputs are newly allocated
  }

  // Bufferization: ONNX operations are not converted here.
  // They remain as tensor operations and will be handled by Krnl later.
  // Return failure to let allowUnknownOps=true handle them.
  // This allows One-Shot Bufferize to skip these operations and handle
  // type conversions automatically via bufferization.to_tensor casts.
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    // Return failure to indicate this operation should be treated as unknown
    // and handled by allowUnknownOps=true option
    return failure();
  }
};

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Interface Registration
//===----------------------------------------------------------------------===//

void onnx_mlir::registerONNXBufferizableOpInterfaces(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, mlir::ONNXDialect *dialect) {
    // Register BufferizableOpInterface for ONNX operations used in examples
    // MatMul and Add are the most commonly used operations in test cases

    // MatMul operation (may remain as ONNX if not converted to Linalg)
    mlir::ONNXMatMulOp::attachInterface<
        ONNXOpBufferizableInterface<mlir::ONNXMatMulOp>>(*ctx);

    // Add operation (commonly used with MatMul in examples)
    mlir::ONNXAddOp::attachInterface<
        ONNXOpBufferizableInterface<mlir::ONNXAddOp>>(*ctx);
  });
}

