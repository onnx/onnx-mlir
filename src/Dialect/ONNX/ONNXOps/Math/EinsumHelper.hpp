/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ONNXEinsumOpHelper.hpp - Helper functions for Einsum --------===//
//
// This file contains helper functions for processing the ONNX Einsum Operator.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_EINSUM_HELPER_H
#define ONNX_MLIR_EINSUM_HELPER_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <stdint.h>

namespace mlir {
class InFlightDiagnostic;
class ONNXEinsumOpAdaptor;
} // namespace mlir

namespace onnx_mlir {

namespace einsum {

// Beware that this is only a reference and cannot store a lambda. Don't do:
//
//   einsum::ErrorFn fn = [this]() { return this->emitOpError(".."); };
//
// Instead do:
//
//   auto fn = [this]() { .. };
//
// or don't name it:
//
//   verifyEquation(eqn, n, [this]() { ... })
//
typedef llvm::function_ref<mlir::InFlightDiagnostic()> ErrorFn;

mlir::LogicalResult verifyEquation(
    llvm::StringRef equation, size_t numInputs, ErrorFn emitErrorFn);

mlir::LogicalResult verifyShapes(
    mlir::ONNXEinsumOpAdaptor operandAdaptor, ErrorFn emitErrorFn);

typedef llvm::SmallVector<int64_t, 4> Shape;

mlir::FailureOr<Shape> inferOutputShape(
    mlir::ONNXEinsumOpAdaptor operandAdaptor, ErrorFn emitErrorFn);

typedef llvm::SmallString<4> Subscripts;

struct Parameter {
  Shape shape;
  Subscripts subscripts;
};

struct Signature {
  llvm::SmallVector<Parameter> inputs;
  Parameter output;
};

mlir::FailureOr<Signature> inferSignature(
    mlir::ONNXEinsumOpAdaptor operandAdaptor, ErrorFn emitErrorFn);

} // namespace einsum

} // namespace onnx_mlir
#endif