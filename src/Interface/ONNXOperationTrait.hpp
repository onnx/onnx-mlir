/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- ONNXOperationTrait.hpp ----------------------===//
//
// Declares ONNXOperationTrait.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/OpDefinition.h"

namespace mlir {

/// A trait that specifies an ONNX operation type's name and version.
/// Assumes ConcreteType has static onnxName, onnxDomain, onnxSinceVersion
/// fields.
template <typename OP>
class ONNXOperationTrait : public OpTrait::TraitBase<OP, ONNXOperationTrait> {
public:
  static StringRef getONNXName() { return OP::onnxName; }
  static StringRef getONNXDomain() { return OP::onnxDomain; }
  static int getONNXSinceVersion() { return OP::onnxSinceVersion; }
};

} // namespace mlir
