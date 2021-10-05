/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace onnx_mlir {

enum EmissionTargetType {
  EmitONNXBasic,
  EmitONNXIR,
  EmitMLIR,
  EmitLLVMIR,
  EmitLib,
  EmitJNI,
};

enum InputIRLevelType {
  ONNXLevel,
  MLIRLevel,
  LLVMLevel,
};

} // namespace onnx_mlir
