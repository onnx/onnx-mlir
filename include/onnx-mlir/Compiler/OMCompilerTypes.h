/*
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ONNX_MLIR_OMCOMPILERTYPES_H
#define ONNX_MLIR_OMCOMPILERTYPES_H

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

#endif
