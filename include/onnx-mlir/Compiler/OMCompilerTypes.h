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
  EmitObj,
  EmitLib,
  EmitJNI,
};

enum InputIRLevelType {
  ONNXLevel,
  MLIRLevel,
  LLVMLevel,
};

enum OptLevel {
  O0 = 0,
  O1,
  O2,
  O3
};

} // namespace onnx_mlir

#endif
