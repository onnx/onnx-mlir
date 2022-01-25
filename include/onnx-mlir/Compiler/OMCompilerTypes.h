/*
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ONNX_MLIR_OMCOMPILERTYPES_H
#define ONNX_MLIR_OMCOMPILERTYPES_H

namespace onnx_mlir {

/* Type of compiler emission targets */
enum EmissionTargetType {
  EmitONNXBasic,
  EmitONNXIR,
  EmitMLIR,
  EmitLLVMIR,
  EmitObj,
  EmitLib,
  EmitJNI,
};

/* Input IR can be at one of these levels */
enum InputIRLevelType {
  ONNXLevel,
  MLIRLevel,
  LLVMLevel,
};

/* Compiler optimization level (traditional -O0 ... -O3 flags) */
enum OptLevel {
  O0 = 0,
  O1,
  O2,
  O3
};

/* Compiler options to describe the architecture, optimization level,... */
enum OptionKind {
  TargetTriple,     /* Kind for mtriple string. */
  TargetArch,       /* Kind for march string. */
  TargetCPU,        /* Kind for mcpu string. */
  CompilerOptLevel, /* Kind for '0'...'3' string describing OptLevel. */
};

} // namespace onnx_mlir

#endif
