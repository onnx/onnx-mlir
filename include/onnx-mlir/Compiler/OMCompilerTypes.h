/*
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ONNX_MLIR_OMCOMPILERTYPES_H
#define ONNX_MLIR_OMCOMPILERTYPES_H

//#ifdef __cplusplus
namespace onnx_mlir {
//#endif

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
  TargetTriple = 0, /* Kind for mtriple string. */
  TargetArch,       /* Kind for march string. */
  TargetCPU,        /* Kind for mcpu string. */
  CompilerOptLevel, /* Kind for '0'...'3' string describing OptLevel. */
  LastOptionKind = CompilerOptLevel, /* last option */
};

//#ifdef _cplusplus
} /* namespace onnx_mlir */
//#else 
//typedef enum EmissionTargetType EmissionTargetType;
//typedef enum InputIRLevelType InputIRLevelType;
//typedef enum OptLevel OptLevel;
//typedef enum OptionKind OptionKind;
//#endif 

#endif
