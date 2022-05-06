/*
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ONNX_MLIR_OMCOMPILERTYPES_H
#define ONNX_MLIR_OMCOMPILERTYPES_H

#ifdef __cplusplus
namespace onnx_mlir {
#endif

/* Type of compiler emission targets */
typedef enum {
  EmitONNXBasic,
  EmitONNXIR,
  EmitMLIR,
  EmitLLVMIR,
  EmitObj,
  EmitLib,
  EmitJNI,
} EmissionTargetType;

/* Input IR can be at one of these levels */
typedef enum {
  ONNXLevel,
  MLIRLevel,
  LLVMLevel,
} InputIRLevelType;

/* Compiler optimization level (traditional -O0 ... -O3 flags) */
typedef enum { O0 = 0, O1, O2, O3 } OptLevel;

/* Compiler options to describe the architecture, optimization level,... */
typedef enum {
  TargetTriple,     /* Kind for mtriple string. */
  TargetArch,       /* Kind for march string. */
  TargetCPU,        /* Kind for mcpu string. */
  CompilerOptLevel, /* Kind for '0'...'3' string describing OptLevel. */
  OPTFlag,          /* Kind for -Xopt string. */
  LLCFlag,          /* Kind for -Xllc string. */
  LLVMFlag,         /* Kind for -mllvm string. */
} OptionKind;

/* Compiler options to describe instrumentation options */
typedef enum {
  InstrumentBeforeOp,
  InstrumentAfterOp,
  InstrumentReportTime,
  InstrumentReportMemory
} InstrumentActions;

#ifdef __cplusplus
} // namespace onnx_mlir
#endif

#endif /* ONNX_MLIR_OMCOMPILERTYPES_H */
