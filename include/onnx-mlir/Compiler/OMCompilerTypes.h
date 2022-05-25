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
  TargetAccel,      /* Kind for maccel string. */
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

/* Onnx Mlir Compiler return code on errors */
typedef enum {
  NoCompilerError = 0,     /* Zero is success. */
  InvalidCompilerOption,   /* Could not process given compiler option. */
  InvalidInputFile,        /* Got a file with an unexpected format. */
  InvalidInputFileAccess,  /* Could not successfully open input file. */
  InvalidOutputFileAccess, /* Could not successfully open output file. */
  InvalidOnnxFormat,       /* Could not successfully parse ONNX file. */
  CompilerFailure,         /* Failed to compile valid input file. */
} OnnxMlirCompilerErrorCodes;

#ifdef __cplusplus
} // namespace onnx_mlir
#endif

#endif /* ONNX_MLIR_OMCOMPILERTYPES_H */
