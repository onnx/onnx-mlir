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

/* Onnx Mlir Compiler return code on errors; zero is success */
typedef enum {
  CompilerSuccess = 0,            /* Zero is success. */
  InvalidCompilerOption = 1,      /* Could not process given compiler option. */
  InvalidInputFile = 2,           /* Got a file with an unexpected format. */
  InvalidInputFileAccess = 3,     /* Could not successfully open input file. */
  InvalidOutputFileAccess = 4,    /* Could not successfully open output file. */
  InvalidTemporaryFileAccess = 5, /* Could not access a temporary file. */
  InvalidOnnxFormat = 6,          /* Could not successfully parse ONNX file. */
  CompilerFailureInMLIRToLLVM = 7, /* Failed to lower MLIR to LLVM */
  CompilerFailureInLLVMOpt = 8,    /* Failed to optimize LLVM */
  CompilerFailureInLLVMToObj = 9,  /* Failed to lower LLVM to obj */
  CompilerFailureInGenJniObj = 10, /* Failed to lower object to Jni object */
  CompilerFailureInGenJni = 11,    /* Failed to lower Jni object to Jni */
  CompilerFailureInObjToLib = 12,  /* Failed to link object to a library */
  CompilerFailure = 13,            /* Failed to compile valid input file. */
} OnnxMlirCompilerErrorCodes;

#ifdef __cplusplus
} // namespace onnx_mlir
#endif

#endif /* ONNX_MLIR_OMCOMPILERTYPES_H */
