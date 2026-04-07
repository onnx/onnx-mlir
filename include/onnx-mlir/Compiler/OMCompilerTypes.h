/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- OMCompilerTypes.h - C/C++ Neutral types -------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains types that are shared between in the compiler.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_OM_COMPILER_TYPES_H
#define ONNX_MLIR_OM_COMPILER_TYPES_H

#ifdef __cplusplus
namespace onnx_mlir {
#endif

/* Type of compiler emission targets */
/* Keep in sync with enumeration in PyOnnxMirCompiler.hpp python module. */
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
/* Onnx Mlir Compiler return code on errors; zero is success */
/* Define error codes with descriptions using X-macro pattern */
#define ONNX_MLIR_COMPILER_ERROR_CODES(X)                                      \
  X(CompilerSuccess, 0, "Success")                                             \
  X(InvalidCompilerOption, 1, "Invalid compiler option")                       \
  X(InvalidInputFile, 2, "Invalid input file format")                          \
  X(InvalidInputFileAccess, 3, "Cannot open input file")                       \
  X(InvalidOutputFileAccess, 4, "Cannot open output file")                     \
  X(InvalidTemporaryFileAccess, 5, "Cannot access temporary file")             \
  X(InvalidOnnxFormat, 6, "Invalid ONNX format")                               \
  X(CompilerFailureInMLIRToLLVM, 7, "Failed to lower MLIR to LLVM")            \
  X(CompilerFailureInLLVMOpt, 8, "Failed to optimize LLVM")                    \
  X(CompilerFailureInLLVMToObj, 9, "Failed to lower LLVM to object")           \
  X(CompilerFailureInGenJniObj, 10, "Failed to generate JNI object")           \
  X(CompilerFailureInGenJni, 11, "Failed to generate JNI")                     \
  X(CompilerFailureInObjToLib, 12, "Failed to link object to library")         \
  X(InvalidCompilerOptions, 13, "Invalid compiler options")                    \
  X(CompilerFailure, 14, "Compilation failed")                                 \
  X(CompilerCrashed, 15, "Compiler failed to execute successfully")            \
  X(CommandNotFound, 16, "Command executable not found (check PATH)")          \
  X(CommandNotExecutable, 17, "Command not executable")                        \
  X(CommandExecutionFailed, 18, "Command execution failed")                    \
  /* TODO: Remove after onnx 1.21.0 update (see                             */ \
  /*   https://github.com/onnx/onnx-mlir/issues/3455).                      */ \
  X(InvalidInputFileLink, 19, "Input file is a hardlink, which is not allowed")

/* Generate enum from the macro */
typedef enum {
  ONNXLevel,
  MLIRLevel,
  LLVMLevel,
} InputIRLevelType;

/* Compiler optimization level (traditional -O0 ... -O3 flags) */
typedef enum { O0 = 0, O1, O2, O3 } OptLevel;

/* Compiler options to describe the architecture, optimization level,... */
/* Keep in sync with enumeration in PyOnnxMirCompiler.hpp python module. */
typedef enum {
  TargetTriple,     /* Kind for mtriple string. */
  TargetArch,       /* Kind for march string. */
  TargetCPU,        /* Kind for mcpu string. */
  TargetAccel,      /* Kind for maccel string. */
  CompilerOptLevel, /* Kind for '0'...'3' string describing OptLevel. */
  OPTFlag,          /* Kind for -Xopt string. */
  LLCFlag,          /* Kind for -Xllc string. */
  LLVMFlag,         /* Kind for -mllvm string. */
  ModelTag,         /* Kind for tag string. */
  Verbose,          /* Kind for enabling -v verbose mode (boolean option)*/
} OptionKind;

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

#endif /* ONNX_MLIR_OM_COMPILER_TYPES_H */
