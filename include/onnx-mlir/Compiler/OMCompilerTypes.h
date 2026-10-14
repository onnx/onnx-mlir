/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- OMCompilerTypes.h - C/C++ Neutral types -------------===//
//
// Copyright 2019-2026 The IBM Research Authors.
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
#define ONNX_MLIR_ERROR_ENUM(name, code, desc) name = code,
  ONNX_MLIR_COMPILER_ERROR_CODES(ONNX_MLIR_ERROR_ENUM)
#undef ONNX_MLIR_ERROR_ENUM
} OnnxMlirCompilerErrorCodes;

/* Generate error description strings array */
static const char *OnnxMlirCompilerErrorCodeDescriptions[] = {
#define ONNX_MLIR_ERROR_DESC(name, code, desc) desc,
    ONNX_MLIR_COMPILER_ERROR_CODES(ONNX_MLIR_ERROR_DESC)
#undef ONNX_MLIR_ERROR_DESC
};

/* Helper function to get error description */
static inline const char *getOnnxMlirCompilerErrorDescription(int code) {
  if (code >= 0 &&
      code < (int)(sizeof(OnnxMlirCompilerErrorCodeDescriptions) /
                   sizeof(OnnxMlirCompilerErrorCodeDescriptions[0]))) {
    return OnnxMlirCompilerErrorCodeDescriptions[code];
  }
  return "Unknown error code";
}

#ifdef __cplusplus
} // namespace onnx_mlir
#endif

#endif /* ONNX_MLIR_OM_COMPILER_TYPES_H */
