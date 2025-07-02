/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====----------------------- ErrorHandling.cpp ---------------------------===//
//
// This file contains common error handling utilities for ONNX-MLIR.
//
//===----------------------------------------------------------------------===//

#include "ErrorHandling.hpp"
#include "llvm/Support/ErrorHandling.h"

namespace onnx_mlir {

namespace {

struct OnnxMlirCompilerErrorCategory : public std::error_category {
  const char *name() const noexcept override { return "ONNX-MLIR"; }

  std::string message(int ev) const override {
    switch (static_cast<OnnxMlirCompilerErrorCodes>(ev)) {
    case CompilerSuccess:
      return "Compiler succeeded";
    case InvalidCompilerOption:
      return "Could not process given compiler option";
    case InvalidInputFile:
      return "Got a file with an unexpected format";
    case InvalidInputFileAccess:
      return "Could not successfully open input file";
    case InvalidOutputFileAccess:
      return "Could not successfully open output file";
    case InvalidTemporaryFileAccess:
      return "Could not access a temporary file";
    case InvalidOnnxFormat:
      return "Could not successfully parse ONNX file";
    case CompilerFailureInMLIRToLLVM:
      return "Failed to lower MLIR to LLVM";
    case CompilerFailureInLLVMOpt:
      return "Failed running LLVM's optimizations";
    case CompilerFailureInLLVMToObj:
      return "Failed to lower LLVM to object file";
    case CompilerFailureInGenJniObj:
      return "Failed to lower object to JNI object";
    case CompilerFailureInGenJni:
      return "Failed to lower JNI object to JNI";
    case CompilerFailureInObjToLib:
      return "Failed to link object to a library";
    case CompilerFailure:
      return "Failed to compile valid input file";
    }
    llvm_unreachable("Unhandled error code");
  }
};

const OnnxMlirCompilerErrorCategory onnxMlirCompilerErrorCategoryInstance{};
} // namespace

std::error_code make_error_code(OnnxMlirCompilerErrorCodes e) {
  return {static_cast<int>(e), onnxMlirCompilerErrorCategoryInstance};
}

} // namespace onnx_mlir
