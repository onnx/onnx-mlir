/*
 * SPDX-License-Identifier: Apache-2.0
 */

// hi alex, to deprecate

//===----- PyOMCompileSession.hpp - PyOMCompileSession Implementation
//-----===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementations of PyOMCompileSession class, which helps
// python programs interact with compiled binary model libraries.
//
//===----------------------------------------------------------------------===//

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "src/Support/SuppressWarnings.h"

SUPPRESS_WARNINGS_PUSH
#include "onnx/onnx_pb.h"
SUPPRESS_WARNINGS_POP

#include "PyOMCompileSession.hpp"

namespace onnx_mlir {
int64_t PyOMCompileSession::pyCompileFromFile(std::string flags) {
#if 0 // hi alex, deprecated
  if (this->inputFileName.empty()) {
    errorMessage =
        "No OMCompileSession was created with the input file name specified.";
    return -1;
  }
  char *outputName = nullptr;
  char *errorMsg = nullptr;
  int64_t rc;
  rc = omCompileFromFile(
      inputFileName.c_str(), flags.c_str(), &outputName, &errorMsg);
  if (rc == 0) {
    // Compilation success: save output file name.
    outputFileName = std::string(outputName);
    // Empty error.
    errorMessage = std::string();
  } else {
    // Compilation failure: save error message.
    errorMessage = std::string(errorMsg);
    // Empty output file name.
    outputFileName = std::string();
  }
  free(outputName);
  free(errorMsg);
  return rc;
#else
  assert(false && "deprecated");
  return -1;
#endif
}

int64_t PyOMCompileSession::pyCompileFromArray(
    std::string outputBaseName, EmissionTargetType emissionTarget) {
  if (this->inputBuffer == nullptr) {
    errorMessage =
        "No OMCompileSession was created with the input buffer specified.";
    return -1;
  }
  char *outputName = nullptr;
  char *errorMsg = nullptr;
  int64_t rc;
  rc = omCompileFromArray(inputBuffer, inputBufferSize, outputBaseName.c_str(),
      emissionTarget, &outputName, &errorMsg);
  if (rc == 0) {
    // Compilation success: save output file name.
    outputFileName = std::string(outputName);
    // Empty error.
    errorMessage = std::string();
  } else {
    // Compilation failure: save error message.
    errorMessage = std::string(errorMsg);
    // Empty output file name.
    outputFileName = std::string();
  }
  free(outputName);
  free(errorMsg);
  return rc;
}

std::string PyOMCompileSession::pyGetCompiledFileName() {
  return outputFileName;
}
std::string PyOMCompileSession::pyGetErrorMessage() { return errorMessage; }

} // namespace onnx_mlir