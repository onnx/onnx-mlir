/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- PyOMCompileExecutionSession.cpp - PyOMCompileExecutionSession
// Implementation -----===//
//
//
// =============================================================================
//
// This file contains implementations of PyOMCompileExecutionSession class,
// which helps python programs to compile and run binary model libraries.
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

#include "PyOMCompileExecutionSession.hpp"

namespace onnx_mlir {

PyOMCompileExecutionSession::PyOMCompileExecutionSession(
    std::string inputFileName, std::string flags, bool defaultEntryPoint) {
  this->inputFileName = inputFileName;
  int64_t rc;
  this->pyOMCompileSession =
      new onnx_mlir::PyOMCompileSession(this->inputFileName);
  rc = pyOMCompileSession->pyCompileFromFile(flags);
  if (rc == 0) {
    // Compilation success: save output file name.
    this->sharedLibPath = this->pyOMCompileSession->pyGetCompiledFileName();
    // Empty error.
    errorMessage = std::string();
  } else {
    // Compilation failure: save error message.
    errorMessage = this->pyOMCompileSession->pyGetErrorMessage();
    // Empty output file name.
    this->sharedLibPath = std::string();
  }
  this->executionSession =
      new onnx_mlir::PyExecutionSession(sharedLibPath, defaultEntryPoint);
}

int64_t PyOMCompileExecutionSession::pyGetCompiledResult() { return this->rc; }

std::string PyOMCompileExecutionSession::pyGetCompiledFileName() {
  return this->sharedLibPath;
}

std::string PyOMCompileExecutionSession::pyGetErrorMessage() {
  return this->errorMessage;
}

std::vector<py::array> PyOMCompileExecutionSession::pyRun(
    const std::vector<py::array> &inputsPyArray) {
  return this->executionSession->pyRun(inputsPyArray);
}

void PyOMCompileExecutionSession::pySetEntryPoint(std::string entryPointName) {
  this->executionSession->pySetEntryPoint(entryPointName);
}

std::vector<std::string> PyOMCompileExecutionSession::pyQueryEntryPoints() {
  return this->executionSession->pyQueryEntryPoints();
}

std::string PyOMCompileExecutionSession::pyInputSignature() {
  return this->executionSession->pyInputSignature();
}

std::string PyOMCompileExecutionSession::pyOutputSignature() {
  return this->executionSession->pyOutputSignature();
}

} // namespace onnx_mlir