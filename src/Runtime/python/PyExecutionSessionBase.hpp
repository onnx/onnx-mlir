/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- PyExecutionSessionBase.hpp - PyExecutionSessionBase Declaration ---===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of PyExecutionSessionBase class, which which
// contains shared code for PyExecutionSession and PyOMCompileExecutionSession.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_PY_EXECUTION_SESSION_BASE_H
#define ONNX_MLIR_PY_EXECUTION_SESSION_BASE_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "src/Runtime/ExecutionSession.hpp"

namespace onnx_mlir {

#if !defined(WIN32) && !defined(_WIN32)
class PYBIND11_EXPORT PyExecutionSessionBase
#else
class PyExecutionSessionBase
#endif
    : public onnx_mlir::ExecutionSession {
public:
  PyExecutionSessionBase(std::string sharedLibPath, std::string tag = "",
      bool defaultEntryPoint = true);
  std::vector<std::string> pyQueryEntryPoints();
  void pySetEntryPoint(std::string entryPointName);
  // pyRun expects a vector of Python numpy.ndarray objects as the first
  // argument, a vector of shapes of the objects as the second argument, and a
  // vector of strides of the object as the third argument. All pyRun arguments
  // should have the same length, otherwise python exceptions occur.
  std::vector<py::array> pyRun(const std::vector<py::array> &inputsPyArray,
      const std::vector<py::array> &shapesPyArray,
      const std::vector<py::array> &stridesPyArray);
  std::string pyInputSignature();
  std::string pyOutputSignature();

protected:
  // Constructor that build the object without initialization (for use by
  // subclass only).
  PyExecutionSessionBase() : onnx_mlir::ExecutionSession() {}
  std::string reportPythonError(std::string errorStr) const;
};
} // namespace onnx_mlir
#endif
