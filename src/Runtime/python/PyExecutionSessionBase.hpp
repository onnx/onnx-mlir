/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- PyExecutionSessionBase.hpp - PyExecutionSessionBase Declaration ---===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of PyExecutionSessionBase class, which which
// contains shared code for PyExecutionSession and PyOMCompileExecutionSession.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "../ExecutionSession.hpp"

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
