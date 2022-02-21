/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ PyExecutionSession.hpp - PyExecutionSession Declaration -------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of PyExecutionSession class, which helps
// python programs interact with compiled binary model libraries.
//
//===----------------------------------------------------------------------===//

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "ExecutionSession.hpp"

namespace onnx_mlir {

class PyExecutionSession : public onnx_mlir::ExecutionSession {
public:
  PyExecutionSession(std::string sharedLibPath)
      : onnx_mlir::ExecutionSession(sharedLibPath) {}
  PyExecutionSession(std::string sharedLibPath, std::string entryPointName)
      : onnx_mlir::ExecutionSession(sharedLibPath, entryPointName) {}

  std::vector<py::array> pyRun(const std::vector<py::array> &inputsPyArray);

  std::string pyInputSignature();
  std::string pyOutputSignature();
};
} // namespace onnx_mlir

PYBIND11_MODULE(PyRuntime, m) {
  py::class_<onnx_mlir::PyExecutionSession>(m, "ExecutionSession")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, const std::string &>())
      .def("run", &onnx_mlir::PyExecutionSession::pyRun)
      .def("input_signature", &onnx_mlir::PyExecutionSession::pyInputSignature)
      .def("output_signature",
          &onnx_mlir::PyExecutionSession::pyOutputSignature);
}
