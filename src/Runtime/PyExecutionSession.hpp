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

#pragma once

#include "PyExecutionSessionBase.hpp"

namespace onnx_mlir {

class PyExecutionSession : public onnx_mlir::PyExecutionSessionBase {
public:
  PyExecutionSession(std::string sharedLibPath, bool defaultEntryPoint = true);
};
} // namespace onnx_mlir

PYBIND11_MODULE(PyRuntime, m) {
  py::class_<onnx_mlir::PyExecutionSession>(m, "OMExecutionSession")
      .def(py::init<const std::string &>(), py::arg("shared_lib_path"))
      .def(py::init<const std::string &, const bool>(),
          py::arg("shared_lib_path"), py::arg("use_default_entry_point"))
      .def("entry_points", &onnx_mlir::PyExecutionSession::pyQueryEntryPoints)
      .def("set_entry_point", &onnx_mlir::PyExecutionSession::pySetEntryPoint,
          py::arg("name"))
      .def("run", &onnx_mlir::PyExecutionSession::pyRun, py::arg("input"))
      .def("input_signature", &onnx_mlir::PyExecutionSession::pyInputSignature)
      .def("output_signature",
          &onnx_mlir::PyExecutionSession::pyOutputSignature);
}