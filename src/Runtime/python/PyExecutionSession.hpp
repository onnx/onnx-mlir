/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ PyExecutionSession.hpp - PyExecutionSession Declaration -------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of PyExecutionSession class, which helps
// python programs interact with compiled binary model libraries.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_PY_EXECUTION_SESSION_H
#define ONNX_MLIR_PY_EXECUTION_SESSION_H

#include "PyExecutionSessionBase.hpp"

namespace onnx_mlir {

class PyExecutionSession : public onnx_mlir::PyExecutionSessionBase {
public:
  PyExecutionSession(std::string sharedLibPath, std::string tag = "",
      bool defaultEntryPoint = true);
};
} // namespace onnx_mlir

PYBIND11_MODULE(PyRuntimeC, m) {
  py::class_<onnx_mlir::PyExecutionSession>(m, "OMExecutionSession")
      .def(py::init<const std::string &, const std::string &, const bool>(),
          py::arg("shared_lib_path"), py::arg("tag") = "",
          py::arg("use_default_entry_point") = 1)
      .def("entry_points", &onnx_mlir::PyExecutionSession::pyQueryEntryPoints)
      .def("set_entry_point", &onnx_mlir::PyExecutionSession::pySetEntryPoint,
          py::arg("name"))
      .def("run", &onnx_mlir::PyExecutionSession::pyRun, py::arg("input"),
          py::arg("shape"), py::arg("stride"))
      .def("input_signature", &onnx_mlir::PyExecutionSession::pyInputSignature)
      .def("output_signature",
          &onnx_mlir::PyExecutionSession::pyOutputSignature);
}
#endif
