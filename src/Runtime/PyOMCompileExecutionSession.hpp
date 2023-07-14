/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ PyOMCompileExecutionSession.hpp - PyOMCompileExecutionSession
// Declaration -------===//
//
//
// =============================================================================
//
// This file contains declaration of PyOMCompileExecutionSession class, which
// helps python programs to compile and run binary model libraries.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "OnnxMlirCompiler.h"
#include "PyExecutionSessionBase.hpp"

namespace onnx_mlir {

class PyOMCompileExecutionSession : public onnx_mlir::PyExecutionSessionBase {
public:
  PyOMCompileExecutionSession(std::string inputFileName,
      std::string flags,
      bool defaultEntryPoint = true, bool reuseCompiledModel = true);
  std::string pyGetCompiledFileName();
  std::string pyGetErrorMessage();
  int64_t pyGetCompiledResult();

private:
  std::string inputFileName;
  std::string outputFileName;
  std::string errorMessage;
  int64_t rc;
};
} // namespace onnx_mlir

PYBIND11_MODULE(PyCompileAndRuntime, m) {
  py::class_<onnx_mlir::PyOMCompileExecutionSession>(
      m, "OMCompileExecutionSession")
      .def(py::init<const std::string &, const std::string &,
               const bool, const bool>(),
          py::arg("input_model_name"), py::arg("flags"), 
          py::arg("use_default_entry_point") = 1,
          py::arg("reuse_compiled_model") = 1)
      .def("get_compiled_result",
          &onnx_mlir::PyOMCompileExecutionSession::pyGetCompiledResult)
      .def("get_compiled_file_name",
          &onnx_mlir::PyOMCompileExecutionSession::pyGetCompiledFileName)
      .def("get_error_message",
          &onnx_mlir::PyOMCompileExecutionSession::pyGetErrorMessage)
      .def("entry_points",
          &onnx_mlir::PyOMCompileExecutionSession::pyQueryEntryPoints)
      .def("set_entry_point",
          &onnx_mlir::PyOMCompileExecutionSession::pySetEntryPoint,
          py::arg("name"))
      .def("run", &onnx_mlir::PyOMCompileExecutionSession::pyRun,
          py::arg("input"))
      .def("input_signature",
          &onnx_mlir::PyOMCompileExecutionSession::pyInputSignature)
      .def("output_signature",
          &onnx_mlir::PyOMCompileExecutionSession::pyOutputSignature);
}