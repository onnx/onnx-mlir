/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ PyCompileExecutionSession.hpp - PyCompileExecutionSession
// Declaration -------===//
//
//
// =============================================================================
//
// This file contains declaration of PyCompileExecutionSession class, which
// helps python programs to compile and run binary model libraries.
//
//===----------------------------------------------------------------------===//

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "ExecutionSession.hpp"
#include "OnnxMlirCompiler.h"
#include "PyExecutionSession.hpp"

namespace onnx_mlir {

class PyCompileExecutionSession : public onnx_mlir::PyExecutionSession {
public:
  PyCompileExecutionSession(std::string inputFileName,
      std::string sharedLibPath, std::string flags,
      bool defaultEntryPoint = true);
  std::string pyGetCompiledFileName();
  std::string pyGetErrorMessage();
  int64_t pyGetCompiledResult();

private:
  std::string inputFileName;
  std::string sharedLibPath;
  std::string errorMessage;
  int64_t rc;
};
} // namespace onnx_mlir

PYBIND11_MODULE(PyCompileAndRuntime, m) {
  py::class_<onnx_mlir::PyCompileExecutionSession>(
      m, "PyCompileExecutionSession")
      .def(py::init<const std::string &, const std::string &,
               const std::string &>(),
          py::arg("input_model_path"), py::arg("compiled_file_path"),
          py::arg("flags"))
      .def(py::init<const std::string &, const std::string &,
               const std::string &, const bool>(),
          py::arg("input_model_path"), py::arg("compiled_file_path"),
          py::arg("flags"), py::arg("use_default_entry_point"))
      .def("get_compiled_result",
          &onnx_mlir::PyCompileExecutionSession::pyGetCompiledResult)
      .def("get_compiled_file_name",
          &onnx_mlir::PyCompileExecutionSession::pyGetCompiledFileName)
      .def("get_error_message",
          &onnx_mlir::PyCompileExecutionSession::pyGetErrorMessage);
}