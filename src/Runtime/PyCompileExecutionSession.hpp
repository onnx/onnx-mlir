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
  PyCompileExecutionSession(
      std::string fileName, bool defaultEntryPoint = true);
  int64_t pyCompileFromFile(std::string flags);
  std::string pyGetCompiledFileName();
  std::string pyGetErrorMessage();

private:
  std::string inputFileName;
  std::string sharedLibPath;
  std::string errorMessage;
};
} // namespace onnx_mlir

PYBIND11_MODULE(PyCompileAndRuntime, m) {
  py::class_<onnx_mlir::PyCompileExecutionSession>(
      m, "PyCompileExecutionSession")
      .def(py::init<const std::string &>(), py::arg("file_name"))
      .def(py::init<const std::string &, const bool>(), py::arg("file_name"),
          py::arg("use_default_entry_point"))
      .def("compile_from_file",
          &onnx_mlir::PyCompileExecutionSession::pyCompileFromFile,
          py::arg("flags"))
      .def(
          "run", &onnx_mlir::PyCompileExecutionSession::pyRun, py::arg("input"))
      .def("input_signature",
          &onnx_mlir::PyCompileExecutionSession::pyInputSignature)
      .def("output_signature",
          &onnx_mlir::PyCompileExecutionSession::pyOutputSignature);
}