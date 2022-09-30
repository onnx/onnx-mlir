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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "ExecutionSession.hpp"
#include "OnnxMlirCompiler.h"

namespace onnx_mlir {

class PyOMCompileExecutionSession : public onnx_mlir::ExecutionSession {
public:
  PyOMCompileExecutionSession(std::string inputFileName,
      std::string sharedLibPath, std::string flags,
      bool defaultEntryPoint = true);
  std::string pyGetCompiledFileName();
  std::string pyGetErrorMessage();
  int64_t pyGetCompiledResult();
  std::vector<std::string> pyQueryEntryPoints();
  void pySetEntryPoint(std::string entryPointName);
  std::vector<py::array> pyRun(const std::vector<py::array> &inputsPyArray);
  std::string pyInputSignature();
  std::string pyOutputSignature();

private:
  std::string inputFileName;
  std::string sharedLibPath;
  std::string errorMessage;
  int64_t rc;
};
} // namespace onnx_mlir

PYBIND11_MODULE(PyCompileAndRuntime, m) {
  py::class_<onnx_mlir::PyOMCompileExecutionSession>(
      m, "PyOMCompileExecutionSession")
      .def(py::init<const std::string &, const std::string &,
               const std::string &>(),
          py::arg("input_model_path"), py::arg("compiled_file_path"),
          py::arg("flags"))
      .def(py::init<const std::string &, const std::string &,
               const std::string &, const bool>(),
          py::arg("input_model_path"), py::arg("compiled_file_path"),
          py::arg("flags"), py::arg("use_default_entry_point"))
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