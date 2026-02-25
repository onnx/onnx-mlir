/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- PyOMCompilerSession.hpp - PyOMCompilerSession Declaration ------===//
//
//
// =============================================================================
//
// This file contains declaration of PyOMCompilerSession class, which
// helps python programs to compile and run binary model libraries.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_PY_OM_COMPILE_SESSION_BASE_H
#define ONNX_MLIR_PY_OM_COMPILE_SESSION_BASE_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "src/Compiler/OMCompilerSession.hpp"

namespace onnx_mlir {

class PyOMCompilerSession {
public:
  PyOMCompilerSession(std::string modelPath, std::string flags,
      const std::string &logFilename = {}, bool reuseCompiledModel = true);
  std::string pyGetOutputFilename();
  std::string pyGetModelTag();

private:
  onnx_mlir::CompilerSession compilerSession; // To compile a model.
  std::string modelPath;
  std::string flags;
};

} // namespace onnx_mlir

PYBIND11_MODULE(PyCompileC, m) {
  m.doc() = "OMCompilerSession enables users to compile an ONNX model "
            "in a python script.";
  py::class_<onnx_mlir::PyOMCompilerSession>(m, "OMCompilerSession")
      .def(py::init<const std::string &, const std::string &,
               const std::string &, const bool>(),
          py::arg("input_model_path"), py::arg("flags"),
          py::arg("log_filename") = "", py::arg("reuse_compiled_model") = 1)
      .def("get_output_file_name",
          &onnx_mlir::PyOMCompilerSession::pyGetOutputFilename)
      .def("get_model_tag", &onnx_mlir::PyOMCompilerSession::pyGetModelTag);
}
#endif
