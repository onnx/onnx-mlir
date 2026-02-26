/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- PyOMCompileSession.hpp - PyOMCompileSession Declaration --------===//
//
// Copyright 2024-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of PyOMCompileSession class, which
// helps python programs to compile and run binary model libraries.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_PY_OM_COMPILE_SESSION_BASE_H
#define ONNX_MLIR_PY_OM_COMPILE_SESSION_BASE_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "src/Compiler/OMCompileSession.hpp"

namespace onnx_mlir {

class PyOMCompileSession {
public:
  PyOMCompileSession(std::string modelPath, std::string flags,
      const std::string &logFilename = {}, bool reuseCompiledModel = true);
  std::string pyGetOutputFilename();
  std::string pyGetModelTag();

private:
  onnx_mlir::CompilerSession compilerSession; // To compile a model.
  std::string modelPath;
  std::string flags;
};

} // namespace onnx_mlir

PYBIND11_MODULE(PyOMCompileC, m) {
  m.doc() = "OMCompileSession enables users to compile an ONNX model "
            "in a python script.";
  py::class_<onnx_mlir::PyOMCompileSession>(m, "OMCompileSession")
      .def(py::init<const std::string &, const std::string &,
               const std::string &, const bool>(),
          py::arg("input_model_path"), py::arg("flags"),
          py::arg("log_filename") = "", py::arg("reuse_compiled_model") = 1)
      .def("get_output_file_name",
          &onnx_mlir::PyOMCompileSession::pyGetOutputFilename)
      .def("get_model_tag", &onnx_mlir::PyOMCompileSession::pyGetModelTag);
}
#endif
