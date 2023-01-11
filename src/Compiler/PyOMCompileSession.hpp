/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ PyOMCompileSession.hpp - PyOMCompileSession Declaration -------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of PyOMCompileSession class, which helps
// python programs compile onnx programs into executable binaries.
//
//===----------------------------------------------------------------------===//

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "OnnxMlirCompiler.h"

namespace onnx_mlir {

class PyOMCompileSession {
public:
  PyOMCompileSession(std::string fileName) : inputFileName(fileName) {}
  PyOMCompileSession(void *inputBuffer, int64_t bufferSize)
      : inputFileName(), inputBuffer(inputBuffer), inputBufferSize(bufferSize) {
  }

  int64_t pyCompileFromFile(std::string flags);
  int64_t pyCompileFromArray(
      std::string outputBaseName, EmissionTargetType emissionTarget);
  std::string pyGetCompiledFileName();
  std::string pyGetErrorMessage();

private:
  std::string inputFileName;
  std::string outputFileName;
  std::string errorMessage;
  void *inputBuffer = nullptr;
  int64_t inputBufferSize = 0;
};
} // namespace onnx_mlir

PYBIND11_MODULE(PyCompile, m) {
  py::class_<onnx_mlir::PyOMCompileSession>(m, "OMCompileSession")
      .def(py::init<std::string &>(), py::arg("file_name"))
      .def(py::init<void *, int64_t>(), py::arg("input_buffer"),
          py::arg("buffer_size"))
      .def("compile", &onnx_mlir::PyOMCompileSession::pyCompileFromFile,
          py::arg("flags"))
      .def("compile_from_array",
          &onnx_mlir::PyOMCompileSession::pyCompileFromArray,
          py::arg("output_base_name"), py::arg("target"))
      .def("get_compiled_file_name",
          &onnx_mlir::PyOMCompileSession::pyGetCompiledFileName)
      .def("get_error_message",
          &onnx_mlir::PyOMCompileSession::pyGetErrorMessage);
  py::enum_<onnx_mlir::EmissionTargetType>(m, "OnnxMlirTarget")
      .value("emit_onnx_basic", onnx_mlir::EmissionTargetType::EmitONNXBasic)
      .value("emit_onnxir", onnx_mlir::EmissionTargetType::EmitONNXIR)
      .value("emit_mlir", onnx_mlir::EmissionTargetType::EmitMLIR)
      .value("emit_llvmir", onnx_mlir::EmissionTargetType::EmitLLVMIR)
      .value("emit_obj", onnx_mlir::EmissionTargetType::EmitObj)
      .value("emit_lib", onnx_mlir::EmissionTargetType::EmitLib)
      .value("emit_jni", onnx_mlir::EmissionTargetType::EmitJNI)
      .export_values();
}