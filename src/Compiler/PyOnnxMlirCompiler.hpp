/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ PyOnnxMirCompiler.hpp - PyOnnxMirCompiler Declaration -------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of PyOnnxMirCompiler class, which helps
// python programs compile onnx programs into executable binaries.
//
//===----------------------------------------------------------------------===//

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "OnnxMlirCompiler.h"

namespace onnx_mlir {

class PyOnnxMirCompiler {
public:
  PyOnnxMirCompiler(std::string fileName) : inputFileName(fileName) {}
  PyOnnxMirCompiler(void *inputBuffer, int64_t bufferSize)
      : inputFileName(), inputBuffer(inputBuffer), inputBufferSize(bufferSize) {
  }

  // Options
  int64_t pySetOptionsFromEnv(std::string envVarName);
  int64_t pySetOption(const OptionKind kind, std::string val);
  void pyClearOption(const OptionKind kind);
  std::string pyGetOption(const OptionKind kind);

  int64_t pyCompile(
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

PYBIND11_MODULE(PyOnnxMlirCompiler, m) {
  py::class_<onnx_mlir::PyOnnxMirCompiler>(m, "OnnxMlirCompiler")
      .def(py::init<std::string &>(), py::arg("file_name"))
      .def(py::init<void *, int64_t>(), py::arg("input_buffer"),
          py::arg("buffer_size"))
      .def("set_option_from_env",
          &onnx_mlir::PyOnnxMirCompiler::pySetOptionsFromEnv,
          py::arg("env_var_ame"))
      .def("set_option", &onnx_mlir::PyOnnxMirCompiler::pySetOption,
          py::arg("kind"), py::arg("val"))
      .def("clear_option", &onnx_mlir::PyOnnxMirCompiler::pyClearOption,
          py::arg("kind"))
      .def("get_option", &onnx_mlir::PyOnnxMirCompiler::pyGetOption,
          py::arg("kind"))
      .def("compile", &onnx_mlir::PyOnnxMirCompiler::pyCompile,
          py::arg("output_base_name"), py::arg("target"))
      .def("get_output_file_name",
          &onnx_mlir::PyOnnxMirCompiler::pyGetCompiledFileName)
      .def("get_error_message",
          &onnx_mlir::PyOnnxMirCompiler::pyGetErrorMessage);
  py::enum_<onnx_mlir::EmissionTargetType>(m, "OnnxMlirTarget")
      .value("emit_onnx_basic", onnx_mlir::EmissionTargetType::EmitONNXBasic)
      .value("emit_onnxir", onnx_mlir::EmissionTargetType::EmitONNXIR)
      .value("emit_mlir", onnx_mlir::EmissionTargetType::EmitMLIR)
      .value("emit_llvmir", onnx_mlir::EmissionTargetType::EmitLLVMIR)
      .value("emit_obj", onnx_mlir::EmissionTargetType::EmitObj)
      .value("emit_lib", onnx_mlir::EmissionTargetType::EmitLib)
      .value("emit_jni", onnx_mlir::EmissionTargetType::EmitJNI)
      .export_values();
  py::enum_<onnx_mlir::OptionKind>(m, "OnnxMlirOption")
      .value("target_triple", onnx_mlir::OptionKind::TargetTriple)
      .value("target_arch", onnx_mlir::OptionKind::TargetArch)
      .value("target_cpu", onnx_mlir::OptionKind::TargetCPU)
      .value("target_accel", onnx_mlir::OptionKind::TargetAccel)
      .value("opt_level", onnx_mlir::OptionKind::CompilerOptLevel)
      .value("opt_flag", onnx_mlir::OptionKind::OPTFlag)
      .value("llc_flag", onnx_mlir::OptionKind::LLCFlag)
      .value("llvm_flag", onnx_mlir::OptionKind::LLVMFlag)
      .value("verbose", onnx_mlir::OptionKind::Verbose)
      .export_values();
}