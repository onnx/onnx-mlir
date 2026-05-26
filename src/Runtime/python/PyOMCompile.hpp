/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- PyOMCompile.hpp - PyOMCompile Declaration --------===//
//
// Copyright 2024-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of PyOMCompile class, which
// helps python programs to compile and run binary model libraries.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_PY_OM_COMPILE_SESSION_H
#define ONNX_MLIR_PY_OM_COMPILE_SESSION_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "src/Compiler/OMCompile.hpp"

namespace onnx_mlir {

class PyOMCompile {
public:
  PyOMCompile(const std::string &modelPath, const std::string &flags,
      const std::string &compilerPath, const std::string &logFilename,
      bool reuseCompiledModel);
  std::string pyGetOutputFilename();
  std::string pyGetOutputConstantFilename();
  std::string pyGetModelTag();

private:
  onnx_mlir::OMCompile OMcompile; // To compile a model.
};

} // namespace onnx_mlir

// clang-format off
PYBIND11_MODULE(PyOMCompileC, m) {
  m.doc() = "PyOMCompile module provides Python bindings for compiling ONNX models.\n\n"
            "This module enables users to compile ONNX models using the onnx-mlir compiler\n"
            "from Python scripts. It provides a simple interface to invoke the compiler,\n"
            "manage compilation flags, and retrieve information about compiled models.\n";
  py::class_<onnx_mlir::PyOMCompile>(m, "OMCompile",
      "Compiler session for ONNX models.\n\n"
      "This class provides an interface to compile ONNX models and retrieve\n"
      "information about the compilation results. It wraps the onnx-mlir compiler\n"
      "and handles model compilation, caching, and output file management.\n\n"
      "Example:\n"
      "    >>> from PyCompile import OMCompile\n"
      "    >>> compiler = OMCompile('model.onnx', '-O3 -o output')\n"
      "    >>> output_file = compiler.get_output_file_name()\n"
      "    >>> print(f'Compiled to: {output_file}')")
      .def(py::init<const std::string &, const std::string &, const std::string &,
               const std::string &, bool>(),
          py::arg("input_model_path"),
          py::arg("flags"),
          py::arg("compiler_path") = "",
          py::arg("log_file_name") = "",
          py::arg("reuse_compiled_model") = false,
          "Compile an ONNX model.\n\n"
          "Args:\n"
          "    input_model_path (str): Path to the input ONNX model file (.onnx).\n"
          "        Must be a valid path to an existing ONNX or MLIR model.\n"
          "    flags (str): Compilation flags as a single string.\n"
          "        Examples: '-O3', '-O3 -o output_name', '--EmitLib'.\n"
          "        All onnx-mlir command-line options are supported.\n"
          "    compiler_path (str, optional): Path to onnx-mlir compiler binary,\n"
          "        namely path plus binary name. If empty (default), use onnx-mlir\n"
          "        at its default location.\n"
          "    log_file_name (str, optional): Path to log file for compilation output.\n"
          "        If empty (default), output goes to stdout/stderr.\n"
          "    reuse_compiled_model (bool, optional): If True, reuse existing compiled\n"
          "        model if it exists instead of recompiling. Default: False.\n\n"
          "Raises:\n"
          "    RuntimeError: If the model file doesn't exist, compilation fails,\n"
          "        or no input model is provided.\n\n"
          "Example:\n"
          "    >>> # Basic compilation\n"
          "    >>> compiler = OMCompile('mnist.onnx', '-O3')\n"
          "    >>> \n"
          "    >>> # With custom output name and optimization\n"
          "    >>> compiler = OMCompile('model.onnx', '-O3 -o my_model')\n"
          "    >>> \n"
          "    >>> # Force recompilation\n"
          "    >>> compiler = OMCompile('model.onnx', '-O3', \n"
          "    ...                             reuse_compiled_model=False)")
      .def("get_output_file_name",
          &onnx_mlir::PyOMCompile::pyGetOutputFilename,
          "Get the output filename of the compiled model.\n\n"
          "Returns the absolute path to the compiled model file. The filename is\n"
          "determined by the input model name and compilation flags (especially\n"
          "the '-o' flag if provided).\n\n"
          "Returns:\n"
          "    str: Full path to the compiled model output file.\n"
          "Raises:\n"
          "    RuntimeError: If the compilation failed\n\n"
          "Example:\n"
          "    >>> compiler = OMCompile('mnist.onnx', '-O3 -o mnist_opt')\n"
          "    >>> output = compiler.get_output_file_name()\n"
          "    >>> print(output)  # e.g., '/home/me/mnist_opt.so' on Linux")
      .def("get_output_constant_file_name",
          &onnx_mlir::PyOMCompile::pyGetOutputConstantFilename,
          "Get the output filename of the compiled model constant file, if any.\n\n"
          "If the compiler did generate a data constant file, return its\n"
          "absolute path; otherwise, return an emtpy string.\n\n"
          "Returns:\n"
          "    str: Full path to the constant file of the compiled model.\n"
          "Raises:\n"
          "    RuntimeError: If the compilation failed\n\n"
          "Example:\n"
          "    >>> compiler = OMCompile('mnist.onnx', '-O3 -o mnist_opt')\n"
          "    >>> output = compiler.get_output_constant_file_name()\n"
          "    >>> print(output)  # e.g., '/home/me/mnist_opt.constant.bin' on Linux")
      .def("get_model_tag",
          &onnx_mlir::PyOMCompile::pyGetModelTag,
          "Get the model tag for the compiled model.\n\n"
          "Returns a unique identifier/tag for the compiled model based on the\n"
          "compilation flags. This can be used for model identification and\n"
          "caching purposes.\n\n"
          "Returns:\n"
          "    str: Model tag string.\n"
          "Raises:\n"
          "    RuntimeError: If the compilation failed\n\n"
          "Example:\n"
          "    >>> compiler = OMCompile('model.onnx', '-O3 --tag=key_model')\n"
          "    >>> tag = compiler.get_model_tag()\n"
          "    >>> print(f'Model tag: {tag}') # e.g., `key_model`");
}
// clang-format off

#endif
