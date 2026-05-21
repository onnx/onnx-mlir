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
  // Constructor for local compilation
  PyOMCompile(const std::string &compilerPath = {}, bool verbose = false);

  // Constructor for container-based compilation
  PyOMCompile(const std::string &containerImage,
      const std::string &compilerPathInContainer,
      const std::string &engine = "auto", bool autoPull = true,
      bool verbose = false);

  // Compile method
  void compile(const std::string &modelPath, const std::string &flags,
      const std::string &compilerPath = {},
      const std::string &logFilename = {});

  std::string pyGetOutputFilename();
  std::string pyGetOutputConstantFilename();
  std::string pyGetModelTag();
  bool pyIsSuccessfullyCompiled();
  bool pyHasOutputConstantFilename();

private:
  onnx_mlir::OMCompile OMcompile; // To compile a model.
};

} // namespace onnx_mlir

// clang-format off
PYBIND11_MODULE(PyOMCompileC, m) {
  m.doc() = "PyOMCompile module provides Python bindings for compiling ONNX models.\n\n"
            "This module enables users to compile ONNX models using the onnx-mlir compiler\n"
            "from Python scripts. It provides a simple interface to invoke the compiler,\n"
            "manage compilation flags, and retrieve information about compiled models.\n"
            "Supports both local and container-based compilation.\n";
  py::class_<onnx_mlir::PyOMCompile>(m, "OMCompile",
      "Compiler session for ONNX models.\n\n"
      "This class provides an interface to compile ONNX models and retrieve\n"
      "information about the compilation results. It wraps the onnx-mlir compiler\n"
      "and supports both local and container-based compilation.\n\n"
      "Example:\n"
      "    >>> from PyCompile import OMCompile\n"
      "    >>> compiler = OMCompile()  # Local compilation\n"
      "    >>> compiler.compile('model.onnx', '-O3 -o output')\n"
      "    >>> output_file = compiler.get_output_file_name()\n"
      "    >>> print(f'Compiled to: {output_file}')")
      .def(py::init<const std::string &, bool>(),
          py::arg("compiler_path") = "",
          py::arg("verbose") = false,
          "Create a compiler session for local compilation.\n\n"
          "Args:\n"
          "    compiler_path (str, optional): Path to onnx-mlir compiler binary.\n"
          "        If empty (default), use onnx-mlir from PATH.\n"
          "    verbose (bool, optional): Enable verbose output. Default: False.\n\n"
          "Example:\n"
          "    >>> # Use default compiler from PATH\n"
          "    >>> compiler = OMCompile()\n"
          "    >>> \n"
          "    >>> # Use specific compiler\n"
          "    >>> compiler = OMCompile('/path/to/onnx-mlir')\n"
          "    >>> \n"
          "    >>> # With verbose output\n"
          "    >>> compiler = OMCompile('', True)")
      .def(py::init<const std::string &, const std::string &, const std::string &,
               bool, bool>(),
          py::arg("container_image"),
          py::arg("compiler_path_in_container"),
          py::arg("engine") = "auto",
          py::arg("auto_pull") = true,
          py::arg("verbose") = false,
          "Create a compiler session for container-based compilation.\n\n"
          "Args:\n"
          "    container_image (str): Container image name (e.g., 'ghcr.io/onnxmlir/onnx-mlir').\n"
          "    compiler_path_in_container (str): Path to compiler inside the container.\n"
          "    engine (str, optional): Container engine to use ('docker', 'podman', or 'auto').\n"
          "        Default: 'auto' (auto-detect).\n"
          "    auto_pull (bool, optional): Automatically pull missing images. Default: True.\n"
          "    verbose (bool, optional): Enable verbose output. Default: False.\n\n"
          "Example:\n"
          "    >>> # Use containerized compiler\n"
          "    >>> compiler = OMCompile(\n"
          "    ...     'ghcr.io/onnxmlir/onnx-mlir',\n"
          "    ...     '/workdir/onnx-mlir/build/Debug/bin/onnx-mlir'\n"
          "    ... )\n"
          "    >>> \n"
          "    >>> # With specific engine\n"
          "    >>> compiler = OMCompile('image', 'path', 'docker')")
      .def("compile",
          &onnx_mlir::PyOMCompile::compile,
          py::arg("model_path"),
          py::arg("flags"),
          py::arg("compiler_path") = "",
          py::arg("log_file_name") = "",
          "Compile an ONNX model with specified flags.\n\n"
          "Args:\n"
          "    model_path (str): Path to the input ONNX model file (.onnx, .mlir, or .onnxtext).\n"
          "    flags (str): Compilation flags as a single string.\n"
          "        Examples: '-O3', '-O3 -o output_name', '--EmitLib'.\n"
          "        All onnx-mlir command-line options are supported.\n"
          "    compiler_path (str, optional): Path to onnx-mlir compiler binary.\n"
          "        Only used in local mode. If empty (default), uses the path from constructor.\n"
          "    log_file_name (str, optional): Path to log file for compilation output.\n"
          "        If empty (default), output goes to stdout/stderr.\n\n"
          "Raises:\n"
          "    RuntimeError: If compilation fails.\n\n"
          "Example:\n"
          "    >>> compiler = OMCompile()\n"
          "    >>> compiler.compile('mnist.onnx', '-O3')\n"
          "    >>> \n"
          "    >>> # With custom output name\n"
          "    >>> compiler.compile('model.onnx', '-O3 -o my_model')")
      .def("get_output_file_name",
          &onnx_mlir::PyOMCompile::pyGetOutputFilename,
          "Get the output filename of the compiled model.\n\n"
          "Returns the absolute path to the compiled model file. The filename is\n"
          "determined by the input model name and compilation flags (especially\n"
          "the '-o' flag if provided).\n\n"
          "Returns:\n"
          "    str: Full path to the compiled model output file.\n"
          "Raises:\n"
          "    RuntimeError: If called before a successful compilation.\n\n"
          "Example:\n"
          "    >>> compiler = OMCompile()\n"
          "    >>> compiler.compile('mnist.onnx', '-O3 -o mnist_opt')\n"
          "    >>> output = compiler.get_output_file_name()\n"
          "    >>> print(output)  # e.g., '/home/me/mnist_opt.so' on Linux")
      .def("get_output_constant_file_name",
          &onnx_mlir::PyOMCompile::pyGetOutputConstantFilename,
          "Get the output filename of the compiled model constant file, if any.\n\n"
          "If the compiler did generate a data constant file, return its\n"
          "absolute path; otherwise, return an empty string.\n\n"
          "Returns:\n"
          "    str: Full path to the constant file of the compiled model, or empty string.\n"
          "Raises:\n"
          "    RuntimeError: If called before a successful compilation.\n\n"
          "Example:\n"
          "    >>> compiler = OMCompile()\n"
          "    >>> compiler.compile('mnist.onnx', '-O3 -o mnist_opt')\n"
          "    >>> output = compiler.get_output_constant_file_name()\n"
          "    >>> print(output)  # e.g., '/home/me/mnist_opt.constant.bin' on Linux")
      .def("get_model_tag",
          &onnx_mlir::PyOMCompile::pyGetModelTag,
          "Get the model tag for the compiled model.\n\n"
          "Returns the tag specified via the --tag flag during compilation, or an\n"
          "empty string if no tag was specified.\n\n"
          "Returns:\n"
          "    str: Model tag string, or empty if no tag was set.\n"
          "Raises:\n"
          "    RuntimeError: If called before a successful compilation.\n\n"
          "Example:\n"
          "    >>> compiler = OMCompile()\n"
          "    >>> compiler.compile('model.onnx', '-O3 --tag=key_model')\n"
          "    >>> tag = compiler.get_model_tag()\n"
          "    >>> print(f'Model tag: {tag}')  # e.g., 'key_model'")
      .def("is_successfully_compiled",
          &onnx_mlir::PyOMCompile::pyIsSuccessfullyCompiled,
          "Check if the last compilation was successful.\n\n"
          "Returns:\n"
          "    bool: True if compile() completed successfully, False otherwise.\n\n"
          "Example:\n"
          "    >>> compiler = OMCompile()\n"
          "    >>> compiler.compile('model.onnx', '-O3')\n"
          "    >>> if compiler.is_successfully_compiled():\n"
          "    ...     print('Compilation succeeded!')")
      .def("has_output_constant_file_name",
          &onnx_mlir::PyOMCompile::pyHasOutputConstantFilename,
          "Check if the compiled model has an associated constant file.\n\n"
          "Returns:\n"
          "    bool: True if the output includes a constant file, False otherwise.\n\n"
          "Example:\n"
          "    >>> compiler = OMCompile()\n"
          "    >>> compiler.compile('model.onnx', '-O3')\n"
          "    >>> if compiler.has_output_constant_file_name():\n"
          "    ...     const_file = compiler.get_output_constant_file_name()\n"
          "    ...     print(f'Constant file: {const_file}')");
}
// clang-format off

#endif
