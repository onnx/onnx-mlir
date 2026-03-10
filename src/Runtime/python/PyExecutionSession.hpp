/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ PyExecutionSession.hpp - PyExecutionSession Declaration -------===//
//
// Copyright 2019-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of PyExecutionSession class, which helps
// python programs interact with compiled binary model libraries.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_PY_EXECUTION_SESSION_H
#define ONNX_MLIR_PY_EXECUTION_SESSION_H

#include "PyExecutionSessionBase.hpp"

namespace onnx_mlir {

class PyExecutionSession : public onnx_mlir::PyExecutionSessionBase {
public:
  PyExecutionSession(std::string sharedLibPath, std::string tag = "",
      bool defaultEntryPoint = true);
};
} // namespace onnx_mlir

// clang-format off
PYBIND11_MODULE(PyRuntimeC, m) {
  m.doc() = "PyRuntimeC module provides Python bindings for executing compiled ONNX models.\n\n"
            "This module enables users to load and run compiled ONNX models (shared libraries)\n"
            "from Python scripts. It provides interfaces for model inference, querying model\n"
            "signatures, and managing multiple entry points in compiled models.";
  py::class_<onnx_mlir::PyExecutionSession>(m, "OMExecutionSession",
      "Execution session for running compiled ONNX models.\n\n"
      "This class provides an interface to load compiled ONNX model libraries\n"
      "(shared objects) and execute inference on them. It handles model loading,\n"
      "input/output management, and supports models with multiple entry points.\n\n"
      "Example:\n"
      "    >>> from PyRuntime import OMExecutionSession\n"
      "    >>> import numpy as np\n"
      "    >>> session = OMExecutionSession('model.so')\n"
      "    >>> inputs = [np.array([[1.0, 2.0, 3.0]], dtype=np.float32)]\n"
      "    >>> outputs = session.run(inputs)\n"
      "    >>> print(outputs[0])")
      .def(py::init<const std::string &, const std::string &, const bool>(),
          py::arg("shared_lib_path"),
          py::arg("tag") = "",
          py::arg("use_default_entry_point") = true,
          "Create an execution session for a compiled model.\n\n"
          "Loads a compiled ONNX model (shared library) and prepares it for inference.\n"
          "The model must have been previously compiled using onnx-mlir.\n\n"
          "Args:\n"
          "    shared_lib_path (str): Path to the compiled model shared library.\n"
          "        Examples: './model.so' (Linux), 'model.dll' (Windows).\n"
          "        Absolute path are preferred (otherwise see default search for given OS)"
          "    tag (str, optional): Model tag for identification. If provided, must\n"
          "        match the tag used during compilation. Default: ''.\n"
          "    use_default_entry_point (bool, optional): If True, use the default\n"
          "        entry point ('run_main_graph'). If False, you must call\n"
          "        set_entry_point() before running inference. Default: True.\n\n"
          "Raises:\n"
          "    RuntimeError: If the shared library cannot be loaded, the tag doesn't\n"
          "        match, or the entry point is not found.\n\n"
          "Example:\n"
          "    >>> # Load with default entry point\n"
          "    >>> session = OMExecutionSession('mnist.so')\n"
          "    >>> \n"
          "    >>> # Load with specific tag\n"
          "    >>> session = OMExecutionSession('model.so', tag='v1.0')\n"
          "    >>> \n"
          "    >>> # Load without default entry point (manual selection)\n"
          "    >>> session = OMExecutionSession('model.so', use_default_entry_point=False)\n"
          "    >>> session.set_entry_point('custom_entry')")
      .def("entry_points",
          &onnx_mlir::PyExecutionSession::pyQueryEntryPoints,
          "Get list of available entry points in the compiled model.\n\n"
          "Returns all entry point names that can be used for inference.\n"
          "Useful for models compiled with multiple entry points.\n\n"
          "Returns:\n"
          "    list[str]: List of entry point names available in the model.\n\n"
          "Example:\n"
          "    >>> session = OMExecutionSession('model.so', use_default_entry_point=False)\n"
          "    >>> entry_points = session.entry_points()\n"
          "    >>> print(entry_points)  # ['run_main_graph', 'run_subgraph_1']\n"
          "    >>> session.set_entry_point(entry_points[0])")
      .def("set_entry_point",
          &onnx_mlir::PyExecutionSession::pySetEntryPoint,
          py::arg("name"),
          "Set the active entry point for inference.\n\n"
          "Selects which entry point to use when calling run(). This is required\n"
          "if the session was created with use_default_entry_point=False, or if\n"
          "you want to switch between multiple entry points.\n\n"
          "Args:\n"
          "    name (str): Name of the entry point to use. Must be one of the\n"
          "        entry points returned by entry_points().\n\n"
          "Raises:\n"
          "    RuntimeError: If the entry point name is not found in the model.\n\n"
          "Example:\n"
          "    >>> session = OMExecutionSession('model.so', use_default_entry_point=False)\n"
          "    >>> session.set_entry_point('run_main_graph')\n"
          "    >>> outputs = session.run(inputs)")
      .def("run",
          &onnx_mlir::PyExecutionSession::pyRun,
          py::arg("input"),
          py::arg("shape"),
          py::arg("stride"),
          "Run inference on the model.\n\n"
          "Executes the model with the provided inputs and returns the outputs.\n"
          "All inputs must be numpy arrays with compatible shapes and types.\n\n"
          "Args:\n"
          "    input (list[numpy.ndarray]): List of input tensors as numpy arrays.\n"
          "        The number and types must match the model's input signature.\n"
          "    shape (list[numpy.ndarray]): List of shape arrays for each input.\n"
          "        Each shape array contains the dimensions of the corresponding input.\n"
          "    stride (list[numpy.ndarray]): List of stride arrays for each input.\n"
          "        Each stride array contains the memory strides of the corresponding input.\n\n"
          "Returns:\n"
          "    list[numpy.ndarray]: List of output tensors as numpy arrays.\n\n"
          "Raises:\n"
          "    RuntimeError: If input shapes/types don't match the model signature,\n"
          "        or if inference fails.\n\n"
          "Example:\n"
          "    >>> import numpy as np\n"
          "    >>> session = OMExecutionSession('mnist.so')\n"
          "    >>> \n"
          "    >>> # Prepare input\n"
          "    >>> img = np.random.rand(1, 1, 28, 28).astype(np.float32)\n"
          "    >>> inputs = [img]\n"
          "    >>> shapes = [np.array(img.shape, dtype=np.int64)]\n"
          "    >>> strides = [np.array(img.strides, dtype=np.int64)]\n"
          "    >>> \n"
          "    >>> # Run inference\n"
          "    >>> outputs = session.run(inputs, shapes, strides)\n"
          "    >>> predictions = outputs[0]\n"
          "    >>> print(f'Predicted class: {np.argmax(predictions)}')")
      .def("run_with_signal_handler",
          &onnx_mlir::PyExecutionSession::pyRunWithSignalHandler,
          py::arg("input"),
          py::arg("shape"),
          py::arg("stride"),
          "Run inference on the model with signal handlers.\n\n"
          "Same as run, but with signal handler (only effective in posix env).\n"
          "Handlers are not thread safe, execution may not be safe after catching \n"
          "a signal as process memory might be corrupted. Use for debugging purpose only.\n")
      .def("input_signature",
          &onnx_mlir::PyExecutionSession::pyInputSignature,
          "Get the input signature of the model.\n\n"
          "Returns a string describing the expected input tensors, including\n"
          "their names, shapes, and data types. Useful for understanding what\n"
          "inputs the model expects.\n\n"
          "Returns:\n"
          "    str: Human-readable description of the model's input signature.\n\n"
          "Example:\n"
          "    >>> session = OMExecutionSession('model.so')\n"
          "    >>> print(session.input_signature())\n"
          "    # Output: \"Input 0: name='input', shape=[1,3,224,224], type=float32\"")
      .def("output_signature",
          &onnx_mlir::PyExecutionSession::pyOutputSignature,
          "Get the output signature of the model.\n\n"
          "Returns a string describing the model's output tensors, including\n"
          "their names, shapes, and data types. Useful for understanding what\n"
          "outputs the model produces.\n\n"
          "Returns:\n"
          "    str: Human-readable description of the model's output signature.\n\n"
          "Example:\n"
          "    >>> session = OMExecutionSession('model.so')\n"
          "    >>> print(session.output_signature())\n"
          "    # Output: \"Output 0: name='output', shape=[1,1000], type=float32\"")
      .def("print_instrumentation",
          &onnx_mlir::PyExecutionSession::pyPrintInstrumentation,
          "Print instrumentation data from the model execution.\n\n"
          "If the model was compiled with instrumentation enabled, this method\n"
          "prints performance metrics and profiling information collected during\n"
          "inference. If no instrumentation is available, this does nothing.\n\n"
          "Example:\n"
          "    >>> session = OMExecutionSession('model.so')\n"
          "    >>> outputs = session.run(inputs, shapes, strides)\n"
          "    >>> session.print_instrumentation()\n"
          "    # Prints timing and performance data if instrumentation was enabled");
}
// clang-format on
#endif
