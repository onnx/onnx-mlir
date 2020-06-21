//===----- PyExecusionSession.hpp - PyExecutionSession Implementation -----===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementations of PyExecusionSession class, which helps
// python programs interact with compiled binary model libraries.
//
//===----------------------------------------------------------------------===//

#include "onnx/onnx_pb.h"

#include "PyExecutionSession.hpp"

namespace onnx_mlir {

std::vector<py::array> PyExecutionSession::pyRun(
    std::vector<py::array> inputsPyArray) {
  assert(_entryPointFunc && "Entry point not loaded.");
  auto *wrappedInput = createOrderedDynMemRefDict();
  int inputIdx = 0;
  for (auto inputPyArray : inputsPyArray) {
    auto *inputDynMemRef = createDynMemRef(inputPyArray.ndim());
    assert(inputPyArray.flags() && py::array::c_style &&
           "Expect contiguous python array.");

    if (inputPyArray.writeable()) {
      inputDynMemRef->data = inputPyArray.mutable_data();
      inputDynMemRef->alignedData = inputPyArray.mutable_data();
    } else {
      // If data is not writable, copy them to a writable buffer.
      auto *copiedData = (float *)malloc(inputPyArray.nbytes());
      memcpy(copiedData, inputPyArray.data(), inputPyArray.nbytes());
      inputDynMemRef->data = copiedData;
      inputDynMemRef->alignedData = copiedData;
    }

    for (int i = 0; i < inputPyArray.ndim(); i++) {
      inputDynMemRef->sizes[i] = inputPyArray.shape(i);
      inputDynMemRef->strides[i] = inputPyArray.strides(i);
    }

    setDynMemRef(wrappedInput, inputIdx++, inputDynMemRef);
  }

  std::vector<py::array> outputPyArrays;
  auto *wrappedOutput = _entryPointFunc(wrappedInput);
  for (int i = 0; i < numDynMemRefs(wrappedOutput); i++) {
    auto *dynMemRef = getDynMemRef(wrappedOutput, i);
    auto shape = std::vector<int64_t>(
        dynMemRef->sizes, dynMemRef->sizes + dynMemRef->rank);

    // https://numpy.org/devdocs/user/basics.types.html
    py::dtype dtype;
    if (dynMemRef->dtype == onnx::TensorProto::INT32)
      dtype = py::dtype("int32");
    else if (dynMemRef->dtype == onnx::TensorProto::FLOAT)
      dtype = py::dtype("float32");

    outputPyArrays.emplace_back(py::array(dtype, shape, dynMemRef->data));
  }

  return outputPyArrays;
}
} // namespace onnx_mlir