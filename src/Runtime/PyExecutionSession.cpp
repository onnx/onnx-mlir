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
#include <third_party/onnx/onnx/onnx_pb.h>

#include "PyExecutionSession.hpp"

namespace onnx_mlir {

std::vector<py::array> PyExecutionSession::pyRun(
    std::vector<py::array> inputsPyArray) {
  assert(_entryPointFunc && "Entry point not loaded.");

  std::vector<OMTensor *> omts;
  for (auto inputPyArray : inputsPyArray) {
    auto *inputOMTensor = omTensorCreateEmptyDeprecated(inputPyArray.ndim());
    assert(inputPyArray.flags() && py::array::c_style &&
           "Expect contiguous python array.");

    if (inputPyArray.writeable()) {
      omTensorSetPtr(inputOMTensor, /*owning=*/false,
          /*allocatedPtr=*/inputPyArray.mutable_data(),
          /*alignedPtr=*/inputPyArray.mutable_data());
    } else {
      // If data is not writable, copy them to a writable buffer.
      auto *copiedData = (float *)malloc(inputPyArray.nbytes());
      memcpy(copiedData, inputPyArray.data(), inputPyArray.nbytes());
      omTensorSetPtr(inputOMTensor, /*owning=*/true,
          /*allocatedPtr=*/copiedData, /*alignedPtr=*/copiedData);
    }

    omTensorSetShape(inputOMTensor, (int64_t *)inputPyArray.shape());
    omTensorSetStrides(inputOMTensor, (int64_t *)inputPyArray.strides());
    omts.emplace_back(inputOMTensor);
  }
  auto *wrappedInput = omTensorListCreate(&omts[0], omts.size());

  auto *wrappedOutput = _entryPointFunc(wrappedInput);

  std::vector<py::array> outputPyArrays;
  for (int i = 0; i < omTensorListGetSize(wrappedOutput); i++) {
    auto *omt = omTensorListGetOmtByIndex(wrappedOutput, i);
    auto shape = std::vector<int64_t>(omTensorGetDataShape(omt),
        omTensorGetDataShape(omt) + omTensorGetRank(omt));

    // https://numpy.org/devdocs/user/basics.types.html
    py::dtype dtype;
    if (omTensorGetDataType(omt) == onnx::TensorProto::FLOAT)
      dtype = py::dtype("float32");
    else if (omTensorGetDataType(omt) == onnx::TensorProto::UINT8)
      dtype = py::dtype("uint8");
    else if (omTensorGetDataType(omt) == onnx::TensorProto::INT8)
      dtype = py::dtype("int8");
    else if (omTensorGetDataType(omt) == onnx::TensorProto::UINT16)
      dtype = py::dtype("uint16");
    else if (omTensorGetDataType(omt) == onnx::TensorProto::INT16)
      dtype = py::dtype("int16");
    else if (omTensorGetDataType(omt) == onnx::TensorProto::INT32)
      dtype = py::dtype("int32");
    else if (omTensorGetDataType(omt) == onnx::TensorProto::INT64)
      dtype = py::dtype("int64");
    // TODO(tjingrant) wait for Tong's input for how to represent string.
    else if (omTensorGetDataType(omt) == onnx::TensorProto::BOOL)
      dtype = py::dtype("bool_");
    else if (omTensorGetDataType(omt) == onnx::TensorProto::FLOAT16)
      dtype = py::dtype("float32");
    else if (omTensorGetDataType(omt) == onnx::TensorProto::DOUBLE)
      dtype = py::dtype("float64");
    else if (omTensorGetDataType(omt) == onnx::TensorProto::UINT32)
      dtype = py::dtype("uint32");
    else if (omTensorGetDataType(omt) == onnx::TensorProto::UINT64)
      dtype = py::dtype("uint64");
    else {
      fprintf(stderr, "Unsupported ONNX type in OMTensor.");
      exit(1);
    }

    outputPyArrays.emplace_back(
        py::array(dtype, shape, omTensorGetAlignedPtr(omt)));
  }

  return outputPyArrays;
}
} // namespace onnx_mlir
