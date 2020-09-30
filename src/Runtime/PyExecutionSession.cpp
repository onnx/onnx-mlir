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
    const std::vector<py::array> &inputsPyArray) {
  assert(_entryPointFunc && "Entry point not loaded.");

  std::vector<OMTensor *> omts;
  for (auto inputPyArray : inputsPyArray) {
    assert(inputPyArray.flags() && py::array::c_style &&
           "Expect contiguous python array.");

    void *dataPtr;
    int ownData = 0;
    if (inputPyArray.writeable()) {
      dataPtr = inputPyArray.mutable_data();
    } else {
      // If data is not writable, copy them to a writable buffer.
      auto *copiedData = (float *)malloc(inputPyArray.nbytes());
      memcpy(copiedData, inputPyArray.data(), inputPyArray.nbytes());
      dataPtr = copiedData;
      // We want OMTensor to free up the memory space upon destruction.
      ownData = 1;
    }

    // Borrowed from:
    // https://github.com/pybind/pybind11/issues/563#issuecomment-267835542
    OM_DATA_TYPE dtype;
    if (py::isinstance<py::array_t<std::float_t>>(inputPyArray))
      dtype = ONNX_TYPE_FLOAT;
    else if (py::isinstance<py::array_t<std::uint8_t>>(inputPyArray))
      dtype = ONNX_TYPE_UINT8;
    else if (py::isinstance<py::array_t<std::int8_t>>(inputPyArray))
      dtype = ONNX_TYPE_INT8;
    else if (py::isinstance<py::array_t<std::uint16_t>>(inputPyArray))
      dtype = ONNX_TYPE_UINT16;
    else if (py::isinstance<py::array_t<std::int16_t>>(inputPyArray))
      dtype = ONNX_TYPE_INT16;
    else if (py::isinstance<py::array_t<std::int32_t>>(inputPyArray))
      dtype = ONNX_TYPE_INT32;
    else if (py::isinstance<py::array_t<std::int64_t>>(inputPyArray))
      dtype = ONNX_TYPE_INT64;
    else if (py::isinstance<py::array_t<bool>>(inputPyArray))
      dtype = ONNX_TYPE_BOOL;
    // Missing fp16 support.
    else if (py::isinstance<py::array_t<std::double_t>>(inputPyArray))
      dtype = ONNX_TYPE_DOUBLE;
    else if (py::isinstance<py::array_t<std::uint32_t>>(inputPyArray))
      dtype = ONNX_TYPE_UINT32;
    else if (py::isinstance<py::array_t<std::uint64_t>>(inputPyArray))
      dtype = ONNX_TYPE_UINT64;
    else {
      std::cerr << "Numpy type not supported: " << inputPyArray.dtype() << "\n";
      exit(1);
    }

    auto *inputOMTensor = omTensorCreateWithOwnership(dataPtr,
        (int64_t *)inputPyArray.shape(), inputPyArray.ndim(), dtype, ownData);
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
        py::array(dtype, shape, omTensorGetDataPtr(omt)));
  }

  return outputPyArrays;
}
} // namespace onnx_mlir
