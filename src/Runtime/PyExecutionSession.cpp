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
  auto *wrappedInput = ormrd_create();
  int inputIdx = 0;
  for (auto inputPyArray : inputsPyArray) {
    auto *inputRtMemRef = rmr_create(inputPyArray.ndim());
    assert(inputPyArray.flags() && py::array::c_style &&
           "Expect contiguous python array.");

    if (inputPyArray.writeable()) {
      rmr_setData(inputRtMemRef, inputPyArray.mutable_data());
      rmr_setAlignedData(inputRtMemRef, inputPyArray.mutable_data());
    } else {
      // If data is not writable, copy them to a writable buffer.
      auto *copiedData = (float *)malloc(inputPyArray.nbytes());
      memcpy(copiedData, inputPyArray.data(), inputPyArray.nbytes());
      rmr_setData(inputRtMemRef, copiedData);
      rmr_setAlignedData(inputRtMemRef, copiedData);
    }

    rmr_setDataSizes(inputRtMemRef, (INDEX_TYPE *)inputPyArray.shape());
    rmr_setDataStrides(inputRtMemRef, (int64_t *)inputPyArray.strides());

    ormrd_setRmrByIndex(wrappedInput, inputRtMemRef, inputIdx++);
  }

  std::vector<py::array> outputPyArrays;
  auto *wrappedOutput = _entryPointFunc(wrappedInput);
  for (int i = 0; i < ormrd_getNumOfRmrs(wrappedOutput); i++) {
    auto *rmr = ormrd_getRmrByIndex(wrappedOutput, i);
    auto shape = std::vector<int64_t>(
        rmr_getDataSizes(rmr), rmr_getDataSizes(rmr) + rmr_getRank(rmr));

    // https://numpy.org/devdocs/user/basics.types.html
    py::dtype dtype;
    if (rmr_getDataType(rmr) == onnx::TensorProto::FLOAT)
      dtype = py::dtype("float32");
    else if (rmr_getDataType(rmr) == onnx::TensorProto::UINT8)
      dtype = py::dtype("uint8");
    else if (rmr_getDataType(rmr) == onnx::TensorProto::INT8)
      dtype = py::dtype("int8");
    else if (rmr_getDataType(rmr) == onnx::TensorProto::UINT16)
      dtype = py::dtype("uint16");
    else if (rmr_getDataType(rmr) == onnx::TensorProto::INT16)
      dtype = py::dtype("int16");
    else if (rmr_getDataType(rmr) == onnx::TensorProto::INT32)
      dtype = py::dtype("int32");
    else if (rmr_getDataType(rmr) == onnx::TensorProto::INT64)
      dtype = py::dtype("int64");
    // TODO(tjingrant) wait for Tong's input for how to represent string.
    else if (rmr_getDataType(rmr) == onnx::TensorProto::BOOL)
      dtype = py::dtype("bool_");
    else if (rmr_getDataType(rmr) == onnx::TensorProto::FLOAT16)
      dtype = py::dtype("float32");
    else if (rmr_getDataType(rmr) == onnx::TensorProto::DOUBLE)
      dtype = py::dtype("float64");
    else if (rmr_getDataType(rmr) == onnx::TensorProto::UINT32)
      dtype = py::dtype("uint32");
    else if (rmr_getDataType(rmr) == onnx::TensorProto::UINT64)
      dtype = py::dtype("uint64");
    else {
      fprintf(stderr, "Unsupported ONNX type in RtMemRef.");
      exit(1);
    }

    outputPyArrays.emplace_back(py::array(dtype, shape, rmr_getData(rmr)));
  }

  return outputPyArrays;
}
} // namespace onnx_mlir
