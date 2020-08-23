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

  std::vector<RtMemRef *> rmrs;
  int inputIdx = 0;
  for (auto inputPyArray : inputsPyArray) {
    auto *inputRtMemRef = rmrCreate(inputPyArray.ndim());
    assert(inputPyArray.flags() && py::array::c_style &&
           "Expect contiguous python array.");

    if (inputPyArray.writeable()) {
      rmrSetData(inputRtMemRef, inputPyArray.mutable_data());
      rmrSetAlignedData(inputRtMemRef, inputPyArray.mutable_data());
    } else {
      // If data is not writable, copy them to a writable buffer.
      auto *copiedData = (float *)malloc(inputPyArray.nbytes());
      memcpy(copiedData, inputPyArray.data(), inputPyArray.nbytes());
      rmrSetData(inputRtMemRef, copiedData);
      rmrSetAlignedData(inputRtMemRef, copiedData);
    }

    rmrSetDataShape(inputRtMemRef, (INDEX_TYPE *)inputPyArray.shape());
    rmrSetDataStrides(inputRtMemRef, (int64_t *)inputPyArray.strides());
    rmrs.emplace_back(inputRtMemRef);
  }
  auto *wrappedInput = rmrListCreate(&rmrs[0], rmrs.size());

  std::vector<py::array> outputPyArrays;
  auto *wrappedOutput = _entryPointFunc(wrappedInput);
  for (int i = 0; i < rmrListGetNumRmrs(wrappedOutput); i++) {
    auto *rmr = rmrListGetRmrByIndex(wrappedOutput, i);
    auto shape = std::vector<int64_t>(
        rmrGetDataShape(rmr), rmrGetDataShape(rmr) + rmrGetRank(rmr));

    // https://numpy.org/devdocs/user/basics.types.html
    py::dtype dtype;
    if (rmrGetDataType(rmr) == onnx::TensorProto::FLOAT)
      dtype = py::dtype("float32");
    else if (rmrGetDataType(rmr) == onnx::TensorProto::UINT8)
      dtype = py::dtype("uint8");
    else if (rmrGetDataType(rmr) == onnx::TensorProto::INT8)
      dtype = py::dtype("int8");
    else if (rmrGetDataType(rmr) == onnx::TensorProto::UINT16)
      dtype = py::dtype("uint16");
    else if (rmrGetDataType(rmr) == onnx::TensorProto::INT16)
      dtype = py::dtype("int16");
    else if (rmrGetDataType(rmr) == onnx::TensorProto::INT32)
      dtype = py::dtype("int32");
    else if (rmrGetDataType(rmr) == onnx::TensorProto::INT64)
      dtype = py::dtype("int64");
    // TODO(tjingrant) wait for Tong's input for how to represent string.
    else if (rmrGetDataType(rmr) == onnx::TensorProto::BOOL)
      dtype = py::dtype("bool_");
    else if (rmrGetDataType(rmr) == onnx::TensorProto::FLOAT16)
      dtype = py::dtype("float32");
    else if (rmrGetDataType(rmr) == onnx::TensorProto::DOUBLE)
      dtype = py::dtype("float64");
    else if (rmrGetDataType(rmr) == onnx::TensorProto::UINT32)
      dtype = py::dtype("uint32");
    else if (rmrGetDataType(rmr) == onnx::TensorProto::UINT64)
      dtype = py::dtype("uint64");
    else {
      fprintf(stderr, "Unsupported ONNX type in RtMemRef.");
      exit(1);
    }

    outputPyArrays.emplace_back(py::array(dtype, shape, rmrGetData(rmr)));
  }

  return outputPyArrays;
}
} // namespace onnx_mlir
