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
    auto *inputOMTensor = omtCreate(inputPyArray.ndim());
    assert(inputPyArray.flags() && py::array::c_style &&
           "Expect contiguous python array.");

    if (inputPyArray.writeable()) {
      omtSetData(inputOMTensor, inputPyArray.mutable_data());
      omtSetAlignedData(inputOMTensor, inputPyArray.mutable_data());
    } else {
      // If data is not writable, copy them to a writable buffer.
      auto *copiedData = (float *)malloc(inputPyArray.nbytes());
      memcpy(copiedData, inputPyArray.data(), inputPyArray.nbytes());
      omtSetData(inputOMTensor, copiedData);
      omtSetAlignedData(inputOMTensor, copiedData);
    }

    omtSetDataShape(inputOMTensor, (INDEX_TYPE *)inputPyArray.shape());
    omtSetDataStrides(inputOMTensor, (int64_t *)inputPyArray.strides());
    omts.emplace_back(inputOMTensor);
  }
  auto *wrappedInput = omTensorListCreate(&omts[0], omts.size());

  auto *wrappedOutput = _entryPointFunc(wrappedInput);

  std::vector<py::array> outputPyArrays;
  for (int i = 0; i < omTensorListGetNumOmts(wrappedOutput); i++) {
    auto *omt = omTensorListGetOmtByIndex(wrappedOutput, i);
    auto shape = std::vector<int64_t>(
        omtGetDataShape(omt), omtGetDataShape(omt) + omtGetRank(omt));

    // https://numpy.org/devdocs/user/basics.types.html
    py::dtype dtype;
    if (omtGetDataType(omt) == onnx::TensorProto::FLOAT)
      dtype = py::dtype("float32");
    else if (omtGetDataType(omt) == onnx::TensorProto::UINT8)
      dtype = py::dtype("uint8");
    else if (omtGetDataType(omt) == onnx::TensorProto::INT8)
      dtype = py::dtype("int8");
    else if (omtGetDataType(omt) == onnx::TensorProto::UINT16)
      dtype = py::dtype("uint16");
    else if (omtGetDataType(omt) == onnx::TensorProto::INT16)
      dtype = py::dtype("int16");
    else if (omtGetDataType(omt) == onnx::TensorProto::INT32)
      dtype = py::dtype("int32");
    else if (omtGetDataType(omt) == onnx::TensorProto::INT64)
      dtype = py::dtype("int64");
    // TODO(tjingrant) wait for Tong's input for how to represent string.
    else if (omtGetDataType(omt) == onnx::TensorProto::BOOL)
      dtype = py::dtype("bool_");
    else if (omtGetDataType(omt) == onnx::TensorProto::FLOAT16)
      dtype = py::dtype("float32");
    else if (omtGetDataType(omt) == onnx::TensorProto::DOUBLE)
      dtype = py::dtype("float64");
    else if (omtGetDataType(omt) == onnx::TensorProto::UINT32)
      dtype = py::dtype("uint32");
    else if (omtGetDataType(omt) == onnx::TensorProto::UINT64)
      dtype = py::dtype("uint64");
    else {
      fprintf(stderr, "Unsupported ONNX type in OMTensor.");
      exit(1);
    }

    outputPyArrays.emplace_back(py::array(dtype, shape, omtGetData(omt)));
  }

  return outputPyArrays;
}
} // namespace onnx_mlir
