/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- PyExecutionSession.hpp - PyExecutionSession Implementation -----===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementations of PyExecutionSession class, which helps
// python programs interact with compiled binary model libraries.
//
//===----------------------------------------------------------------------===//

#include "src/Support/SuppressWarnings.h"

SUPPRESS_WARNINGS_PUSH
#include "onnx/onnx_pb.h"
SUPPRESS_WARNINGS_POP

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
    int64_t ownData = 0;
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
    if (py::isinstance<py::array_t<float>>(inputPyArray))
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
    // string type missing
    else if (py::isinstance<py::array_t<bool>>(inputPyArray))
      dtype = ONNX_TYPE_BOOL;
    // Missing fp16 support.
    else if (py::isinstance<py::array_t<double>>(inputPyArray))
      dtype = ONNX_TYPE_DOUBLE;
    else if (py::isinstance<py::array_t<std::uint32_t>>(inputPyArray))
      dtype = ONNX_TYPE_UINT32;
    else if (py::isinstance<py::array_t<std::uint64_t>>(inputPyArray))
      dtype = ONNX_TYPE_UINT64;
    else if (py::isinstance<py::array_t<std::complex<float>>>(inputPyArray))
      dtype = ONNX_TYPE_COMPLEX64;
    else if (py::isinstance<py::array_t<std::complex<double>>>(inputPyArray))
      dtype = ONNX_TYPE_COMPLEX128;
    // Missing bfloat16 support
    else {
      std::cerr << "Numpy type not supported: " << inputPyArray.dtype()
                << ".\n";
      exit(1);
    }

    auto *inputOMTensor = omTensorCreateWithOwnership(dataPtr,
        (int64_t *)(const_cast<ssize_t *>(inputPyArray.shape())),
        (int64_t)inputPyArray.ndim(), dtype, ownData);
    omTensorSetStridesWithPyArrayStrides(inputOMTensor,
        (int64_t *)const_cast<ssize_t *>(inputPyArray.strides()));

    omts.emplace_back(inputOMTensor);
  }

  auto *wrappedInput = omTensorListCreate(&omts[0], omts.size());
  auto *wrappedOutput = _entryPointFunc(wrappedInput);
  if (!wrappedOutput)
    throw std::runtime_error(reportErrnoError());
  std::vector<py::array> outputPyArrays;
  for (int64_t i = 0; i < omTensorListGetSize(wrappedOutput); i++) {
    auto *omt = omTensorListGetOmtByIndex(wrappedOutput, i);
    auto shape = std::vector<int64_t>(
        omTensorGetShape(omt), omTensorGetShape(omt) + omTensorGetRank(omt));

    // https://numpy.org/devdocs/user/basics.types.html
    py::dtype dtype;
    switch (omTensorGetDataType(omt)) {
    case (OM_DATA_TYPE)onnx::TensorProto::FLOAT:
      dtype = py::dtype("float32");
      break;
    case (OM_DATA_TYPE)onnx::TensorProto::UINT8:
      dtype = py::dtype("uint8");
      break;
    case (OM_DATA_TYPE)onnx::TensorProto::INT8:
      dtype = py::dtype("int8");
      break;
    case (OM_DATA_TYPE)onnx::TensorProto::UINT16:
      dtype = py::dtype("uint16");
      break;
    case (OM_DATA_TYPE)onnx::TensorProto::INT16:
      dtype = py::dtype("int16");
      break;
    case (OM_DATA_TYPE)onnx::TensorProto::INT32:
      dtype = py::dtype("int32");
      break;
    case (OM_DATA_TYPE)onnx::TensorProto::INT64:
      dtype = py::dtype("int64");
      break;
    case (OM_DATA_TYPE)onnx::TensorProto::STRING:
      dtype = py::dtype("str");
      break;
    case (OM_DATA_TYPE)onnx::TensorProto::BOOL:
      dtype = py::dtype("bool_");
      break;
    case (OM_DATA_TYPE)onnx::TensorProto::FLOAT16:
      dtype = py::dtype("float32");
      break;
    case (OM_DATA_TYPE)onnx::TensorProto::DOUBLE:
      dtype = py::dtype("float64");
      break;
    case (OM_DATA_TYPE)onnx::TensorProto::UINT32:
      dtype = py::dtype("uint32");
      break;
    case (OM_DATA_TYPE)onnx::TensorProto::UINT64:
      dtype = py::dtype("uint64");
      break;
    case (OM_DATA_TYPE)onnx::TensorProto::COMPLEX64:
      dtype = py::dtype("csingle");
      break;
    case (OM_DATA_TYPE)onnx::TensorProto::COMPLEX128:
      dtype = py::dtype("cdouble");
      break;
    default:
      std::cerr << "Unsupported ONNX type in OMTensor: "
                << omTensorGetDataType(omt) << ".\n";
      exit(1);
    }

    outputPyArrays.emplace_back(
        py::array(dtype, shape, omTensorGetDataPtr(omt)));
  }

  return outputPyArrays;
}

void PyExecutionSession::pySetEntryPoint(std::string entryPointName) {
  setEntryPoint(entryPointName);
}

std::vector<std::string> PyExecutionSession::pyQueryEntryPoints() {
  assert(_queryEntryPointsFunc && "Query entry point not loaded.");
  const char **entryPointArr = _queryEntryPointsFunc(NULL);

  std::vector<std::string> outputPyArrays;
  int i = 0;
  while (entryPointArr[i] != NULL) {
    outputPyArrays.emplace_back(entryPointArr[i]);
    i++;
  }
  return outputPyArrays;
}

std::string PyExecutionSession::pyInputSignature() {
  assert(_inputSignatureFunc && "Input signature entry point not loaded.");
  return inputSignature();
}

std::string PyExecutionSession::pyOutputSignature() {
  assert(_outputSignatureFunc && "Output signature entry point not loaded.");
  return outputSignature();
}

} // namespace onnx_mlir
