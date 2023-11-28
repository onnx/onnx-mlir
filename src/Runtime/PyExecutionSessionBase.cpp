/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===- PyExecutionSessionBase.cpp - PyExecutionSessionBase Implementation -===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementations of PyExecutionSessionBase class, which
// contains shared code for PyExecutionSession and PyOMCompileExecutionSession.
//
//===----------------------------------------------------------------------===//

#include "src/Support/SmallFP.hpp"
#include "src/Support/SuppressWarnings.h"

SUPPRESS_WARNINGS_PUSH
#include "onnx/onnx_pb.h"
SUPPRESS_WARNINGS_POP

#include "PyExecutionSessionBase.hpp"

namespace pybind11 {
namespace detail {

// Note: Since float16 is not a builtin type in C++, we register
// onnx_mlir::float_16 as numpy.float16.
// Ref: https://github.com/pybind/pybind11/issues/1776
//
// This implementation is copied from https://github.com/PaddlePaddle/Paddle
template <>
struct npy_format_descriptor<onnx_mlir::float_16> {
  static py::dtype dtype() {
    // Note: use same enum number of float16 in numpy.
    // import numpy as np
    // print np.dtype(np.float16).num  # 23
    constexpr int NPY_FLOAT16 = 23;
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
    return reinterpret_borrow<py::dtype>(ptr);
  }
  static std::string format() {
    // Note: "e" represents float16.
    // Details at:
    // https://docs.python.org/3/library/struct.html#format-characters.
    return "e";
  }
  static constexpr auto name = _("float16");
};

} // namespace detail
} // namespace pybind11

namespace onnx_mlir {

PyExecutionSessionBase::PyExecutionSessionBase(
    std::string sharedLibPath, std::string tag, bool defaultEntryPoint)
    : onnx_mlir::ExecutionSession(sharedLibPath, tag, defaultEntryPoint) {}

// =============================================================================
// Run.

std::vector<py::array> PyExecutionSessionBase::pyRun(
    const std::vector<py::array> &inputsPyArray) {
  if (!isInitialized)
    throw std::runtime_error(reportInitError());
  if (!_entryPointFunc)
    throw std::runtime_error(reportUndefinedEntryPointIn("run"));

  // 1. Process inputs.
  std::vector<OMTensor *> omts;
  for (auto inputPyArray : inputsPyArray) {
    if (!inputPyArray.flags() || !py::array::c_style)
      throw std::runtime_error(
          reportPythonError("Expect contiguous python array."));

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
    else if (py::isinstance<py::array_t<float_16>>(inputPyArray))
      dtype = ONNX_TYPE_FLOAT16;
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
    else if (inputPyArray.dtype().kind() == 'O') // case of py::object type
      dtype = ONNX_TYPE_STRING;
    // Missing bfloat16 support
    else {
      std::stringstream errStr;
      errStr << "Numpy type not supported: " << inputPyArray.dtype()
             << std::endl;
      throw std::runtime_error(reportPythonError(errStr.str()));
    }
    OMTensor *inputOMTensor = NULL;
    if (dtype == ONNX_TYPE_STRING) {
      auto shape = inputPyArray.shape();
      uint64_t numElem = 1;
      for (size_t i = 0; i < (size_t)inputPyArray.ndim(); ++i)
        numElem *= shape[i];
      //
      // Convert multi-dimensional array (for string data pointers) to
      // one-dimensional array to mange multi-dimensional array in an
      // integrated way.
      //
      // For numerical array, pybind11 can convert multi-dimensional array into
      // one-dimensional array without manual conversion, but pybind11 has
      // an issue about "dtype caster does not accept strings or type objects"
      // (https://github.com/pybind/pybind11/issues/1538). The issue page
      // shows a solution to avoid the bug, but it does not work for our case.
      // The following part solves the issue temporally, and will be updated
      // if we find better way to avoid the issue or pybind11 fixes the issue.
      //

      // Allocate buffer for one-dimentional array for string data pointers
      char **strPointerArray = (char **)alloca(sizeof(char *) * numElem);
      assert(
          strPointerArray && "fail to alloc array for pointers to string data");
      switch (inputPyArray.ndim()) {
      case 1: {
        auto vec = inputPyArray.cast<std::vector<std::string>>();
        for (int64_t i = 0; i < shape[0]; ++i)
          strPointerArray[i] = vec[i].data();
        break;
      }
      case 2: {
        auto vec = inputPyArray.cast<std::vector<std::vector<std::string>>>();
        int off = 0;
        for (int64_t i = 0; i < shape[0]; ++i)
          for (int64_t j = 0; j < shape[1]; ++j)
            strPointerArray[off++] = vec[i][j].data();
        break;
      }
      case 3: {
        auto vec =
            inputPyArray
                .cast<std::vector<std::vector<std::vector<std::string>>>>();
        int off = 0;
        for (int64_t i = 0; i < shape[0]; ++i)
          for (int64_t j = 0; j < shape[1]; ++j)
            for (int64_t k = 0; k < shape[2]; ++k)
              strPointerArray[off++] = vec[i][j][k].data();
        break;
      }
      case 4: {
        auto vec = inputPyArray.cast<
            std::vector<std::vector<std::vector<std::vector<std::string>>>>>();
        int off = 0;
        for (int64_t i = 0; i < shape[0]; ++i)
          for (int64_t j = 0; j < shape[1]; ++j)
            for (int64_t k = 0; k < shape[2]; ++k)
              for (int64_t l = 0; l < shape[3]; ++l)
                strPointerArray[off++] = vec[i][j][k][l].data();
        break;
      }
      case 5: {
        auto vec = inputPyArray.cast<std::vector<
            std::vector<std::vector<std::vector<std::vector<std::string>>>>>>();
        int off = 0;
        for (int64_t i = 0; i < shape[0]; ++i)
          for (int64_t j = 0; j < shape[1]; ++j)
            for (int64_t k = 0; k < shape[2]; ++k)
              for (int64_t l = 0; l < shape[3]; ++l)
                for (int64_t m = 0; m < shape[4]; ++m)
                  strPointerArray[off++] = vec[i][j][k][l][m].data();
        break;
      }
      case 6: {
        auto vec = inputPyArray.cast<std::vector<std::vector<std::vector<
            std::vector<std::vector<std::vector<std::string>>>>>>>();
        int off = 0;
        for (int64_t i = 0; i < shape[0]; ++i)
          for (int64_t j = 0; j < shape[1]; ++j)
            for (int64_t k = 0; k < shape[2]; ++k)
              for (int64_t l = 0; l < shape[3]; ++l)
                for (int64_t m = 0; m < shape[4]; ++m)
                  for (int64_t n = 0; n < shape[5]; ++n)
                    strPointerArray[off++] = vec[i][j][k][l][m][n].data();
        break;
      }
      default:
        assert(false && "not implemented");
      }
      // Calculate total length of all strings including null termination.
      uint64_t strLenTotal = 0;
      for (uint64_t i = 0; i < numElem; i++)
        strLenTotal += strlen(strPointerArray[i]) + 1;
      // allocate OMTensor->_allocatedPtr for holding both string data pointers
      // and strind data themselves.
      void *strArray =
          malloc(sizeof(char *) * numElem + sizeof(char) * strLenTotal);
      assert(strArray && "fail to alloc array for pointers to string data and "
                         "string data themselves");
      memcpy(strArray, (void *)strPointerArray, sizeof(char *) * numElem);
      void *strDataBuffer = (void *)(((char **)strArray) + numElem);
      char *strDataPtr = (char *)strDataBuffer;
      for (uint64_t i = 0; i < numElem; i++) {
        strcpy(strDataPtr, strPointerArray[i]);
        ((char **)strArray)[i] = strDataPtr;
        strDataPtr += strlen(strPointerArray[i]) + 1;
      }
      inputOMTensor = omTensorCreateWithOwnership(strArray,
          reinterpret_cast<const int64_t *>(inputPyArray.shape()),
          static_cast<int64_t>(inputPyArray.ndim()), dtype, /*own_data=*/true);
      omTensorSetStridesWithPyArrayStrides(inputOMTensor,
          reinterpret_cast<const int64_t *>(inputPyArray.strides()));
    } else if (std::is_same<int64_t, pybind11::ssize_t>::value) {
      inputOMTensor = omTensorCreateWithOwnership(dataPtr,
          reinterpret_cast<const int64_t *>(inputPyArray.shape()),
          static_cast<int64_t>(inputPyArray.ndim()), dtype, ownData);
      omTensorSetStridesWithPyArrayStrides(inputOMTensor,
          reinterpret_cast<const int64_t *>(inputPyArray.strides()));
    } else {
      std::vector<int64_t> safeShape(
          inputPyArray.shape(), inputPyArray.shape() + inputPyArray.ndim());
      std::vector<int64_t> safeStrides(
          inputPyArray.strides(), inputPyArray.strides() + inputPyArray.ndim());
      inputOMTensor = omTensorCreateWithOwnership(dataPtr, safeShape.data(),
          (int64_t)inputPyArray.ndim(), dtype, ownData);
      omTensorSetStridesWithPyArrayStrides(inputOMTensor, safeStrides.data());
    }
    omts.emplace_back(inputOMTensor);
  }

  // 2. Call entry point.
  auto *wrappedInput = omTensorListCreate(&omts[0], omts.size());
  auto *wrappedOutput = _entryPointFunc(wrappedInput);
  if (!wrappedOutput)
    throw std::runtime_error(reportErrnoError());

  // 3. Process outputs.
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
      dtype = py::dtype("float16");
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
    default: {
      std::stringstream errStr;
      errStr << "Unsupported ONNX type in OMTensor: "
             << omTensorGetDataType(omt) << std::endl;

      throw std::runtime_error(reportPythonError(errStr.str()));
    }
    }

    outputPyArrays.emplace_back(
        py::array(dtype, shape, omTensorGetDataPtr(omt)));
  }
  omTensorListDestroy(wrappedOutput);
  omTensorListDestroy(wrappedInput);

  return outputPyArrays;
}

// =============================================================================
// Setter and getter.

void PyExecutionSessionBase::pySetEntryPoint(std::string entryPointName) {
  setEntryPoint(entryPointName);
}

std::vector<std::string> PyExecutionSessionBase::pyQueryEntryPoints() {
  if (!isInitialized)
    throw std::runtime_error(reportInitError());
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

std::string PyExecutionSessionBase::pyInputSignature() {
  return inputSignature();
}

std::string PyExecutionSessionBase::pyOutputSignature() {
  return outputSignature();
}

// =============================================================================
// Error reporting

std::string PyExecutionSessionBase::reportPythonError(
    std::string errorStr) const {
  errno = EFAULT; // Bad Address.
  std::stringstream errStr;
  errStr << "Execution session: encountered python error `" << errorStr << "'."
         << std::endl;
  return errStr.str();
}

} // namespace onnx_mlir
