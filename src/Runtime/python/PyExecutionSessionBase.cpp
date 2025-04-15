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

#ifndef ENABLE_PYRUNTIME_LIGHT
#include "src/Support/SmallFP.hpp"
#else
// ToFix: how to handle float_16
#endif

// SuppressWarnings.h only defines macros, not functions.
#include "src/Support/SuppressWarnings.h"

SUPPRESS_WARNINGS_PUSH
#include "onnx/onnx_pb.h"
SUPPRESS_WARNINGS_POP

#include "PyExecutionSessionBase.hpp"

#define OM_DRIVER_TIMING 1 /* 1 for timing, 0 for no timing/overheads */
#include "src/Runtime/OMInstrumentHelper.h"

namespace pybind11 {
namespace detail {

// Note: Since float16 is not a builtin type in C++, we register
// onnx_mlir::float_16 as numpy.float16.
// Ref: https://github.com/pybind/pybind11/issues/1776
//
// This implementation is copied from https://github.com/PaddlePaddle/Paddle

#ifndef ENABLE_PYRUNTIME_LIGHT
// ToFix: support for float_16
// Now onnx_mlir::float_16 is not defined without SmallFP.h

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
#endif

} // namespace detail
} // namespace pybind11

namespace onnx_mlir {

//
// Generate OMTensor's buffer for string input in pyArray
//
// For numerical array, pybind11 can convert multi-dimensional array into
// one-dimensional array without manual conversion, but pybind11 cannot
// convert multi-dimensional string array into one-dimensional array
// automatically.
// This function will be rewritten when pybind fixes the issue, or
// better way for fixing it is found.
//
void *generateOMTensorBufferForStringData(py::array pyArray) {
  auto shape = pyArray.shape();
  uint64_t numElem = 1;
  for (size_t i = 0; i < (size_t)pyArray.ndim(); ++i)
    numElem *= shape[i];
  uint64_t strLenTotal = 0;
  uint64_t off = 0;
  void *dataBuffer = NULL;
  assert(pyArray.ndim() == 1 && "input pyArray should be flatten");
  auto vec = pyArray.cast<std::vector<std::string>>();
  for (int64_t i = 0; i < shape[0]; ++i)
    strLenTotal += strlen(vec[i].data()) + 1;
  dataBuffer = malloc(sizeof(char *) * numElem + strLenTotal);
  if (dataBuffer == NULL)
    return NULL;
  char **strArray = (char **)dataBuffer;
  char *strPos = (char *)(((char *)dataBuffer) + sizeof(char *) * numElem);
  for (int64_t i = 0; i < shape[0]; ++i) {
    strcpy(strPos, vec[i].data());
    strArray[off++] = strPos;
    strPos += strlen(vec[i].data()) + 1;
  }
  return dataBuffer;
}

PyExecutionSessionBase::PyExecutionSessionBase(
    std::string sharedLibPath, std::string tag, bool defaultEntryPoint)
    : onnx_mlir::ExecutionSession(sharedLibPath, tag, defaultEntryPoint) {}

// =============================================================================
// Run.

std::vector<py::array> PyExecutionSessionBase::pyRun(
    const std::vector<py::array> &inputsPyArray,
    const std::vector<py::array> &shapesPyArray,
    const std::vector<py::array> &stridesPyArray) {
  if (!isInitialized)
    throw std::runtime_error(reportInitError());
  if (!_entryPointFunc)
    throw std::runtime_error(reportUndefinedEntryPointIn("run"));

  // 1. Process inputs.
  TIMING_INIT_START(process_input);
  std::vector<OMTensor *> omts;
  if (inputsPyArray.size() != shapesPyArray.size())
    throw std::runtime_error(
        reportPythonError("numbers of inputs and shapes should be the same"));
  if (inputsPyArray.size() != stridesPyArray.size())
    throw std::runtime_error(
        reportPythonError("numbers of inputs and strides should be the same"));
  for (size_t argId = 0; argId < inputsPyArray.size(); argId++) {
    auto inputPyArray = inputsPyArray[argId];
    auto shapePyArray = shapesPyArray[argId];
    auto stridePyArray = stridesPyArray[argId];
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

    // Prepare shape
    assert(py::isinstance<py::array_t<std::int64_t>>(shapePyArray) &&
           shapePyArray.writeable() &&
           "shape should be writable py::array_t<std::int64_t>");
    const int64_t *shape =
        reinterpret_cast<const int64_t *>(shapePyArray.mutable_data());

    // Prepare stride
    assert(py::isinstance<py::array_t<std::int64_t>>(stridePyArray) &&
           stridePyArray.writeable() &&
           "stride should be writable py::array_t<std::int64_t>");
    const int64_t *stride =
        reinterpret_cast<const int64_t *>(stridePyArray.mutable_data());

    // Prepare ndim
    auto ndim = shapePyArray.size();

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
#ifndef ENABLE_PYRUNTIME_LIGHT
    else if (py::isinstance<py::array_t<float_16>>(inputPyArray))
      dtype = ONNX_TYPE_FLOAT16;
#endif
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
      void *tensorBuffer = generateOMTensorBufferForStringData(inputPyArray);
      if (tensorBuffer == NULL)
        throw std::runtime_error(reportPythonError(
            "fail to allocate Tensor buffer for string data"));
      inputOMTensor = omTensorCreateWithOwnership(
          tensorBuffer, shape, ndim, dtype, /*own_data=*/true);

      omTensorSetStridesWithPyArrayStrides(inputOMTensor, stride);
      // omTensorPrint("PyExecutionSessionBase::pyRun: %t %d", inputOMTensor);
    } else if (std::is_same<int64_t, pybind11::ssize_t>::value) {
      inputOMTensor =
          omTensorCreateWithOwnership(dataPtr, shape, ndim, dtype, ownData);
      omTensorSetStridesWithPyArrayStrides(inputOMTensor, stride);
    } else {
      std::vector<int64_t> safeShape(shape, shape + ndim);
      std::vector<int64_t> safeStrides(stride, stride + ndim);
      inputOMTensor = omTensorCreateWithOwnership(
          dataPtr, safeShape.data(), ndim, dtype, ownData);
      omTensorSetStridesWithPyArrayStrides(inputOMTensor, safeStrides.data());
    }
    omts.emplace_back(inputOMTensor);
  }
  TIMING_STOP_PRINT(process_input);

  // 2. Call entry point.
  TIMING_INIT_START(inference);
  auto *wrappedInput = omTensorListCreate(&omts[0], omts.size());
  auto *wrappedOutput = _entryPointFunc(wrappedInput);
  if (!wrappedOutput)
    throw std::runtime_error(reportErrnoError());
  TIMING_STOP_PRINT(inference);

  // 3. Process outputs.
  TIMING_INIT_START(process_output);
  std::vector<py::array> outputPyArrays;
  for (int64_t i = 0; i < omTensorListGetSize(wrappedOutput); i++) {
    TIMING_INIT_START(process_output_types);
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
    TIMING_STOP_PRINT(process_output_types);

    TIMING_INIT_START(process_output_pyarray);
    // Use data pointer to indicate to the numpy array where the data is, and
    // use allocated pointer for the custom deallocator. These two pointers may
    // be different when trying to allocate data that must be at specific
    // boundaries. Data pointer will be at the custom boundary, but alloc will
    // be whatever the malloc returned.
    void *omtAllocPtr = omTensorGetAllocatedPtr(omt);
    void *omtDataPtr = omTensorGetDataPtr(omt);
    // Check if the return value is a static constant, which cannot be freed and
    // thus would have been created with the "owning" flag being false.
    if (omTensorGetOwning(omt)) {
      // CWe have a regular tensor which we will need to free at the right time.
      // Create the capsule that points to the data to be freed (allocated
      // pointer).
      py::capsule free_data_with_allocate_ptr(
          omtAllocPtr, [](void *ptr) { free(ptr); });
      // Set owning to false as we migrate the ownership to python
      omTensorSetOwning(omt, false);
      // Pass the py::capsule to the numpy array for proper bookkeeping.
      outputPyArrays.emplace_back(
          py::array(dtype, shape, omtDataPtr, free_data_with_allocate_ptr));
    } else {
      // We have a constant, its a very rare case, just do like in the past:
      // copy.
      outputPyArrays.emplace_back(py::array(dtype, shape, omtDataPtr));
    }
    TIMING_STOP_PRINT(process_output_pyarray);
  }
  TIMING_STOP_PRINT(process_output);

  TIMING_INIT_START(delete_out_lists);
  omTensorListDestroy(wrappedOutput);
  TIMING_STOP_PRINT(delete_out_lists);

  TIMING_INIT_START(delete_in_lists);
  omTensorListDestroy(wrappedInput);
  TIMING_STOP_PRINT(delete_in_lists);

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
