/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- PyExecutionSession.cpp - PyExecutionSession Implementation -----===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementations of PyExecutionSession class, which helps
// python programs interact with compiled binary model libraries.
//
//===----------------------------------------------------------------------===//

#include "PyExecutionSession.hpp"
#include "PyFloat16.hpp"
#include "src/Runtime/OMTensorHelper.hpp"
#include "src/Runtime/OMTensorInternal.h"

#define OM_DRIVER_TIMING 0 /* 1 for timing, 0 for no timing/overheads */
#include "src/Runtime/OMInstrumentHelper.h"

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

PyExecutionSession::PyExecutionSession(
    std::string sharedLibPath, std::string tag, bool defaultEntryPoint)
    : onnx_mlir::PyExecutionSessionBase(sharedLibPath, tag, defaultEntryPoint) {
}

} // namespace onnx_mlir
