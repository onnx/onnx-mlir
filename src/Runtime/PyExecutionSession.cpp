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

namespace onnx_mlir {

PyExecutionSession::PyExecutionSession(
    std::string sharedLibPath, std::string tag, bool defaultEntryPoint)
    : onnx_mlir::PyExecutionSessionBase(sharedLibPath, tag, defaultEntryPoint) {
}

} // namespace onnx_mlir
