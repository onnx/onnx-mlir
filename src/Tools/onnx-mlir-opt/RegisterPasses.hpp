/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- RegisterPasses.hpp -------------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace onnx_mlir {

// Makes select mlir and onnx-mlir passes available as command-line options.
void registerPasses(int optLevel);

} // namespace onnx_mlir
