/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- RegisterPasses.hpp -------------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_REGISTER_PASSES_H
#define ONNX_MLIR_REGISTER_PASSES_H

namespace onnx_mlir {

// Makes select mlir and onnx-mlir passes available as command-line options.
void registerPasses(int optLevel);

} // namespace onnx_mlir
#endif
