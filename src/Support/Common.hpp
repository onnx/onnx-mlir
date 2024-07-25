/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====--------------- Common.hpp - Common Utilities -----------------------===//
//
// Copyright 2021-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common utilities and support code.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_COMMON_H
#define ONNX_MLIR_COMMON_H

#if defined(__GNUC__) || defined(__clang__)
#define ATTRIBUTE(x) __attribute__((x))
#else
#define ATTRIBUTE(x)
#endif
#endif
