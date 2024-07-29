/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====--------------- SuppressWarnings.h - Suppress Warnings --------------===//
//
// Copyright 2021-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains support code used to suppress warnings.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_SUPPRESS_WARNINGS_H
#define ONNX_MLIR_SUPPRESS_WARNINGS_H

// clang-format off
#if defined(SUPPRESS_THIRD_PARTY_WARNINGS)
  #if defined(__clang__)
    #define SUPPRESS_WARNINGS_PUSH                                          \
      _Pragma("clang diagnostic push")                                      \
      _Pragma("clang diagnostic ignored \"-Wcast-qual\"")                   \
      _Pragma("clang diagnostic ignored \"-Wstring-conversion\"")           \
      _Pragma("clang diagnostic ignored \"-Wmissing-field-initializers\"")  \
      _Pragma("clang diagnostic ignored \"-Wsuggest-override\"")            \
      _Pragma("clang diagnostic ignored \"-Wc++98-compat-extra-semi\"")

    #define SUPPRESS_WARNINGS_POP _Pragma("clang diagnostic pop")
  #elif defined(__GNUC__)
    #define SUPPRESS_WARNINGS_PUSH                                          \
      _Pragma("GCC diagnostic push")                                        \
      _Pragma("GCC diagnostic ignored \"-Wcast-qual\"")                     \
      _Pragma("GCC diagnostic ignored \"-Wmissing-field-initializers\"")    \
      _Pragma("GCC diagnostic ignored \"-Wsuggest-override\"")

    #define SUPPRESS_WARNINGS_POP _Pragma("GCC diagnostic pop")
  #else
    #define SUPPRESS_WARNINGS_PUSH
    #define SUPPRESS_WARNINGS_POP
  #endif
#else
  #define SUPPRESS_WARNINGS_PUSH
  #define SUPPRESS_WARNINGS_POP
#endif
// clang-format on
#endif
