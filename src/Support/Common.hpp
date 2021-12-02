/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====--------------- Common.hpp - Common Utilities -----------------------===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common utilities and support code.
//
//===----------------------------------------------------------------------===//

#pragma once

#if defined(__GNUC__) || defined(__clang__)
#define ATTRIBUTE(x) __attribute__((x))
#else
#define ATTRIBUTE(x)
#endif
