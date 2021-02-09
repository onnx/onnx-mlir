/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- GetEmbeddedConstPool.h - Get Embedded Const Pool API Func Decl---===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains runtime API declarations to extract constant pool values
// embedded within the shared library binary files.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <stdint.h>

extern "C" {
void *getEmbeddedConstPool(int64_t size_in_byte);
}