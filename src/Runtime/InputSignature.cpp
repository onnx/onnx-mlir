/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- InputSignature.cpp - Get Input Signature API Func Impl---===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains runtime API implementation to return the model's
// input signature as a JSON string
//
//===----------------------------------------------------------------------===//

#include "InputSignature.h"
#include <stdlib.h>
#include <string.h>

extern char _in_signature;

char* inputSignature() {
  int size = strlen(&_in_signature)+1;
  char *buffer;

  buffer = (char *)malloc(size);
  memcpy(buffer, &_in_signature, size+1);
  return buffer;
}
