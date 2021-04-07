/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- OutputSignature.cpp - Get Output Signature API Func Impl---===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains runtime API implementation to return the model's
// output signature as a JSON string
//
//===----------------------------------------------------------------------===//

#include "OutputSignature.h"
#include <stdlib.h>
#include <string.h>

extern char _out_signature;

char *outputSignature() {
  int size = strlen(&_out_signature) + 1;
  char *buffer;

  buffer = (char *)malloc(size);
  memcpy(buffer, &_out_signature, size + 1);
  return buffer;
}
