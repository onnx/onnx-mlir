/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- OMSignature.inc - C/C++ Model Input/Output Signature API Impl ---===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains runtime API implementation to return the model's
// input and output signature as a JSON string
//
//===----------------------------------------------------------------------===//

extern const char _in_signature[];
extern const char _out_signature[];

#include "onnx-mlir/Runtime/OMSignature.h"

const char *omInputSignature() {
  return _in_signature;
}

const char *omOutputSignature() {
  return _out_signature;
}
