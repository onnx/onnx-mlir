/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- OMTensor.cpp - OMTensor C++ Implementation -------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementation of OMTensor and data structures and
// helper functions. It is compiled into OMTensorUtils, which is linked by the
// Execution Session (C++/Python). Compiler-internal functions that are only
// called from compiler-generated wrapper code (omTensorCreateUntyped,
// omTensorSetDataPtr, omTensorSetShape, omTensorSetStrides,
// omTensorSetDataType) are excluded here via OM_EXECUTION_SESSION_ONLY;
// they are provided by cruntime (OMTensor.c) which is embedded in model.so.
//
//===----------------------------------------------------------------------===//

#define OM_EXECUTION_SESSION_ONLY
#include "OMTensor.inc"
