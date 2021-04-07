/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- OutputSignature.h - Get JSON output signature ---===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains runtime API declarations to extract get the output
// signature of the model as a JSON string.
// The caller is responsible for freeing the string returned
//
//===----------------------------------------------------------------------===//

#pragma once

#include <stdint.h>

extern "C" {
char *outputSignature();
}