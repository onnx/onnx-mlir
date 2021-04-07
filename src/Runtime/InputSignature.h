/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- InputSignature.h - Get JSON input signature ---===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains runtime API declarations to extract get the input
// signature of the model as a JSON string
//
//===----------------------------------------------------------------------===//

#pragma once

#include <stdint.h>

extern "C" {
char *inputSignature();
}