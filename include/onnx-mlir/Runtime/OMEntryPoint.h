/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- OMEntryPoint.h - OMEntryPoint Declaration header ---------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of OMEntryPoint API functions.
//
//===----------------------------------------------------------------------===//

#ifdef __cplusplus
#pragma once

extern "C" {
#endif

/**
 * \brief Return all entry point names in a model.
 *
 * @return an array of strings. The array ends with NULL. For example:
 * ["run_add", "run_sub", NULL].
 */
const char **omQueryEntryPoints();

#ifdef __cplusplus
}
#endif
