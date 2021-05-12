/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- OMSignature.h - OMSignature Declaration header -----------===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of OMSignature API functions.
//
//===----------------------------------------------------------------------===//

#ifdef __cplusplus
#pragma once

extern "C" {
#endif

/**
 * \brief Return the model's input signature as a JSON string.
 *
 * @return pointer to input signature JSON string
 */
const char *omInputSignature();

/**
 * \brief Return the model's output signature as a JSON string.
 *
 * @return pointer to output signature JSON string
 */
const char *omOutputSignature();

#ifdef __cplusplus
}
#endif
