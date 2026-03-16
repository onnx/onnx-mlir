/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- metal_csvquery.h - Metal C CSVQUERY Interface --------------------===//
//
// Copyright 2026 IBM Corporation
//
// =============================================================================
//
// This file contains the interface for Metal C CSVQUERY functionality.
//
//===----------------------------------------------------------------------===//

#ifndef METAL_CSVQUERY_H
#define METAL_CSVQUERY_H

#ifdef __cplusplus
extern "C" {
#endif

/// Get the pathname of the module containing this function.
///
/// Uses z/OS CSVQUERY system service to retrieve the pathname of the
/// load module (executable or shared library) that contains this function.
///
/// \param[out] output_buffer Buffer to receive the pathname string
/// \param[in]  buffer_size   Size of the output buffer in bytes
///
/// \return Length of the pathname on success, 0 on error
///
#pragma linkage(metal_get_module_path, OS)
int metal_get_module_path(char* output_buffer, int buffer_size);

#ifdef __cplusplus
}
#endif

#endif // METAL_CSVQUERY_H
