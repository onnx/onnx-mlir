/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- OMMemoryCache.h ---------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Reuse cache for the per-inference temporary buffers of a compiled model.
//
// Environment variables, read once at the first call into the cache (each
// model .so has its own cache and reads them on its own). Sizes are a number
// with an optional K, M or G suffix (powers of 1024) and an optional B; a
// value that does not parse or overflows size_t is ignored (default used).
//   OM_MEMORY_CACHE_MAX    Ceiling on the bytes of available (cached, not in
//                          use) buffers; the oldest are evicted above it.
//                          Default 1G. 0 disables the cache: every call then
//                          behaves as before the cache (OMHugePageMalloc /
//                          free).
//   OM_MEMORY_CACHE_LARGE  Largest request that is cached; larger ones go to
//                          OMHugePageMalloc. Default 256M, at most 1T. A value
//                          below 128K disables the cache.
//   OM_MEMORY_CACHE_COLD   Buffers of 4 MB and more are kept only while their
//                          bin is hot. A request is cold when the cache has
//                          handed out at least this many bytes since its bin's
//                          last free: the bin's available buffers are returned
//                          to the system, and the request gets a fresh buffer,
//                          also returned when freed. Default 1G. 0 keeps all
//                          buffers.
//   OM_MEMORY_CACHE_STATS  Any value other than empty or starting with '0':
//                          print statistics to stderr when the model .so is
//                          unloaded or the process exits.
// See OMMemoryCache.c for the design and for debugging advice.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_OM_MEMORY_CACHE_H
#define ONNX_MLIR_OM_MEMORY_CACHE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Allocate a temporary buffer that is freed with OMCacheFree before the
 * inference ends. Requests between 128 KB and 256 MB are served from a cache of
 * previously freed buffers; other requests, or all requests when the cache is
 * disabled, go to OMHugePageMalloc. Called from compiled code. */
void *OMCacheAlloc(size_t size);

/* Return a buffer to the cache. A pointer that the cache does not own is
 * passed to free(), so any malloc'ed pointer is accepted. NULL is ignored.
 * The reverse does not hold: a buffer from OMCacheAlloc or OMCacheTryAlloc
 * must be released with OMCacheFree, never with plain free(). */
void OMCacheFree(void *buffer);

/* Same as OMCacheFree, given the size that was passed to OMCacheAlloc or
 * OMCacheTryAlloc for this buffer. A size the cache never serves (outside its
 * range, or cache disabled) is freed with free() at once, without taking the
 * cache mutex or looking the buffer up. Called from compiled code, which
 * passes the same size value as to the allocation, and from zdnnx. */
void OMCacheFreeSized(void *buffer, size_t size);

/* Return all available buffers (cached, not in use) to the system. Intended
 * for hosts that keep an idle model loaded. */
void OMCacheTrim(void);

/* Return a cached buffer of at least size bytes, or NULL if the request is
 * not in the cached range, the cache is disabled, or the allocation failed.
 * Buffers are at least 4 KB aligned, and 1 MB aligned from 1 MB up. The caller
 * handles NULL with its own allocator. Used by the zdnnx runtime. */
void *OMCacheTryAlloc(size_t size);

#ifdef __cplusplus
}
#endif
#endif
