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
//
//   OM_MEMORY_CACHE_MAX    Ceiling on the bytes of available (cached, not in
//                          use) buffers; the oldest are evicted above it.
//                          Default 1G. 0 disables the cache: every buffer is
//                          then allocated as before the cache (malloc below
//                          1 MB, on Linux 1 MB-aligned huge-page memory
//                          above) and freed with free().
//   OM_MEMORY_CACHE_LARGE  Largest request that is cached; larger ones are
//                          allocated uncached (on Linux, 1 MB-aligned
//                          huge-page memory). Default 256M, at most 1T. A
//                          value below 128K disables the cache.
//   OM_MEMORY_CACHE_POLICY epoch (default): a buffer of 16 MB or more freed
//                          in an earlier inference is never reused; it is
//                          returned to the system and a fresh one is mapped.
//                          none: keep all buffers. Without transparent huge
//                          pages the default is none. Other values are
//                          ignored (default used). Epochs are counted per
//                          model .so, not per session: with concurrent
//                          inferences on one model, large buffers are reused
//                          less (results are unchanged); none avoids that.
//   OM_MEMORY_CACHE_STATS  Any value other than empty or starting with '0':
//                          print statistics to stderr when the model .so is
//                          unloaded or the process exits.
//
// Build option: CMake option ONNX_MLIR_NNPA_MEMORY_CACHE (ON by default,
// except on z/OS, where the cache's benefit and defaults are not measured
// yet). OFF compiles OMMemoryCachePassThrough.c instead of OMMemoryCache.c:
// the functions below then manage memory as before the cache, and compiled
// models still link.
//
// See OMMemoryCache.c for the design and for debugging advice.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_OM_MEMORY_CACHE_H
#define ONNX_MLIR_OM_MEMORY_CACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Each model .so links its own copy of the NNPA runtime, hence its own cache.
 * Hidden visibility keeps a model's calls bound to its own cache even when
 * several model .so are loaded with RTLD_GLOBAL. OMCacheTrim stays visible: a
 * host may look it up with dlsym. */
#if defined(__linux__) || defined(__APPLE__)
#define OM_CACHE_INTERNAL __attribute__((visibility("hidden")))
#else
#define OM_CACHE_INTERNAL
#endif

/* Allocate a temporary buffer that is freed with OMCacheFree before the
 * inference ends. Requests between 128 KB and 256 MB are served from a cache of
 * previously freed buffers; other requests, or all requests when the cache is
 * disabled, are allocated uncached (malloc below 1 MB, 1 MB-aligned memory
 * advised for huge pages above on Linux; freed with free()). Size 0 gives
 * NULL. Called
 * from compiled code. */
OM_CACHE_INTERNAL void *OMCacheAlloc(size_t size);

/* Return a buffer to the cache. A pointer that the cache does not own is
 * passed to free(), so any malloc'ed pointer is accepted. NULL is ignored.
 * The reverse does not hold: a buffer from OMCacheAlloc or OMCacheTryAlloc
 * must be released with OMCacheFree, never with plain free(). */
OM_CACHE_INTERNAL void OMCacheFree(void *buffer);

/* Same as OMCacheFree, but faster: sizes without a bin skip the lock. Use it
 * only where the size is the allocation size by construction (compiled code,
 * which passes the same value as to OMCacheAlloc). A wrong size can send a
 * cache buffer to free() while it is still registered. Hand-written callers
 * (e.g. zdnnx) use OMCacheFree, which finds the buffer by address. */
OM_CACHE_INTERNAL void OMCacheFreeSized(void *buffer, size_t size);

/* Start or end an inference (epoch policy). OMCacheEpoch(0) at the start
 * returns the epoch's id (never 0); OMCacheEpoch(id) at the end returns 0.
 * Called by compiled code around the model's body. */
OM_CACHE_INTERNAL uint64_t OMCacheEpoch(uint64_t epoch);

/* Return all available buffers (cached, not in use) to the system. Intended
 * for hosts that keep an idle model loaded. */
void OMCacheTrim(void);

/* Return a cached buffer of at least size bytes, or NULL if the request is
 * not in the cached range, the cache is disabled, or the allocation failed.
 * Buffers are at least 4 KB aligned, and on Linux 1 MB aligned from 1 MB up.
 * The caller handles NULL with its own allocator. Used by the zdnnx runtime. */
OM_CACHE_INTERNAL void *OMCacheTryAlloc(size_t size);

#ifdef __cplusplus
}
#endif
#endif
