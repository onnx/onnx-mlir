/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- OMMemoryCachePassThrough.c -----------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// The OMCache* functions without a cache, compiled instead of OMMemoryCache.c
// when the CMake option ONNX_MLIR_NNPA_MEMORY_CACHE is OFF (the default on
// z/OS). Memory is managed as before the cache: OMCacheAlloc allocates as
// OMHugePageMalloc (1 MB-aligned huge-page memory from 1 MB up on Linux, plain
// malloc elsewhere), frees go to free(), and zdnnx uses its own allocator
// since OMCacheTryAlloc returns NULL. Models compiled with the cache calls
// still link and run.
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>

#include "OMMemoryCache.h"

void *OMHugePageMalloc(size_t size); // OMRuntimeNNPA.c

void *OMCacheAlloc(size_t size) {
  return size == 0 ? NULL : OMHugePageMalloc(size);
}

void OMCacheFree(void *buffer) { free(buffer); }

void OMCacheFreeSized(void *buffer, size_t size) {
  (void)size;
  free(buffer);
}

uint64_t OMCacheEpoch(uint64_t epoch) { return epoch == 0 ? 1 : 0; }

void OMCacheTrim(void) {}

void *OMCacheTryAlloc(size_t size) {
  (void)size;
  return NULL;
}
