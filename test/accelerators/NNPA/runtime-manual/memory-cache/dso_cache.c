/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared library for test_dlclose_fork.c (see README.md): a stand-in for a
// model .so, with its own copy of the cache.

#include <stdlib.h>
#include <string.h>

#include "OMMemoryCache.c"

void *OMHugePageMalloc(size_t size) { return malloc(size); }

// Use the cache once (registers the fork handlers on Linux). Return 0 on
// success.
int dsoUseCache(void) {
  void *buffer = OMCacheAlloc(2 * MB);
  if (!buffer)
    return 1;
  memset(buffer, 1, 4096);
  OMCacheFree(buffer);
  return 0;
}
