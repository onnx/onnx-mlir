/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Manual white-box test of OMMemoryCache.c (see README.md): allocation
// failures. posix_memalign is made to fail a given number of times.

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

static int failNext = 0; // Number of next posix_memalign calls that fail.
static int testPosixMemalign(void **p, size_t align, size_t size) {
  if (failNext > 0) {
    failNext--;
    return ENOMEM;
  }
  return posix_memalign(p, align, size);
}
#define posix_memalign testPosixMemalign
#include "OMMemoryCache.c"
#undef posix_memalign

void *OMHugePageMalloc(size_t size) { return malloc(size); }

int main(void) {
  // 1. Fail once while buffers are available: trim, then the retry succeeds.
  void *a = OMCacheAlloc(2 * MB), *b = OMCacheAlloc(3 * MB);
  OMCacheFree(a);
  OMCacheFree(b);
  failNext = 1;
  void *c = OMCacheTryAlloc(5 * MB);
  assert(c && cache.statAllocFailureTrims == 1 && cache.statFailedAllocs == 0);
  assert(cache.availableBytes == 0 && cache.numRegistered == 1);
  assert(cache.bins[binIndexFor(5 * MB)].numInUse == 1);

  // 2. Fail with nothing available: no trim possible; NULL, in-use count
  // undone.
  failNext = 2;
  void *d = OMCacheTryAlloc(6 * MB);
  assert(d == NULL && cache.statFailedAllocs == 1);
  assert(cache.bins[binIndexFor(6 * MB)].numInUse == 0);

  // 3. Fail on the retry too (a buffer is available): one trim, then NULL.
  OMCacheFree(c);
  failNext = 2;
  void *e = OMCacheTryAlloc(7 * MB);
  assert(e == NULL && cache.statAllocFailureTrims == 2);
  assert(cache.statFailedAllocs == 2 && cache.numRegistered == 0);
  assert(cache.bins[binIndexFor(7 * MB)].numInUse == 0);

  printf("test_alloc_failure: ALL OK\n");
  return 0;
}
