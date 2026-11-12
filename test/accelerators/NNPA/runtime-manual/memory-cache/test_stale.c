/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Manual white-box test of OMMemoryCache.c (see README.md): stale descriptors.
//
// If someone releases a cache buffer with plain free(), its descriptor stays in
// the registry; when posix_memalign later returns the same address for a new
// buffer, the cache must replace the stale descriptor and repair its counts.
// The test simulates the address reuse without freeing anything: it makes
// posix_memalign return the address of a buffer that is still registered.

#include <assert.h>
#include <stdlib.h>

static void *forcedAddress = NULL; // Next posix_memalign result, if set.
static int testPosixMemalign(void **p, size_t align, size_t size) {
  if (forcedAddress) {
    *p = forcedAddress;
    forcedAddress = NULL;
    return 0;
  }
  return posix_memalign(p, align, size);
}
#define posix_memalign testPosixMemalign
#include "OMMemoryCache.c"
#undef posix_memalign

void *OMHugePageMalloc(size_t size) { return malloc(size); }

int main(void) {
  // A 320 KB-bin buffer in use, which "the host" will release with free().
  void *lost = OMCacheAlloc(320 * KB);
  int lostBin = binIndexFor(320 * KB);
  assert(cache.bins[lostBin].numInUse == 1 && cache.numRegistered == 1);
  // Its memory is large enough for the 384 KB request below.
  void *memory = NULL;
  int rc = posix_memalign(&memory, 4096, 384 * KB);
  assert(rc == 0 && memory);
  // Simulate free(lost) + reuse: register a new 384 KB buffer at lost's
  // address. The cache never touches the memory, so pointing it at lost's
  // address while lost is still allocated is safe.
  forcedAddress = lost;
  void *reused = OMCacheAlloc(384 * KB);
  assert(reused == lost);
  assert(cache.statStaleDescs == 1);
  assert(cache.bins[lostBin].numInUse == 0); // Stale in-use count repaired.
  assert(cache.bins[binIndexFor(384 * KB)].numInUse == 1);
  assert(cache.numRegistered == 1);
  assert(findBufferDesc(reused)->bufferSize == 384 * KB);
  // Give it back and trim: the cache frees lost's memory exactly once.
  OMCacheFree(reused);
  OMCacheTrim();
  assert(cache.numRegistered == 0 && cache.availableBytes == 0);
  free(memory);
  printf("test_stale: ALL OK\n");
  return 0;
}
