/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Manual white-box test of OMMemoryCache.c (see README.md): cold large buffers
// (reuse distance, OM_MEMORY_CACHE_COLD) and the default largest cached size.
// Usage: test_cold [off]   ("off" runs with OM_MEMORY_CACHE_COLD=0).

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "OMMemoryCache.c"

void *OMHugePageMalloc(size_t size) { return malloc(size); }

static OMCacheBin *binOf(size_t size) { return &cache.bins[binIndexFor(size)]; }

// Advance the free clock by n cache frees of small (256 KB) buffers.
static void idleFrees(int n) {
  for (int i = 0; i < n; i++)
    OMCacheFreeSized(OMCacheAlloc(256 * KB), 256 * KB);
}

int main(int argc, char **argv) {
  int off = argc > 1 && strcmp(argv[1], "off") == 0;
  // The test sets its own configuration (run.sh unsets OM_MEMORY_CACHE_*).
  setenv("OM_MEMORY_CACHE_COLD", off ? "0" : "8", 1);
  ensureInit();
  assert(cache.enabled && cache.coldDistance == (off ? 0 : 8));
  assert(cache.maxCachedSize == 256 * MB); // The default largest cached size.
  OMCacheBin *big = binOf(8 * MB);

  // 1. Hot: an 8 MB buffer reused right after each free stays cached.
  for (int i = 0; i < 20; i++)
    OMCacheFreeSized(OMCacheAlloc(8 * MB), 8 * MB);
  assert(big->statMisses == 1 && big->statHits == 19);
  assert(big->statColdReleases == 0 && big->statNumAvailable == 1);

  // 2. Cold: after more than 8 other frees, the next request is cold. It is a
  // hit (the buffer was kept), but the buffer is released when freed.
  idleFrees(20);
  void *a = OMCacheAlloc(8 * MB);
  assert(big->statHits == 20);
  OMCacheFreeSized(a, 8 * MB);
  if (off) {
    assert(big->statColdReleases == 0 && big->statNumAvailable == 1);
    printf("test_cold (off): ALL OK\n");
    return 0;
  }
  assert(big->statColdReleases == 1 && big->statNumAvailable == 0);
  assert(findBufferDesc(a) == NULL || findBufferDesc(a)->bufferSize != 8 * MB);

  // 3. Hot again: a request right after that free is hot. It misses (nothing
  // was kept), and its buffer is kept when freed.
  void *b = OMCacheAlloc(8 * MB);
  assert(big->statMisses == 2);
  OMCacheFreeSized(b, 8 * MB);
  assert(big->statColdReleases == 1 && big->statNumAvailable == 1);

  // 4. The decision follows each buffer: X and Y in use; free X; long gap;
  // A (cold) gets X's buffer; free Y (hot, kept); B (hot) gets Y's buffer.
  // Freeing A releases it, though the bin's latest request (B) was hot.
  OMCacheTrim();
  void *x = OMCacheAlloc(8 * MB), *y = OMCacheAlloc(8 * MB);
  OMCacheFreeSized(x, 8 * MB);
  idleFrees(20);
  void *ca = OMCacheAlloc(8 * MB); // Cold.
  OMCacheFreeSized(y, 8 * MB);     // Y was hot: kept.
  void *cb = OMCacheAlloc(8 * MB); // Hot (just after Y's free).
  uint64_t releases = big->statColdReleases;
  OMCacheFreeSized(ca, 8 * MB);
  assert(big->statColdReleases == releases + 1);
  OMCacheFreeSized(cb, 8 * MB);
  assert(big->statColdReleases == releases + 1 && big->statNumAvailable == 1);

  // 5. Bins below 4 MB are never released, even after a long gap.
  OMCacheBin *mid = binOf(2 * MB);
  OMCacheFreeSized(OMCacheAlloc(2 * MB), 2 * MB);
  idleFrees(20);
  OMCacheFreeSized(OMCacheAlloc(2 * MB), 2 * MB);
  assert(mid->statColdReleases == 0 && mid->statNumAvailable == 1);

  // Consistency: nothing in use, counts agree.
  size_t mapped = 0;
  for (size_t i = 0; i < CACHE_REGISTRY_BUCKETS; i++)
    for (OMCacheBufferDesc *d = cache.buckets[i]; d; d = d->bucketNext)
      mapped++;
  uint64_t avail = 0;
  for (int i = 0; i < cache.numBins; i++) {
    assert(cache.bins[i].numInUse == 0);
    avail += cache.bins[i].statNumAvailable;
  }
  assert(mapped == cache.numRegistered && mapped == avail);
  assert(cache.statInUseBytes == 0);
  printf("test_cold: ALL OK\n");
  return 0;
}
