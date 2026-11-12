/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Manual white-box test of OMMemoryCache.c (see README.md): cold requests of
// large bins (reuse distance in bytes, OM_MEMORY_CACHE_COLD) and the default
// largest cached size. Usage: test_cold [off]  ("off": OM_MEMORY_CACHE_COLD=0).

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "OMMemoryCache.c"

void *OMHugePageMalloc(size_t size) { return malloc(size); }

static OMCacheBin *binOf(size_t size) { return &cache.bins[binIndexFor(size)]; }

// Hand out (and free) more than 64 MB of 1 MB buffers: a long idle gap for
// every other bin.
static void idleGap(void) {
  for (int i = 0; i < 80; i++)
    OMCacheFreeSized(OMCacheAlloc(MB), MB);
}

int main(int argc, char **argv) {
  int off = argc > 1 && strcmp(argv[1], "off") == 0;
  // The test sets its own configuration (run.sh unsets OM_MEMORY_CACHE_*).
  setenv("OM_MEMORY_CACHE_COLD", off ? "0" : "64M", 1);
  ensureInit();
  assert(cache.enabled && cache.coldBytes == (off ? 0 : 64 * MB));
  assert(cache.maxCachedSize == 256 * MB); // The default largest cached size.
  OMCacheBin *big = binOf(8 * MB);

  // 1. Hot: an 8 MB buffer reused right after each free stays cached.
  for (int i = 0; i < 20; i++)
    OMCacheFreeSized(OMCacheAlloc(8 * MB), 8 * MB);
  assert(big->statMisses == 1 && big->statHits == 19);
  assert(big->statColdReleases == 0 && big->statNumAvailable == 1);

  // 2. Cold: after more than 64 MB handed out elsewhere, the next request is
  // cold. The kept (cold) buffer is returned, the request gets a fresh one,
  // and that one is returned at its free: 2 cold releases, nothing kept.
  idleGap();
  void *a = OMCacheAlloc(8 * MB);
  if (off) {
    assert(big->statHits == 20 && big->statColdReleases == 0);
    OMCacheFreeSized(a, 8 * MB);
    assert(big->statNumAvailable == 1);
    printf("test_cold (off): ALL OK\n");
    return 0;
  }
  assert(big->statHits == 19 && big->statMisses == 2);
  assert(big->statColdReleases == 1 && big->statNumAvailable == 0);
  assert(big->statMaxReuseDistance >= 64 * MB);
  OMCacheFreeSized(a, 8 * MB);
  assert(big->statColdReleases == 2 && big->statNumAvailable == 0);

  // 3. Hot again: a request right after that free is hot. It misses (nothing
  // was kept), and its buffer is kept when freed.
  void *b = OMCacheAlloc(8 * MB);
  assert(big->statMisses == 3);
  OMCacheFreeSized(b, 8 * MB);
  assert(big->statColdReleases == 2 && big->statNumAvailable == 1);

  // 4. The decision follows each buffer. X and Y in use; free X; long gap;
  // A is cold (X's kept buffer is returned, A is fresh); free Y (its request
  // was hot: kept); B, right after, is hot and gets Y's buffer. Freeing A
  // returns it, though the bin's latest request (B) was hot.
  OMCacheTrim();
  void *x = OMCacheAlloc(8 * MB), *y = OMCacheAlloc(8 * MB);
  OMCacheFreeSized(x, 8 * MB);
  idleGap();
  uint64_t releases = big->statColdReleases;
  void *ca = OMCacheAlloc(8 * MB); // Cold: X's buffer returned.
  assert(big->statColdReleases == releases + 1);
  OMCacheFreeSized(y, 8 * MB);     // Kept.
  void *cb = OMCacheAlloc(8 * MB); // Hot: gets Y's buffer.
  assert(cb == y);
  OMCacheFreeSized(ca, 8 * MB); // Returned.
  assert(big->statColdReleases == releases + 2);
  OMCacheFreeSized(cb, 8 * MB); // Kept.
  assert(big->statColdReleases == releases + 2 && big->statNumAvailable == 1);

  // 5. Bins below 4 MB are never cold, even after a long gap.
  OMCacheBin *mid = binOf(2 * MB);
  OMCacheFreeSized(OMCacheAlloc(2 * MB), 2 * MB);
  idleGap();
  OMCacheFreeSized(OMCacheAlloc(2 * MB), 2 * MB);
  assert(mid->statColdReleases == 0 && mid->statNumAvailable == 1);
  assert(mid->statHits >= 1);

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
