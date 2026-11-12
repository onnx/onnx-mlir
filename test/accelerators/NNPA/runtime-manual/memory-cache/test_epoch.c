/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Manual white-box test of OMMemoryCache.c (see README.md): the epoch policy
// (OMCacheEpoch, buffers of 16 MB or more not reused across inferences) and
// the default largest cached size.
// Usage: test_epoch [epoch|none|default]   (OM_MEMORY_CACHE_POLICY; "default"
// leaves it unset: epoch with transparent huge pages, none without).

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "OMMemoryCache.c"

void *OMHugePageMalloc(size_t size) { return malloc(size); }

static OMCacheBin *binOf(size_t size) { return &cache.bins[binIndexFor(size)]; }

static void allocFree(size_t size) {
  OMCacheFreeSized(OMCacheAlloc(size), size);
}

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "epoch";
  if (strcmp(mode, "default") != 0)
    setenv("OM_MEMORY_CACHE_POLICY", mode, 1);
  ensureInit();
  assert(cache.enabled && cache.maxCachedSize == 256 * MB);
  if (strcmp(mode, "default") == 0) {
    assert(cache.policy ==
           (cache.thpAvailable ? CACHE_POLICY_EPOCH : CACHE_POLICY_NONE));
    printf("test_epoch (default): policy %s, THP %s: ALL OK\n",
        cache.policy == CACHE_POLICY_EPOCH ? "epoch" : "none",
        cache.thpAvailable ? "on" : "off");
    return 0;
  }
  int none = strcmp(mode, "none") == 0;
  assert(cache.policy == (none ? CACHE_POLICY_NONE : CACHE_POLICY_EPOCH));
  OMCacheBin *big = binOf(32 * MB), *small = binOf(8 * MB);

  // 1. Within an epoch, a large buffer is reused.
  uint64_t e1 = OMCacheEpoch(0);
  assert(e1 != 0);
  for (int i = 0; i < 5; i++) {
    allocFree(32 * MB);
    allocFree(8 * MB);
  }
  assert(big->statMisses == 1 && big->statHits == 4);
  assert(small->statMisses == 1 && small->statHits == 4);

  // 2. At the end of the epoch, available buffers of 16 MB or more are
  // returned; smaller ones are kept.
  assert(OMCacheEpoch(e1) == 0);
  if (none) {
    assert(big->statNumAvailable == 1 && big->statStaleReleases == 0);
    printf("test_epoch (none): ALL OK\n");
    return 0;
  }
  assert(big->statNumAvailable == 0 && big->statStaleReleases == 1);
  assert(small->statNumAvailable == 1);

  // 3. Next epoch: the large request misses (fresh), the small one hits.
  uint64_t e2 = OMCacheEpoch(0);
  allocFree(32 * MB);
  allocFree(8 * MB);
  assert(big->statMisses == 2 && small->statHits == 5);

  // 4. No end call (e.g. a failed inference): at the next start, the large
  // buffer kept in e2 is stale and returned at the next request.
  uint64_t releases = big->statStaleReleases;
  uint64_t e3 = OMCacheEpoch(0);
  (void)e2;
  assert(big->statNumAvailable == 1);
  void *a = OMCacheAlloc(32 * MB); // Stale head returned, fresh buffer.
  assert(big->statStaleReleases == releases + 1 && big->statMisses == 3);
  OMCacheFreeSized(a, 32 * MB);

  // 5. Concurrent inferences: e3 ends after e4 started, so nothing is
  // released at e3's end (e4 may reuse e3's buffers... after its own frees).
  uint64_t e4 = OMCacheEpoch(0);
  releases = big->statStaleReleases;
  OMCacheEpoch(e3);
  assert(big->statStaleReleases == releases && big->statNumAvailable == 1);
  OMCacheEpoch(e4); // The latest epoch ends: large buffers are returned.
  assert(big->statNumAvailable == 0 && big->statStaleReleases == releases + 1);

  // 6. Before any epoch (no OMCacheEpoch call yet in a process) nothing is
  // stale; covered by test_cache, which never calls OMCacheEpoch.

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
  assert(cache.statEpochsStarted == 4 &&
         cache.statEpochsEnded == 3); // e2 never ends.
  printf("test_epoch: ALL OK\n");
  return 0;
}
