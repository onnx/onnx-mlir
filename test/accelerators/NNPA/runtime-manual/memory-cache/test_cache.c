/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Manual white-box test of OMMemoryCache.c (see README.md): bins, registry,
// hits, trim, double free, eviction order, threads and fork.

#include <assert.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#include "OMMemoryCache.c"

void *OMHugePageMalloc(size_t size) { return malloc(size); }

#if defined(__APPLE__)
// The runtime registers no fork handlers on macOS; register equivalent ones so
// that the fork-with-threads section tests the Linux behavior. (On Linux the
// runtime's own handlers are used; registering these too would deadlock.)
static void testLockForFork(void) { pthread_mutex_lock(&cacheMutex); }
static void testUnlockForFork(void) { pthread_mutex_unlock(&cacheMutex); }
#endif

// Check that the registry, the available lists and the counters agree.
static void check(void) {
  pthread_mutex_lock(&cacheMutex);
  size_t numMapped = 0, mappedBytes = 0, availBytes = 0, numAvail = 0;
  size_t inUseBytes = 0, inUseRequestBytes = 0;
  for (size_t i = 0; i < CACHE_REGISTRY_BUCKETS; i++)
    for (OMCacheBufferDesc *d = cache.buckets[i]; d; d = d->bucketNext) {
      numMapped++;
      mappedBytes += d->bufferSize;
      assert(hashPtr(d->buffer) == i);
      assert(findBufferDesc(d->buffer) == d);
      assert((uintptr_t)d->buffer % (d->bufferSize >= MB ? MB : 4096) == 0);
      if (d->inUse) {
        assert(d->requestSize <= d->bufferSize);
        inUseBytes += d->bufferSize;
        inUseRequestBytes += d->requestSize;
      }
    }
  for (int i = 0; i < cache.numBins; i++) {
    OMCacheBin *bin = &cache.bins[i];
    OMCacheBufferDesc *prev = NULL;
    uint64_t count = 0;
    for (OMCacheBufferDesc *d = bin->availableHead; d;
         prev = d, d = d->availableNext) {
      assert(!d->inUse && d->binIndex == i && d->availablePrev == prev);
      if (prev)
        assert(prev->freedAt > d->freedAt); // Most recently freed first.
      availBytes += d->bufferSize;
      numAvail++;
      count++;
    }
    assert(bin->availableTail == prev && count == bin->statNumAvailable);
  }
  assert(numMapped == cache.numRegistered);
  assert(mappedBytes == cache.statRegisteredBytes);
  assert(availBytes == cache.availableBytes);
  assert(inUseBytes == cache.statInUseBytes);
  assert(inUseRequestBytes == cache.statInUseRequestBytes);
  (void)numAvail;
  pthread_mutex_unlock(&cacheMutex);
}

// Many descriptors in one bucket: map, find, and remove them from the head,
// the tail and the middle of the chain.
static void collisionTest(void) {
  enum { N = 40 };
  OMCacheBufferDesc d[N];
  size_t found = 0, target = (size_t)-1;
  pthread_mutex_lock(&cacheMutex);
  size_t before = cache.numRegistered;
  for (uintptr_t a = (uintptr_t)1 << 40; found < N; a += 4096) {
    size_t h = hashPtr((void *)a);
    if (target == (size_t)-1)
      target = h;
    if (h != target)
      continue;
    memset(&d[found], 0, sizeof(d[found]));
    d[found].buffer = (void *)a;
    mapBufferToDesc(&d[found]);
    found++;
  }
  for (int i = 0; i < N; i++)
    assert(findBufferDesc(d[i].buffer) == &d[i]);
  if (cache.printStats)
    assert(cache.statMaxBucketChain >= N);
  // Removal order: head (last mapped), tail (first mapped), middle, then rest.
  int order[N], k = 0;
  order[k++] = N - 1;
  order[k++] = 0;
  order[k++] = N / 2;
  for (int i = 1; i < N - 1; i++)
    if (i != N / 2)
      order[k++] = i;
  for (int r = 0; r < N; r++) {
    removeBufferMapping(&d[order[r]]);
    assert(findBufferDesc(d[order[r]].buffer) == NULL);
    for (int q = r + 1; q < N; q++)
      assert(findBufferDesc(d[order[q]].buffer) == &d[order[q]]);
  }
  assert(cache.numRegistered == before && cache.buckets[target] == NULL);
  pthread_mutex_unlock(&cacheMutex);
  printf("collisions: %d buffers in bucket %zu\n", N, target);
}

// Sizes of one "inference": cached ones, one below 128 KB, one above 64 MB.
static const size_t trace[] = {200 * KB, 6 * MB, 6 * MB, 1536 * KB, 50 * KB,
    3 * MB, 900 * KB, 64 * MB, 70 * MB};
#define NUM_TRACE (sizeof(trace) / sizeof(trace[0]))
#define NUM_TRACE_CACHED 7

static void *worker(void *arg) {
  unsigned seed = (unsigned)(uintptr_t)arg;
  for (int it = 0; it < 2000; it++) {
    void *p[NUM_TRACE];
    for (size_t i = 0; i < NUM_TRACE; i++) {
      size_t s = trace[(i + rand_r(&seed)) % NUM_TRACE] / 8 + 4096;
      p[i] = OMCacheAlloc(s);
      memset(p[i], (int)i, 4096);
    }
    for (size_t i = 0; i < NUM_TRACE; i++) {
      assert(((unsigned char *)p[i])[100] == (unsigned char)i);
      OMCacheFree(p[i]);
    }
  }
  return NULL;
}

int main(void) {
  ensureInit();
  assert(cache.enabled);
#if defined(__APPLE__)
  pthread_atfork(testLockForFork, testUnlockForFork, testUnlockForFork);
#endif

  // Bins and rounding.
  printf("%d bins up to %zu MB:", cache.numBins, cache.maxCachedSize / MB);
  for (int i = 0; i < cache.numBins; i++)
    printf(" %zu", cache.bins[i].bufferSize / KB);
  printf(" (KB)\n");
  assert(cache.bins[binIndexFor(128 * KB)].bufferSize == 128 * KB);
  assert(cache.bins[binIndexFor(128 * KB + 1)].bufferSize == 192 * KB);
  assert(cache.bins[binIndexFor(MB + 1)].bufferSize == 2 * MB);
  assert(cache.bins[binIndexFor(9 * MB)].bufferSize == 10 * MB);
  assert(cache.bins[binIndexFor(33 * MB)].bufferSize == 40 * MB);
  assert(cache.bins[binIndexFor(64 * MB)].bufferSize == 64 * MB);
  assert(OMCacheTryAlloc(64 * MB + 1) == NULL);
  assert(OMCacheTryAlloc(100 * KB) == NULL);

  // 1. Repeated inference: misses only on the first one.
  for (int inf = 0; inf < 5; inf++) {
    void *p[NUM_TRACE];
    for (size_t i = 0; i < NUM_TRACE; i++)
      p[i] = OMCacheAlloc(trace[i]);
    for (size_t i = 0; i < NUM_TRACE; i++)
      OMCacheFree(p[i]);
    check();
  }
  uint64_t hits = 0, misses = 0;
  for (int i = 0; i < cache.numBins; i++) {
    hits += cache.bins[i].statHits;
    misses += cache.bins[i].statMisses;
  }
  printf("repeat: hits %llu, misses %llu, available %zu KB\n",
      (unsigned long long)hits, (unsigned long long)misses,
      cache.availableBytes / KB);
  assert(misses == NUM_TRACE_CACHED && hits == 4 * NUM_TRACE_CACHED);

  // 2. Registry collisions.
  collisionTest();
  check();

  // 3. Trim.
  OMCacheTrim();
  check();
  assert(cache.availableBytes == 0 && cache.numRegistered == 0);

  // 3b. Peak in use at once vs registered: a 6 MB buffer, then (after it is
  // freed) a 2 MB one. Registered holds both bins (7 MB + 2 MB); the peak in
  // use is only the larger one; requested bytes are below the bin sizes.
  {
    cache.statPeakInUseBytes = cache.statPeakInUseRequestBytes = 0;
    cache.statPeakRegisteredBytes = 0;
    void *big = OMCacheAlloc(6 * MB + 64);
    OMCacheFree(big);
    void *small = OMCacheAlloc(1536 * KB);
    OMCacheFree(small);
    check();
    printf("phases: peak in use %zu KB (requested %zu KB), registered %zu KB\n",
        cache.statPeakInUseBytes / KB, cache.statPeakInUseRequestBytes / KB,
        cache.statPeakRegisteredBytes / KB);
    assert(cache.statPeakInUseBytes == 7 * MB);
    assert(cache.statPeakInUseRequestBytes == 6 * MB + 64);
    assert(cache.statPeakRegisteredBytes == 9 * MB);
    assert(cache.statInUseBytes == 0 && cache.statInUseRequestBytes == 0);
    OMCacheTrim();
  }

  // 4. Buffers the cache does not own, and NULL.
  OMCacheFree(malloc(10));
  OMCacheFree(NULL);

  // 4b. Sized free: sizes without a bin are freed without the lock; a cache
  // buffer freed with its size goes back to its bin.
  {
    uint64_t fastBefore = cache.statFastFrees;
    void *small = OMCacheAlloc(100 * KB); // Below the cache: plain malloc.
    OMCacheFreeSized(small, 100 * KB);
    void *large = OMCacheAlloc(70 * MB); // Above the cache (64M here).
    OMCacheFreeSized(large, 70 * MB);
    if (cache.printStats)
      assert(cache.statFastFrees == fastBefore + 2);
    void *cached = OMCacheAlloc(2 * MB);
    size_t availBefore = cache.bins[binIndexFor(2 * MB)].statNumAvailable;
    OMCacheFreeSized(cached, 2 * MB);
    assert(cache.bins[binIndexFor(2 * MB)].statNumAvailable == availBefore + 1);
    assert(cache.statFastFrees == fastBefore + (cache.printStats ? 2 : 0));
    OMCacheFreeSized(NULL, 0);
    check();
  }

  // 5. Double free is ignored.
  void *twice = OMCacheAlloc(MB);
  OMCacheFree(twice);
  OMCacheFree(twice);
  assert(cache.statDoubleFrees == 1);
  check();

  // 6. Ceiling: 8 buffers of 3 MB under a 10 MB ceiling keep the 3 youngest.
  OMCacheTrim();
  size_t savedMax = cache.maxAvailableBytes;
  cache.maxAvailableBytes = 10 * MB;
  void *q[8];
  for (int i = 0; i < 8; i++)
    q[i] = OMCacheAlloc(3 * MB);
  for (int i = 0; i < 8; i++)
    OMCacheFree(q[i]);
  check();
  printf("ceiling 10 MB: available %zu KB, evictions %llu\n",
      cache.availableBytes / KB, (unsigned long long)cache.statEvictions);
  assert(cache.availableBytes <= 10 * MB);
  assert(findOldestAvailable()->buffer == q[5]);
  cache.maxAvailableBytes = savedMax;

  // 7. Eviction order across bins: free buffers of several sizes in a known
  // order; then each free that crosses the ceiling evicts the oldest one.
  for (int k = 0; k < 6; k++) {
    static const size_t sizes[6] = {
        6 * MB, 256 * KB, 2 * MB, 6 * MB, 512 * KB, 3 * MB};
    void *r[6];
    OMCacheTrim();
    for (int i = k; i < 6; i++)
      r[i] = OMCacheAlloc(sizes[i]);
    for (int i = k; i < 6; i++)
      OMCacheFree(r[i]); // r[k] is the oldest available buffer.
    check();
    pthread_mutex_lock(&cacheMutex);
    assert(findOldestAvailable()->buffer == r[k]);
    pthread_mutex_unlock(&cacheMutex);
    // A ceiling just below the current bytes: freeing one more 192 KB buffer
    // evicts exactly one buffer, the oldest (all are >= 256 KB).
    cache.maxAvailableBytes = cache.availableBytes - 1;
    void *t = OMCacheAlloc(128 * KB + 4096);
    OMCacheFree(t);
    check();
    pthread_mutex_lock(&cacheMutex);
    assert(findBufferDesc(r[k]) == NULL);
    for (int i = k + 1; i < 6; i++)
      assert(findBufferDesc(r[i]) != NULL);
    pthread_mutex_unlock(&cacheMutex);
    cache.maxAvailableBytes = savedMax;
  }
  printf("eviction order across bins: ok\n");
  OMCacheTrim();

  // 8. Threads, with forks in the middle (not on z/OS: see README.md).
  pthread_t th[8];
  for (int i = 0; i < 8; i++)
    pthread_create(&th[i], NULL, worker, (void *)(uintptr_t)(i + 1));
#if !defined(__MVS__)
  for (int f = 0; f < 20; f++) {
    pid_t pid = fork();
    if (pid == 0) {
      void *x = OMCacheAlloc(2 * MB);
      OMCacheFree(x);
      check();
      _exit(0);
    }
    int status;
    waitpid(pid, &status, 0);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  }
#endif
  for (int i = 0; i < 8; i++)
    pthread_join(th[i], NULL);
  check();
  printf("threads and fork: registered %zu, peak registered %zu KB\n",
      cache.numRegistered, cache.statPeakRegisteredBytes / KB);
  for (int i = 0; i < cache.numBins; i++)
    assert(cache.bins[i].numInUse == 0);

  printf("test_cache: ALL OK\n");
  return 0;
}
