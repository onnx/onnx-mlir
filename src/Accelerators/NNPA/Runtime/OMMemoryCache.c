/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- OMMemoryCache.c ---------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Reuse cache for the per-inference temporary buffers of a compiled model.
//
// How buffers are managed, from request to release (terms in "Vocabulary"):
// 1. OMCacheAlloc(size): a size from 128 KB to OM_MEMORY_CACHE_LARGE (default
//    256 MB) is served from the cache: it is rounded up to a bin (step 2).
//    Other sizes, and all sizes when the cache is off, get an uncached buffer
//    (plain malloc below 1 MB), freed with free(). In both cases, on Linux, a
//    buffer of 1 MB or more is 1 MB-aligned and advised for huge pages.
// 2. Hit or miss: the bin's most recently freed available buffer is handed
//    out (hit); without one, a new buffer is allocated outside the mutex
//    (miss). Epoch policy: a stale buffer is not handed out; it and the older
//    ones of its bin are returned to the system, and the request misses.
// 3. OMCacheFree / OMCacheFreeSized: a buffer that was not served from the
//    cache goes to free(); OMCacheFreeSized, given the size, does so without
//    taking the mutex. A cache buffer goes back on its bin's available list,
//    unless its bin is in eager release (epoch policy): then it is returned to
//    the system at once.
// 4. Ceiling, checked when a cache buffer is freed: if the available bytes
//    then exceed OM_MEMORY_CACHE_MAX (default 1 GB), the oldest available
//    buffers of the whole cache are returned until they are back under it.
// 5. End of an inference (OMCacheEpoch(id), epoch policy), unless a later one
//    has started meanwhile: the available buffers of large bins are returned,
//    and each large bin used in this inference decides whether to be in eager
//    release during the next ones.
// 6. OMCacheTrim, a failed allocation (retried once if that returned any) and
//    the unload of the model .so return all available buffers.
// Buffers in use are never touched. Without transparent huge pages the
// default policy is none: steps 2, 3 and 5 keep every buffer, and only the
// ceiling, trim and unload return memory.
//
// Design decisions:
// - Why: glibc often returns freed memory to the kernel, so every inference
//   refaults and rezeroes it (and pays a TLB purge per page on s390x).
// - Scope: buffers the model frees itself (selected by the pass
//   ReplaceMallocByHugePageMalloc) and zdnnx scratch. Outputs are not cached.
// - Each buffer is its own posix_memalign, so it can be returned to the system
//   on its own. It must be given back with OMCacheFree or OMCacheFreeSized,
//   never with plain free() (see "orphan descriptor").
// - Bins: 64 KB steps to 1 MB, then 1 MB multiples, <= 4 per doubling; no
//   splitting. < 128 KB (glibc reuses those well) and > OM_MEMORY_CACHE_LARGE
//   (default 256 MB) are not cached.
// - Large buffers that went cold are not reused (epoch policy, the default):
//   writing into a cold buffer first fetches its stale contents from memory,
//   which is slower than a fresh, kernel-zeroed mapping, while a large buffer
//   reused within an inference is worth keeping. Default only with
//   transparent huge pages: with 4 KB pages, releasing is the expensive part.
//   OM_MEMORY_CACHE_POLICY=none keeps all buffers.
// - Freed buffers are kept available (peak use per bin at fixed shapes); a
//   ceiling on available bytes evicts the oldest first. When an allocation
//   fails, all available buffers are returned to the system and, if there
//   were any, the allocation is retried once.
// - One mutex. Buffers are obtained from and returned to the system outside
//   of it, since releasing memory can cost a TLB purge per page and would
//   stall every thread waiting on the mutex; only the release of an orphan
//   descriptor calls free while holding it.
//
// Environment variables and the build option: see OMMemoryCache.h.
//
// Debugging and testing:
// - Reused buffers are NOT zeroed: a hit returns the data of an earlier
//   inference, where a fresh large malloc used to return zero pages. Code that
//   reads a buffer before writing it was hidden before and now gives wrong or
//   run-to-run different results, typically from the second inference in a
//   process on. To check: compare outputs with OM_MEMORY_CACHE_MAX=0 (cache
//   off), and run several inferences in one process. To expose such reads,
//   poison reused buffers (e.g. memset to 0xFF on a hit).
// - Sanitizers and Valgrind: a cached buffer is never really freed, so a
//   use-after-free of a temporary is invisible to ASan and Valgrind. Run
//   memory-checking jobs with OM_MEMORY_CACHE_MAX=0.
// - High memory use: on top of the buffers in use, each model .so may hold up
//   to OM_MEMORY_CACHE_MAX of available buffers (default 1 GB), and a buffer
//   is rounded up to its bin size: up to 50% more just above 128 KB, 25% from
//   256 KB to 1 MB, up to 100% just above 1 MB (the next bin is 2 MB), and at
//   most 25% from 4 MB up. OM_MEMORY_CACHE_STATS=1 prints the largest
//   available and registered bytes held. Remedies: lower OM_MEMORY_CACHE_MAX
//   (e.g. 256M), lower OM_MEMORY_CACHE_LARGE (e.g. 16M) to stop caching big
//   buffers, call OMCacheTrim() from a host that keeps idle models loaded, or
//   set OM_MEMORY_CACHE_MAX=0 to turn the cache off.
//
//===----------------------------------------------------------------------===//

// Include pthreads (need special treatment on Zos).
#ifdef __MVS__
#define _OPEN_THREADS
#endif
#include <pthread.h>

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "OMMemoryCache.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KB ((size_t)1024)
#define MB (KB * KB)
#define GB (KB * MB)

#define CACHE_MIN_SIZE (128 * KB)
#define CACHE_SMALL_BIN_STEP (64 * KB)
// Largest cached request by default; see "Bins" in the design notes above.
#define CACHE_DEFAULT_LARGE (256 * MB)
// Epoch policy: smallest buffer size of a large bin (see the vocabulary).
#define CACHE_EPOCH_MIN_SIZE (16 * MB)
// OM_MEMORY_CACHE_POLICY values.
#define CACHE_POLICY_NONE 0  // Keep every buffer.
#define CACHE_POLICY_EPOCH 1 // Default: large buffers not reused across epochs.
// Per model .so. If memory use is too high, see "Debugging and testing" above.
#define CACHE_DEFAULT_MAX (1 * GB)
#define CACHE_PAGE_4K (4 * KB)
#define CACHE_HUGE_PAGE (1 * MB)
// Huge pages (transparent huge pages, Linux only): buffers of CACHE_HUGE_PAGE
// or more are 1 MB-aligned and advised for huge pages, as in OMHugePageMalloc
// and zdnnx. Elsewhere (z/OS, macOS) 1 MB alignment would only add padding,
// so 4 KB is used, as zdnnx does there.
#if defined(__linux__) && defined(MADV_HUGEPAGE)
#define CACHE_USE_HUGE_PAGES 1
#else
#define CACHE_USE_HUGE_PAGES 0
#endif
#define CACHE_MAX_BINS 256
// Registry buckets (fixed, never resized): 1024, i.e. 8 KB of bucket heads.
#define CACHE_REGISTRY_LOG2 10
#define CACHE_REGISTRY_BUCKETS (1 << CACHE_REGISTRY_LOG2)
// Upper bound on OM_MEMORY_CACHE_LARGE: 1 TB, or SIZE_MAX / 4 if smaller, so
// that building the table of bins cannot overflow size_t.
#define CACHE_MAX_LARGE                                                        \
  ((unsigned long long)SIZE_MAX / 4 < (1ull << 40) ? (size_t)(SIZE_MAX / 4)    \
                                                   : (size_t)(1ull << 40))

//===----------------------------------------------------------------------===//
// Vocabulary.
//===----------------------------------------------------------------------===//
//
// - buffer: the memory handed to the caller (void *).
// - buffer descriptor (OMCacheBufferDesc, "desc"): the cache's record of one
//   buffer; a separate small malloc. Every buffer owned by the cache has one,
//   whether in use or available. It is created with its buffer on a miss (so
//   the buffer starts in use), stays with that buffer through every free and
//   reuse, and is freed only when the buffer is detached and returned to the
//   system.
// - bin (OMCacheBin): everything about one buffer size: its available list,
//   its count of buffers in use, and its statistics.
// - in use / available: the two states of a buffer owned by the cache. In use:
//   handed out, not yet given back with OMCacheFree. Available: kept by the
//   cache, ready to be handed out again.
// - available list: a bin's available buffers, most recently freed first (the
//   reuse order); so its tail is the bin's oldest available buffer.
// - age stamp (freedAt): when an available buffer was freed, from a counter
//   incremented each time a buffer is put on an available list. The oldest
//   available buffer of the whole cache is the bin tail with the smallest stamp
//   (the eviction order).
// - epoch: one inference, delimited by OMCacheEpoch calls that the compiler
//   wraps around the model's body. A kept buffer records the epoch of its
//   free. Compiled code allocates only inside an epoch; outside any (before
//   the first start), the epoch policy's checks are skipped (cache.epoch is
//   0). Epochs are counted per model .so, not per session.
// - large bin (epoch policy): a bin of CACHE_EPOCH_MIN_SIZE (16 MB) or more.
//   The epoch policy acts only on large bins.
// - stale buffer (epoch policy): an available buffer of a large bin freed in
//   an earlier epoch; its memory has likely left the processor caches.
//   Normally there are none, since the end of each epoch returns those
//   buffers; they remain only when inferences overlap (an end releases
//   nothing if a later epoch has started meanwhile). A request that finds a
//   stale buffer returns it and the older ones of its bin to the system and
//   gets a fresh buffer.
// - eager release (epoch policy): a large bin whose buffers were not reused
//   within the last ended epoch that used it (no request after a free of the
//   bin) returns each buffer to the system at its free, instead of keeping it
//   until the end of the epoch. This lowers the peak memory of an inference;
//   the release work is the same, only earlier. Decided again at the end of
//   each epoch that uses the bin (unless a later epoch has started
//   meanwhile), so a bin switches back when its pattern changes.
// - registry: hash maps a buffer to its descriptor.
// - orphan descriptor: a registry entry whose buffer was released behind the
//   cache's back, with plain free() instead of OMCacheFree: a caller bug,
//   never expected (the pass and zdnnx always pair the cache's alloc and free
//   calls). Found when posix_memalign returns that address again for a new
//   cache buffer; asserted in debug builds, repaired (and warned about once)
//   in release builds. A double free is handled the same way: ignored, and
//   warned about once.
// - detach: take a buffer out of every cache structure, under the mutex. An
//   available buffer leaves its available list and the registry; a buffer in
//   eager release, detached at its free, was never on the list and leaves
//   only the registry.
// - return to the system: free() detached buffers and their descriptors,
//   after the mutex is dropped.
// - evict / trim: reasons to detach. Evict: over the ceiling on available
//   bytes, oldest first. Trim: all available buffers (OMCacheTrim, failed
//   allocation, unload). The epoch policy detaches too: stale buffers at a
//   request, the large available buffers at the end of an epoch, and a buffer
//   of a bin in eager release at its free.
//

// Descriptor of one buffer owned by the cache, in use or available. An in-use
// buffer's descriptor is held only by the registry (the caller holds just the
// buffer); an available buffer's descriptor is also on its bin's available
// list.
typedef struct OMCacheBufferDesc {
  void *buffer;        // Memory handed out to the caller.
  size_t bufferSize;   // Size of the bin, not of the request.
  int binIndex;        // Index of the bin in OMCache.bins.
  int inUse;           // 1 between OMCacheTryAlloc and OMCacheFree, else 0.
  uint64_t freedEpoch; // cache.epoch at its last free (epoch policy).
  // Available list of the bin, most recently freed first. Once detached,
  // availableNext chains the detached list instead.
  struct OMCacheBufferDesc *availablePrev, *availableNext;
  // Age stamp: value of cache.freeClock when the buffer was last freed.
  uint64_t freedAt;
  // Next descriptor in the same registry bucket (hash collision chain).
  struct OMCacheBufferDesc *bucketNext;
  size_t requestSize; // Size asked for by the current user (statistics).
} OMCacheBufferDesc;

// One bin: its available buffers, the count of buffers in use, and counters
// for the statistics (stat*, only printed). statPeakInUse is what the bin
// would need to keep to hit on every request at fixed shapes.
typedef struct {
  size_t bufferSize;                // Every buffer of the bin has this size.
  OMCacheBufferDesc *availableHead; // Available list, most recent first.
  OMCacheBufferDesc *availableTail; // Oldest available buffer of the bin.
  uint64_t numInUse;                // Requests of this bin not yet freed.
  // Epoch policy, large bins: frees and requests-after-a-free in the current
  // epoch (reset at each epoch start), and the eager-release decision taken
  // at the end of the last epoch that used the bin.
  uint64_t freesThisEpoch, reusesThisEpoch;
  int eagerRelease;
  // Statistics. statDetached: by eviction, trim, unload or epoch-policy
  // release. statStaleReleases: stale buffers returned (at a request or at
  // the end of an epoch). statEagerReleases: buffers returned at their free.
  uint64_t statNumAvailable, statPeakInUse;
  uint64_t statHits, statMisses, statDetached, statStaleReleases;
  uint64_t statEagerReleases;
} OMCacheBin;

// The whole cache: one static instance per model .so, guarded by cacheMutex
// (except the configuration and bin sizes, which are fixed after init).
typedef struct {
  // Configuration, from the environment.
  int enabled, printStats;
  size_t maxAvailableBytes; // Ceiling on the bytes of available buffers.
  size_t maxCachedSize; // Largest cached request (buffer size of the last bin).
  // OM_MEMORY_CACHE_POLICY (CACHE_POLICY_*), and whether transparent huge
  // pages are available (which decides the default policy).
  int policy, thpAvailable;
  // Epoch policy: current epoch, from OMCacheEpoch (0 until the first start).
  uint64_t epoch;
  // Bins, by ascending buffer size.
  OMCacheBin bins[CACHE_MAX_BINS];
  int numBins;
  // Registry: hash table of all descriptors keyed by buffer, chained through
  // OMCacheBufferDesc.bucketNext. It tells OMCacheFree whether a buffer
  // belongs to the cache, and its descriptor.
  OMCacheBufferDesc *buckets[CACHE_REGISTRY_BUCKETS];
  size_t numRegistered; // Registered buffers, in use or available.
  // Counter for the age stamps, incremented at each OMCacheFree of a buffer.
  uint64_t freeClock;
  // Bytes of available buffers, checked against maxAvailableBytes.
  size_t availableBytes;
  // Statistics (stat*, only printed). statRegisteredBytes: in all registered
  // buffers. statOrphanDescs: orphan descriptors found (see the vocabulary).
  // statAllocFailureTrims: trims after a failed allocation.
  size_t statRegisteredBytes, statPeakAvailableBytes, statPeakRegisteredBytes;
  uint64_t statEvictions, statTrims, statAllocFailureTrims;
  uint64_t statOrphanDescs, statDoubleFrees, statFailedAllocs;
  uint64_t statEpochsStarted, statEpochsEnded, statEpochEndReleases;
  // OMCacheFreeSized calls that freed without the mutex (size without a bin).
  // Updated atomically, and only when statistics are printed.
  uint64_t statFastFrees;
  // Registry statistics: most buffers registered at once, longest chain seen.
  size_t statPeakRegistered, statMaxBucketChain;
  // Bytes of buffers in use at the same time, counted as buffer sizes and as
  // requested sizes, and their peaks. statPeakInUseBytes vs
  // statPeakRegisteredBytes: memory lost because bins do not share buffers
  // across sizes. statPeakInUseRequestBytes vs statPeakInUseBytes: memory
  // lost to rounding requests up to the bin size.
  size_t statInUseBytes, statPeakInUseBytes;
  size_t statInUseRequestBytes, statPeakInUseRequestBytes;
  // Set once a misuse (orphan descriptor or double free) has been reported.
  int warnedMisuse;
} OMCache;

// The cache state and its lock; initCache runs once through cacheOnce.
static OMCache cache;
static pthread_mutex_t cacheMutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t cacheOnce = PTHREAD_ONCE_INIT;

//===----------------------------------------------------------------------===//
// Initialization.
//===----------------------------------------------------------------------===//

// Parse "<number>[K|M|G]"; return defaultSize if the variable is unset,
// malformed, negative, or too large for size_t (including after the K/M/G
// scaling).
static size_t getEnvSize(const char *name, size_t defaultSize) {
  const char *s = getenv(name);
  if (!s)
    return defaultSize;
  while (*s == '\t' || *s == ' ')
    s++;
  // strtoull accepts a sign and negates the value: reject both.
  if (*s < '0' || *s > '9')
    return defaultSize;
  char *end = NULL;
  errno = 0;
  unsigned long long v = strtoull(s, &end, 10);
  if (errno == ERANGE || end == s)
    return defaultSize;
  unsigned long long scale = 1;
  switch (*end) {
  case 'k':
  case 'K':
    scale = KB;
    end++;
    break;
  case 'm':
  case 'M':
    scale = MB;
    end++;
    break;
  case 'g':
  case 'G':
    scale = GB;
    end++;
    break;
  default:
    break;
  }
  if (*end == 'b' || *end == 'B')
    end++;
  if (*end != '\0')
    return defaultSize;
  if (v > (unsigned long long)SIZE_MAX / scale)
    return defaultSize;
  return (size_t)(v * scale);
}

#ifdef __MVS__
// No fork handlers on z/OS (where the cache is off by default). The z/OS
// pthread_atfork documentation says that locking in the prepare handler and
// unlocking in the child handler is "not practical on the z/OS platform", and
// that a child of a multithreaded process may only call async-signal-safe
// functions before exec (POSIX requires the same everywhere), so such a child
// must not use the cache.
// TODO(z/OS): confirm that no z/OS host forks a multithreaded process and then
// runs a model in the child without exec.
static int registerForkHandlers(void) { return 1; }
#elif defined(__APPLE__)
// No fork handlers on macOS (used only for development builds of the NNPA
// runtime): macOS does not remove pthread_atfork handlers when the library
// that registered them is closed, so a fork() after the model .so is unloaded
// would jump into unmapped code. Without handlers, a fork while another thread
// holds the cache mutex leaves the child unable to use the cache, as on z/OS.
static int registerForkHandlers(void) { return 1; }
#else
// Fork handlers: hold the mutex across fork() so that the child gets the cache
// in a consistent state, then release it in both parent and child.
static void lockForFork(void) { pthread_mutex_lock(&cacheMutex); }
static void unlockForFork(void) { pthread_mutex_unlock(&cacheMutex); }

// Return 0 if the handlers could not be registered (ENOMEM). glibc removes
// handlers registered from a dlopen'ed library when the library is closed
// (pthread_atfork passes the library's __dso_handle), so a fork after
// dlclose is safe.
static int registerForkHandlers(void) {
  return pthread_atfork(lockForFork, unlockForFork, unlockForFork) == 0;
}
#endif

// Add one more bin, for buffers of the given size.
static void initNextBinSize(size_t size) {
  if (cache.numBins < CACHE_MAX_BINS)
    cache.bins[cache.numBins++].bufferSize = size;
}

// True if transparent huge pages can back the cache's large buffers: Linux,
// and /sys/kernel/mm/transparent_hugepage/enabled set to always or madvise.
// The file is missing when the kernel does not support THP (e.g. no EDAT-1 on
// s390x) or in sandboxes that hide /sys/kernel/mm.
static int transparentHugePagesAvailable(void) {
#if CACHE_USE_HUGE_PAGES
  FILE *f = fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r");
  if (!f)
    return 0;
  char line[128];
  int ok = fgets(line, sizeof(line), f) != NULL;
  fclose(f);
  return ok && (strstr(line, "[always]") || strstr(line, "[madvise]"));
#else
  return 0;
#endif
}

// Read the environment, build the table of bins, and register the fork
// handlers. Runs once, at the first call into the cache.
static void initCache(void) {
  memset(&cache, 0, sizeof(cache));
  // Get configs from environment to determine if the cache is enabled.
  cache.maxAvailableBytes =
      getEnvSize("OM_MEMORY_CACHE_MAX", CACHE_DEFAULT_MAX);
  cache.maxCachedSize =
      getEnvSize("OM_MEMORY_CACHE_LARGE", CACHE_DEFAULT_LARGE);
  if (cache.maxCachedSize > CACHE_MAX_LARGE)
    cache.maxCachedSize = CACHE_MAX_LARGE;
  // Transparent huge pages: needed even when the cache is off, since its
  // uncached buffers are advised for huge pages too (allocAlignMem).
  cache.thpAvailable = transparentHugePagesAvailable();
  // Off: every call then behaves as before the cache; nothing else to read.
  cache.enabled =
      cache.maxAvailableBytes > 0 && cache.maxCachedSize >= CACHE_MIN_SIZE;
  if (!cache.enabled)
    return;
  // Determine the cache policy: an explicit OM_MEMORY_CACHE_POLICY wins; by
  // default, epoch with transparent huge pages and none without.
  cache.policy = cache.thpAvailable ? CACHE_POLICY_EPOCH : CACHE_POLICY_NONE;
  const char *policy = getenv("OM_MEMORY_CACHE_POLICY");
  if (policy && strcmp(policy, "epoch") == 0)
    cache.policy = CACHE_POLICY_EPOCH;
  else if (policy && strcmp(policy, "none") == 0)
    cache.policy = CACHE_POLICY_NONE;
  // Determine if we want to print statistics when closing the cache.
  const char *stats = getenv("OM_MEMORY_CACHE_STATS");
  cache.printStats = stats && *stats && *stats != '0';
  // Keep the mutex usable in a child process forked while it was held. Done
  // first: if it fails, the cache is disabled before any buffer exists, so a
  // forked child cannot deadlock.
  if (!registerForkHandlers()) {
    cache.enabled = 0;
    if (cache.printStats)
      fprintf(stderr, "==OM-MEMORY-CACHE== disabled: pthread_atfork failed\n");
    return;
  }
  // Set the bin sizes: 64 KB steps up to 1 MB; then multiples of 1 MB, with a
  // step of a quarter of the largest power of two below the size.
  size_t s = CACHE_MIN_SIZE;
  for (; s <= CACHE_HUGE_PAGE; s += CACHE_SMALL_BIN_STEP)
    initNextBinSize(s);
  s = CACHE_HUGE_PAGE;
  while (s < cache.maxCachedSize && cache.numBins < CACHE_MAX_BINS) {
    size_t pow2 = CACHE_HUGE_PAGE;
    while (pow2 * 2 <= s)
      pow2 *= 2;
    size_t step = pow2 / 4 < CACHE_HUGE_PAGE ? CACHE_HUGE_PAGE : pow2 / 4;
    s += step;
    initNextBinSize(s);
  }
  // Requests above the last bin are not cached.
  if (cache.bins[cache.numBins - 1].bufferSize < cache.maxCachedSize)
    cache.maxCachedSize = cache.bins[cache.numBins - 1].bufferSize;
}

static inline void ensureInit(void) { pthread_once(&cacheOnce, initCache); }

// Index of the smallest bin whose buffers hold size bytes; hasBin(size) must
// be true.
static int binIndexFor(size_t size) {
  int lo = 0, hi = cache.numBins - 1;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (cache.bins[mid].bufferSize >= size)
      hi = mid;
    else
      lo = mid + 1;
  }
  return lo;
}

// True if requests of this size are served from the cache. When the cache is
// disabled there are no bins, so this is false for every size.
static inline int hasBin(size_t size) {
  return cache.enabled && size >= CACHE_MIN_SIZE && size <= cache.maxCachedSize;
}

//===----------------------------------------------------------------------===//
// Registry (cache mutex held).
//===----------------------------------------------------------------------===//
//
// Maps each buffer owned by the cache to its descriptor; used to tell whether
// a buffer passed to OMCacheFree belongs to the cache. Hash policy:
// - Separate chaining through the descriptors themselves (intrusive): each
//   bucket heads a singly linked list of OMCacheBufferDesc via bucketNext.
//   Keys are unique (one descriptor per live buffer address).
// - Fixed table of CACHE_REGISTRY_BUCKETS (1024) buckets, never resized. The
//   cache holds a few hundred buffers, so chains stay at about one entry; more
//   buffers only lengthen chains, and insertion can never fail.
// - Hash: multiplicative (Fibonacci) hashing of the address with the low 12
//   bits dropped (buffers are >= 4 KB aligned, so they carry no information);
//   the bucket index is the top CACHE_REGISTRY_LOG2 bits of the 64-bit product.
// - Insert at the chain head; delete by unlinking from the chain. No
//   tombstones and no rehashing.
// - One lookup per OMCacheFree, often unsuccessful (pointers from plain
//   malloc): it walks one short chain. statMaxBucketChain reports the longest.
//

// Bucket index of a buffer address.
static inline size_t hashPtr(const void *buffer) {
  uint64_t h = ((uint64_t)(uintptr_t)buffer >> 12) * 0x9E3779B97F4A7C15ull;
  return (size_t)(h >> (64 - CACHE_REGISTRY_LOG2));
}

// Return the descriptor of buffer, or NULL if the cache does not own buffer.
static OMCacheBufferDesc *findBufferDesc(const void *buffer) {
  for (OMCacheBufferDesc *desc = cache.buckets[hashPtr(buffer)]; desc;
       desc = desc->bucketNext)
    if (desc->buffer == buffer)
      return desc;
  return NULL;
}

// Map desc->buffer to desc; desc->buffer must not be mapped yet.
static void mapBufferToDesc(OMCacheBufferDesc *desc) {
  OMCacheBufferDesc **head = &cache.buckets[hashPtr(desc->buffer)];
  desc->bucketNext = *head;
  *head = desc;
  cache.numRegistered++;
  if (cache.printStats) {
    if (cache.numRegistered > cache.statPeakRegistered)
      cache.statPeakRegistered = cache.numRegistered;
    size_t chain = 0;
    for (OMCacheBufferDesc *d = desc; d; d = d->bucketNext)
      chain++;
    if (chain > cache.statMaxBucketChain)
      cache.statMaxBucketChain = chain;
  }
}

// Remove the mapping of desc->buffer, which must be mapped.
static void removeBufferMapping(OMCacheBufferDesc *desc) {
  OMCacheBufferDesc **link = &cache.buckets[hashPtr(desc->buffer)];
  while (*link != desc)
    link = &(*link)->bucketNext;
  *link = desc->bucketNext;
  desc->bucketNext = NULL;
  cache.numRegistered--;
}

//===----------------------------------------------------------------------===//
// Available list, detach, and return to the system (cache mutex held, except
// returnDetachedToSystem).
//===----------------------------------------------------------------------===//
//
// removeFromAvailableList is a building block: a buffer changes state from
// available to in use. Detach means the buffer leaves the cache for good.
//
// |                       | removeFromAvailableList | detachFromCache         |
// |-----------------------|-------------------------|-------------------------|
// | bin's available list  | removed                 | removed.                |
// |-----------------------|-------------------------|-------------------------|
// | availableBytes        | decreased               | decreased               |
// |-----------------------|-------------------------|-------------------------|
// | registry              | stays mapped            | mapping removed         |
// |-----------------------|-------------------------|-------------------------|
// | afterwards, buffer is | still owned by the      | unknown to the cache,   |
// |                       | cache, now in use       | queued on detached, to  |
// |                       |                         | be returned to system   |
// |-----------------------|-------------------------|-------------------------|
// | used by               | a hit in                | evict; trim (from       |
// |                       | OMCacheTryAlloc; detach | OMCacheTrim, an alloc   |
// |                       |                         | failure, or unload)     |
// |-----------------------|-------------------------|-------------------------|

// Add a buffer that was just freed to its bin's available list (at the head,
// so the most recently used buffer is reused first). Also stamp its age and
// count its bytes as available.
static void addToAvailableList(OMCacheBufferDesc *desc) {
  OMCacheBin *bin = &cache.bins[desc->binIndex];
  // Push desc at the head of the bin's available list (the next one reused).
  desc->availablePrev = NULL;
  desc->availableNext = bin->availableHead;
  if (bin->availableHead)
    bin->availableHead->availablePrev = desc;
  bin->availableHead = desc;
  if (!bin->availableTail)
    bin->availableTail = desc;
  bin->statNumAvailable++;
  // Stamp its age: the youngest available buffer of the cache.
  desc->freedAt = ++cache.freeClock;
  // Count the bytes of available buffers; OMCacheFree enforces the ceiling.
  cache.availableBytes += desc->bufferSize;
  // High-water mark, for the statistics only.
  if (cache.availableBytes > cache.statPeakAvailableBytes)
    cache.statPeakAvailableBytes = cache.availableBytes;
}

// Remove an available buffer from its bin's available list and from the
// available bytes, either to hand it out again or to detach it. The registry
// is not changed.
static void removeFromAvailableList(OMCacheBufferDesc *desc) {
  OMCacheBin *bin = &cache.bins[desc->binIndex];
  // Unlink desc from the bin's available list: the head for a hit, the tail
  // for an eviction, anywhere for an orphan descriptor.
  if (desc->availablePrev)
    desc->availablePrev->availableNext = desc->availableNext;
  else
    bin->availableHead = desc->availableNext;
  if (desc->availableNext)
    desc->availableNext->availablePrev = desc->availablePrev;
  else
    bin->availableTail = desc->availablePrev;
  desc->availablePrev = desc->availableNext = NULL;
  bin->statNumAvailable--;
  // The buffer no longer counts as available.
  cache.availableBytes -= desc->bufferSize;
}

// Detach an available buffer from the cache: remove it from the available
// list and the registry, and chain it into *detached (through availableNext)
// so that it is returned to the system after the mutex is dropped.
static void detachFromCache(
    OMCacheBufferDesc *desc, OMCacheBufferDesc **detached) {
  removeFromAvailableList(desc);
  removeBufferMapping(desc);
  cache.statRegisteredBytes -= desc->bufferSize;
  cache.bins[desc->binIndex].statDetached++;
  desc->availableNext = *detached;
  *detached = desc;
}

// Return detached buffers and their descriptors to the system with free().
// Called without the mutex, and it must stay so (see "One mutex"): freeing a
// large buffer munmaps it, and a trim releases many buffers at once.
static void returnDetachedToSystem(OMCacheBufferDesc *detached) {
  while (detached) {
    OMCacheBufferDesc *nextDesc = detached->availableNext;
    free(detached->buffer);
    free(detached);
    detached = nextDesc;
  }
}

// Detach all available buffers of one bin into *detached (stale buffers);
// return how many were detached.
static inline uint64_t detachAllAvailableInBin(
    OMCacheBin *bin, OMCacheBufferDesc **detached) {
  uint64_t n = 0;
  for (; bin->availableHead; n++)
    detachFromCache(bin->availableHead, detached);
  return n;
}

// Detach every available buffer into *detached; return how many were detached.
static size_t detachAllAvailableInCache(OMCacheBufferDesc **detached) {
  size_t n = 0;
  for (int i = 0; i < cache.numBins; i++)
    n += detachAllAvailableInBin(&cache.bins[i], detached);
  return n;
}

// Return the oldest available buffer of the cache (see "age stamp"), or NULL
// if there is none. Scans the bin tails; only used for eviction.
static OMCacheBufferDesc *findOldestAvailable(void) {
  OMCacheBufferDesc *oldest = NULL;
  for (int i = 0; i < cache.numBins; i++) {
    OMCacheBufferDesc *tail = cache.bins[i].availableTail;
    if (tail && (!oldest || tail->freedAt < oldest->freedAt))
      oldest = tail;
  }
  return oldest;
}

// Trim after an allocation failed: return every available buffer to the
// system, so that the retry can use that memory. Return 1 if any buffer was
// returned. Takes the mutex.
static int trimOnAllocFailure(void) {
  if (!cache.enabled)
    return 0;
  pthread_mutex_lock(&cacheMutex);
  OMCacheBufferDesc *detached = NULL;
  size_t n = detachAllAvailableInCache(&detached);
  if (n)
    cache.statAllocFailureTrims++;
  pthread_mutex_unlock(&cacheMutex);
  // The memory is returned to the system outside the mutex.
  returnDetachedToSystem(detached);
  return n > 0;
}

// Statistics only: count a buffer handed out for a request of requestSize
// bytes, or given back, in the bytes in use at the same time.
static void statCountHandOut(OMCacheBufferDesc *desc, size_t requestSize) {
  desc->requestSize = requestSize;
  cache.statInUseBytes += desc->bufferSize;
  cache.statInUseRequestBytes += requestSize;
  if (cache.statInUseBytes > cache.statPeakInUseBytes)
    cache.statPeakInUseBytes = cache.statInUseBytes;
  if (cache.statInUseRequestBytes > cache.statPeakInUseRequestBytes)
    cache.statPeakInUseRequestBytes = cache.statInUseRequestBytes;
}

static void statCountGiveBack(OMCacheBufferDesc *desc) {
  cache.statInUseBytes -= desc->bufferSize;
  cache.statInUseRequestBytes -= desc->requestSize;
}

//===----------------------------------------------------------------------===//
// Buffer creation.
//===----------------------------------------------------------------------===//

// Get size bytes aligned to align from the system; on Linux, memory of 1 MB or
// more is also advised for huge pages. Returns NULL on failure. The memory is
// released with free().
static void *allocAlignMem(size_t size, size_t align) {
  void *alignedPtr = NULL;
  if (posix_memalign(&alignedPtr, align, size) != 0)
    return NULL;
#if CACHE_USE_HUGE_PAGES
  // Advise once, when the memory is obtained, if the kernel has transparent
  // huge pages enabled; it is ok if the advice fails.
  if (size >= CACHE_HUGE_PAGE && cache.thpAvailable)
    madvise(alignedPtr, size, MADV_HUGEPAGE);
#endif
  return alignedPtr;
}

// Memory of a new cache buffer (size is a bin size, at least 128 KB): 4 KB
// aligned, since zdnnx receives cache buffers through OMCacheTryAlloc and zDNN
// needs 4 KB; on Linux, 1 MB aligned (and advised for huge pages) from 1 MB up.
static void *allocCachedBuffer(size_t size) {
  size_t align = CACHE_USE_HUGE_PAGES && size >= CACHE_HUGE_PAGE
                     ? CACHE_HUGE_PAGE
                     : CACHE_PAGE_4K;
  return allocAlignMem(size, align);
}

// An uncached buffer (size without a bin, or cache off), for compiled code
// only, which aligns its own buffers: on Linux, from 1 MB up, 1 MB-aligned
// memory advised for huge pages, as OMHugePageMalloc does; otherwise, or if
// that fails, plain malloc. Freed with free().
static void *allocUncachedBuffer(size_t size) {
  void *buffer = NULL;
  if (CACHE_USE_HUGE_PAGES && size >= CACHE_HUGE_PAGE)
    buffer = allocAlignMem(size, CACHE_HUGE_PAGE);
  return buffer ? buffer : malloc(size); // Last resort, may be unaligned.
}

// The only allocation retry: get size bytes with alloc (allocCachedBuffer or
// allocUncachedBuffer); if that fails, return every available buffer to the
// system (trimOnAllocFailure) and, if there was any, try once more. Returns
// NULL if memory is still short. size is never 0 (OMCacheAlloc returns NULL
// for it), so a NULL from alloc is always a failure. Called without the mutex.
static void *allocWithTrimRetry(void *(*alloc)(size_t), size_t size) {
  void *buffer = alloc(size);
  if (!buffer && trimOnAllocFailure())
    buffer = alloc(size);
  return buffer;
}

// Report a misuse of the interface once per model .so: it is repaired or
// ignored, but it means a caller bug. Called with the mutex held.
static void warnMisuseOnce(const char *what) {
  if (cache.warnedMisuse)
    return;
  cache.warnedMisuse = 1;
  fprintf(stderr,
      "==OM-MEMORY-CACHE== warning: %s; ignored (reported once). Compare "
      "with OM_MEMORY_CACHE_MAX=0.\n",
      what);
}

// Fill desc for a new in-use buffer of bin binIndex, handed out for a request
// of requestSize bytes, and map buffer to it. Called with the mutex held;
// releases it and returns buffer.
static void *mapNewBufferAndUnlock(
    OMCacheBufferDesc *desc, void *buffer, int binIndex, size_t requestSize) {
  size_t size = cache.bins[binIndex].bufferSize;
  // Never expected: this address is still mapped, so it has an orphan
  // descriptor (see the vocabulary). Release builds forget the orphan and
  // repair the counts.
  OMCacheBufferDesc *orphanDesc = findBufferDesc(buffer);
  assert(!orphanDesc && "cache buffer released with plain free()");
  if (orphanDesc) {
    removeBufferMapping(orphanDesc);
    if (orphanDesc->inUse) {
      cache.bins[orphanDesc->binIndex].numInUse--;
      statCountGiveBack(orphanDesc);
    } else {
      // Even less expected: freed with OMCacheFree, then again with free().
      removeFromAvailableList(orphanDesc);
    }
    cache.statRegisteredBytes -= orphanDesc->bufferSize;
    cache.statOrphanDescs++;
    warnMisuseOnce("cache buffer released with plain free()");
    free(orphanDesc);
  }
  desc->buffer = buffer;
  desc->bufferSize = size;
  desc->binIndex = binIndex;
  desc->inUse = 1;
  desc->availablePrev = desc->availableNext = NULL;
  desc->freedAt = 0;
  desc->freedEpoch = 0;
  statCountHandOut(desc, requestSize);
  mapBufferToDesc(desc);
  cache.statRegisteredBytes += size;
  // High-water mark, for the statistics only.
  if (cache.statRegisteredBytes > cache.statPeakRegisteredBytes)
    cache.statPeakRegisteredBytes = cache.statRegisteredBytes;
  pthread_mutex_unlock(&cacheMutex);
  return buffer;
}

// Create and map a buffer of bin binIndex on a cache miss. The caller has
// already counted it as in use; on failure, undo that count and return NULL.
// Memory is obtained before taking the mutex (see "One mutex"). A buffer is
// never handed out unmapped.
static void *createBufferOnMiss(int binIndex, size_t requestSize) {
  size_t size = cache.bins[binIndex].bufferSize;
  void *buffer = allocWithTrimRetry(allocCachedBuffer, size);
  OMCacheBufferDesc *desc = buffer ? malloc(sizeof(OMCacheBufferDesc)) : NULL;
  if (!desc) {
    // Out of memory even after the trim. OMCacheTryAlloc counted this request
    // as in use before calling us; no buffer is handed out, so undo that count.
    free(buffer); // NULL, or a buffer that got no descriptor.
    pthread_mutex_lock(&cacheMutex);
    cache.bins[binIndex].numInUse--;
    cache.statFailedAllocs++;
    pthread_mutex_unlock(&cacheMutex);
    return NULL;
  }
  pthread_mutex_lock(&cacheMutex);
  return mapNewBufferAndUnlock(desc, buffer, binIndex, requestSize);
}

//===----------------------------------------------------------------------===//
// Cache paths, shared by the interface functions. Each interface function
// calls ensureInit exactly once and checks its arguments (NULL buffer, size
// without a bin, cache off); these functions never do.
//===----------------------------------------------------------------------===//

// Hand out the most recently freed available buffer of the request's bin (a
// hit), or create a new buffer (a miss). Returns NULL only if memory is short
// even after a trim. The caller checked hasBin(size).
static void *tryAllocInCache(size_t size) {
  int binIndex = binIndexFor(size);
  OMCacheBin *bin = &cache.bins[binIndex];
  pthread_mutex_lock(&cacheMutex);
  // Count the request as in use now, before a miss drops the mutex to allocate:
  // numInUse means "requests not yet freed", in-flight misses included.
  // createBufferOnMiss undoes the count if it fails.
  bin->numInUse++;
  if (bin->numInUse > bin->statPeakInUse)
    bin->statPeakInUse = bin->numInUse;
  // Epoch policy: a stale head is returned to the system with the older
  // buffers behind it, and the request gets a fresh buffer.
  OMCacheBufferDesc *detached = NULL;
  OMCacheBufferDesc *head = bin->availableHead; // Most recently freed.
  if (cache.policy == CACHE_POLICY_EPOCH && cache.epoch > 0 &&
      bin->bufferSize >= CACHE_EPOCH_MIN_SIZE) {
    // A request after a free of this bin in this epoch is a reuse, whether or
    // not that buffer was kept (see "eager release").
    if (bin->freesThisEpoch > 0)
      bin->reusesThisEpoch++;
    if (head && head->freedEpoch != cache.epoch)
      bin->statStaleReleases += detachAllAvailableInBin(bin, &detached);
  }
  OMCacheBufferDesc *desc = bin->availableHead; // NULL if stale.
  if (desc) {
    // Hit: the buffer is still mapped, so it is handed out as is.
    removeFromAvailableList(desc);
    desc->inUse = 1;
    statCountHandOut(desc, size);
    bin->statHits++;
    // Not zeroed (see "Debugging and testing"). Read the buffer pointer while
    // the mutex is held.
    void *buffer = desc->buffer;
    pthread_mutex_unlock(&cacheMutex);
    return buffer;
  }
  // Miss.
  bin->statMisses++;
  // Drop the mutex: stale buffers are returned and the new buffer is
  // allocated without it (see createBufferOnMiss).
  pthread_mutex_unlock(&cacheMutex);
  returnDetachedToSystem(detached);
  return createBufferOnMiss(binIndex, size);
}

// Make a cache buffer available again, then evict the oldest available buffers
// while the ceiling is exceeded. Buffers the cache does not own go to free().
// The caller checked that buffer is not NULL and that the cache is enabled.
static void freeInCache(void *buffer) {
  pthread_mutex_lock(&cacheMutex);
  OMCacheBufferDesc *desc = findBufferDesc(buffer);
  if (!desc) {
    // Not a cache buffer: OMCacheAlloc used allocUncachedBuffer (size without
    // a bin, i.e. < 128 KB or > maxCachedSize), or zdnnx used its own
    // allocator. Both give plain malloc/posix_memalign memory, so free() it.
    // (A buffer in use is always mapped; only available buffers are ever
    // detached.)
    pthread_mutex_unlock(&cacheMutex);
    free(buffer);
    return;
  }
  if (!desc->inUse) {
    // Freed twice: keep the buffer in the cache and ignore the call.
    cache.statDoubleFrees++;
    warnMisuseOnce("cache buffer freed twice");
    pthread_mutex_unlock(&cacheMutex);
    return;
  }
  // Kept, not freed (see "Debugging and testing").
  OMCacheBin *bin = &cache.bins[desc->binIndex];
  desc->inUse = 0;
  bin->numInUse--;
  statCountGiveBack(desc);
  OMCacheBufferDesc *detached = NULL;
  int large = cache.policy == CACHE_POLICY_EPOCH && cache.epoch > 0 &&
              bin->bufferSize >= CACHE_EPOCH_MIN_SIZE;
  if (large)
    bin->freesThisEpoch++;
  if (large && bin->eagerRelease) {
    // Eager release: return it after the mutex is dropped. It is in use, so
    // not on the available list: only its mapping is removed.
    removeBufferMapping(desc);
    cache.statRegisteredBytes -= desc->bufferSize;
    bin->statDetached++;
    bin->statEagerReleases++;
    desc->availableNext = detached;
    detached = desc;
  } else {
    addToAvailableList(desc); // Stamps desc->freedAt from cache.freeClock.
    desc->freedEpoch = cache.epoch;
  }
  while (cache.availableBytes > cache.maxAvailableBytes) {
    OMCacheBufferDesc *oldest = findOldestAvailable();
    if (!oldest)
      break;
    detachFromCache(oldest, &detached);
    cache.statEvictions++;
  }
  pthread_mutex_unlock(&cacheMutex);
  returnDetachedToSystem(detached);
}

//===----------------------------------------------------------------------===//
// Interface.
//===----------------------------------------------------------------------===//

// Cached buffer for a size with a bin; NULL for any other size (the caller,
// zdnnx, then uses its own allocator) or if memory is short.
void *OMCacheTryAlloc(size_t size) {
  ensureInit();
  if (!hasBin(size)) // Cache disabled or outside of cache range.
    return NULL;
  return tryAllocInCache(size);
}

// The size decides the path, once: a size with a bin is served by the cache
// (a hit, or a new cache buffer); any other size (too small, too large, or
// cache off) gets an uncached buffer, the same memory as before the cache.
// Either way, a failed allocation is retried once after a trim
// (allocWithTrimRetry); NULL means memory is still short (or size 0).
void *OMCacheAlloc(size_t size) {
  // Size 0: NULL, as OMCacheTryAlloc returns. Compiled code never accesses a
  // 0-byte buffer and frees NULL as a no-op; it also keeps allocWithTrimRetry
  // from mistaking a NULL malloc(0) for a failure.
  if (size == 0)
    return NULL;
  ensureInit();
  if (hasBin(size)) // Cache enabled and inside of cache range.
    return tryAllocInCache(size);
  return allocWithTrimRetry(allocUncachedBuffer, size);
}

// Free a buffer from OMCacheAlloc, OMCacheTryAlloc, or plain malloc. NULL is
// ignored.
void OMCacheFree(void *buffer) {
  if (!buffer)
    return;
  ensureInit();
  if (!cache.enabled) { // Cache disabled.
    free(buffer);
    return;
  }
  freeInCache(buffer);
}

// Free given the allocation size: a size without a bin cannot be a cache
// buffer (hasBin depends only on the configuration, fixed at init), so it is
// freed at once, without the mutex or a registry lookup. This is the common
// case for the many small temporaries of compiled code. Other sizes take the
// cache path (they may still not be cache buffers: zdnnx's own allocation
// after OMCacheTryAlloc returned NULL).
void OMCacheFreeSized(void *buffer, size_t size) {
  if (!buffer)
    return;
  ensureInit();
  if (!hasBin(size)) { // Cache disabled or outside of cache range.
    if (cache.printStats)
      __atomic_fetch_add(&cache.statFastFrees, 1, __ATOMIC_RELAXED);
    free(buffer);
    return;
  }
  // Use cache.
  freeInCache(buffer);
}

// Start (epoch == 0) or end (epoch = the id returned by the start) of an
// inference; called by the compiled entry point around the model's body. See
// steps 2 and 5 at the top of the file.
uint64_t OMCacheEpoch(uint64_t epoch) {
  ensureInit();
  if (!cache.enabled)
    return epoch == 0 ? 1 : 0;
  pthread_mutex_lock(&cacheMutex);
  if (epoch == 0) {
    // Epoch starts.
    uint64_t started = ++cache.epoch;
    cache.statEpochsStarted++;
    for (int i = 0; i < cache.numBins; i++)
      cache.bins[i].freesThisEpoch = cache.bins[i].reusesThisEpoch = 0;
    pthread_mutex_unlock(&cacheMutex);
    return started;
  }
  // Epoch ends.
  OMCacheBufferDesc *detached = NULL;
  cache.statEpochsEnded++;
  if (cache.policy == CACHE_POLICY_EPOCH && epoch == cache.epoch) {
    for (int i = 0; i < cache.numBins; i++) {
      OMCacheBin *bin = &cache.bins[i];
      if (bin->bufferSize < CACHE_EPOCH_MIN_SIZE) // Large bins only.
        continue;
      // Bins used in this epoch: eager release from now on if none of their
      // buffers was reused within it. Bins not used keep their decision.
      if (bin->freesThisEpoch > 0)
        bin->eagerRelease = bin->reusesThisEpoch == 0;
      // No later epoch may reuse them (epoch rule): detach every available
      // buffer of this bin now; they are returned to the system after the mutex
      // is dropped.
      uint64_t n = detachAllAvailableInBin(bin, &detached);
      bin->statStaleReleases += n;
      cache.statEpochEndReleases += n;
    }
  }
  pthread_mutex_unlock(&cacheMutex);
  returnDetachedToSystem(detached);
  return 0;
}

// Trim: return every available buffer to the system now; buffers in use are
// not affected.
void OMCacheTrim(void) {
  ensureInit();
  if (!cache.enabled)
    return;
  pthread_mutex_lock(&cacheMutex);
  OMCacheBufferDesc *detached = NULL;
  detachAllAvailableInCache(&detached);
  cache.statTrims++;
  pthread_mutex_unlock(&cacheMutex);
  returnDetachedToSystem(detached);
}

// Format a byte count for the statistics: KB below 1 MB, MB below 1 GB, then
// GB; one decimal unless the value is a whole number of the unit (bin sizes
// usually are). Writes into buf and returns it.
static const char *formatBytes(size_t bytes, char buf[32]) {
  const char *unit = "KB";
  size_t scale = KB;
  if (bytes >= GB) {
    unit = "GB";
    scale = GB;
  } else if (bytes >= MB) {
    unit = "MB";
    scale = MB;
  }
  if (bytes % scale == 0 || scale == KB)
    snprintf(buf, 32, "%zu %s", bytes / scale, unit);
  else
    snprintf(buf, 32, "%.1f %s", (double)bytes / (double)scale, unit);
  return buf;
}

// Print totals, then one line per bin that was used (OM_MEMORY_CACHE_STATS).
static void printStats(void) {
  char b1[32], b2[32], b3[32];
  fprintf(stderr,
      "==OM-MEMORY-CACHE== policy %s (transparent huge pages %s), epochs "
      "started %llu, ended %llu, buffers released at epoch end %llu\n",
      cache.policy == CACHE_POLICY_EPOCH ? "epoch" : "none",
      cache.thpAvailable ? "on" : "off",
      (unsigned long long)cache.statEpochsStarted,
      (unsigned long long)cache.statEpochsEnded,
      (unsigned long long)cache.statEpochEndReleases);
  fprintf(stderr,
      "==OM-MEMORY-CACHE== max available %s, max registered %s, evictions "
      "%llu, trims %llu, alloc-failure trims %llu, orphan descs %llu, double "
      "frees %llu, failed allocs %llu\n",
      formatBytes(cache.statPeakAvailableBytes, b1),
      formatBytes(cache.statPeakRegisteredBytes, b2),
      (unsigned long long)cache.statEvictions,
      (unsigned long long)cache.statTrims,
      (unsigned long long)cache.statAllocFailureTrims,
      (unsigned long long)cache.statOrphanDescs,
      (unsigned long long)cache.statDoubleFrees,
      (unsigned long long)cache.statFailedAllocs);
  fprintf(stderr,
      "==OM-MEMORY-CACHE== peak in use at once: %s of buffers, %s requested "
      "(vs max registered %s)\n",
      formatBytes(cache.statPeakInUseBytes, b1),
      formatBytes(cache.statPeakInUseRequestBytes, b2),
      formatBytes(cache.statPeakRegisteredBytes, b3));
  fprintf(stderr,
      "==OM-MEMORY-CACHE== registry: peak %zu buffers, longest chain %zu "
      "(%d buckets); %llu frees without the lock (size without a bin)\n",
      cache.statPeakRegistered, cache.statMaxBucketChain,
      CACHE_REGISTRY_BUCKETS, (unsigned long long)cache.statFastFrees);
  for (int i = 0; i < cache.numBins; i++) {
    OMCacheBin *bin = &cache.bins[i];
    if (bin->statHits + bin->statMisses == 0)
      continue;
    fprintf(stderr,
        "==OM-MEMORY-CACHE== bin %s: hits %llu, misses %llu, peak in use "
        "%llu, available %llu, in use %llu, detached %llu (stale releases "
        "%llu, eager releases %llu)\n",
        formatBytes(bin->bufferSize, b1), (unsigned long long)bin->statHits,
        (unsigned long long)bin->statMisses,
        (unsigned long long)bin->statPeakInUse,
        (unsigned long long)bin->statNumAvailable,
        (unsigned long long)bin->numInUse,
        (unsigned long long)bin->statDetached,
        (unsigned long long)bin->statStaleReleases,
        (unsigned long long)bin->statEagerReleases);
  }
}

// Runs when the model .so is unloaded, or at process exit. Buffers still in
// use (there should be none) are left alone, and so are their descriptors: a
// late OMCacheFree (another destructor, or a thread still running at exit)
// must still find them in the registry, whose buckets are static. That leaks
// them once per dlclose. If the mutex is held (a thread still in the cache at
// exit, or a child forked while it was held where there are no fork
// handlers), nothing is returned rather than risking a hang.
// TODO(z/OS, if the cache is built there): check that the destructor runs when
// the model DLL is freed with dllfree/dlclose; if not, available buffers are
// only reclaimed at process exit.
__attribute__((destructor)) static void finalizeCache(void) {
  if (!cache.enabled)
    return;
  if (pthread_mutex_trylock(&cacheMutex) != 0)
    return;
  if (cache.printStats)
    printStats();
  OMCacheBufferDesc *detached = NULL;
  detachAllAvailableInCache(&detached);
  pthread_mutex_unlock(&cacheMutex);
  returnDetachedToSystem(detached);
}

#ifdef __cplusplus
}
#endif
