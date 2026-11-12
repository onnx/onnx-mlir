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
// Design decisions:
// - Why: glibc often returns freed memory to the kernel, so every inference
//   refaults and rezeroes it (and pays a TLB purge per page on s390x).
// - Scope: buffers the model frees itself (selected by the pass
//   ReplaceMallocByHugePageMalloc) and zdnnx scratch. Outputs are not cached.
// - Each buffer is its own posix_memalign. A buffer must still be given back
//   with OMCacheFree, never with plain free(): that would leave a stale entry
//   in the registry, and a later allocation at the same address that reaches
//   OMCacheFree would be taken for a cache buffer of the wrong size. All
//   current callers pair OMCacheAlloc/OMCacheTryAlloc with OMCacheFree.
// - Bins: 64 KB steps to 1 MB, then 1 MB multiples, <= 4 per doubling; no
//   splitting. < 128 KB (glibc reuses those well) and > 256 MB are not cached.
// - Large buffers that went cold are not reused: writing into a cold buffer
//   first fetches its stale contents from memory, which is slower than a
//   fresh, kernel-zeroed mapping (granite-embedding S=2048 on z17: Stick 11.9
//   vs 7.6 ms into its 96 MB mask), while a large buffer reused every layer
//   is worth keeping (granite-4 prefill). Epoch policy (default): a buffer of
//   16 MB or more freed in an earlier inference is never reused; the compiler
//   marks each inference with OMCacheEpoch. Only with transparent huge pages
//   (otherwise all buffers are kept): with 4 KB pages, releasing is the
//   expensive part. OM_MEMORY_CACHE_POLICY=none keeps all buffers.
// - Freed buffers are kept available (peak use per bin at fixed shapes); a
//   ceiling on available bytes evicts the oldest first. When an allocation
//   fails, all available buffers are returned to the system and the
//   allocation is retried once.
// - One mutex. Buffers are obtained from and returned to the system outside
//   of it; only the release of a stale descriptor calls free while holding it.
//
// Environment variables: see OMMemoryCache.h.
//
// Debugging and testing:
// - Reused buffers are NOT zeroed: a hit returns the data of an earlier
//   inference, where a fresh large malloc used to return zero pages. Code that
//   reads a buffer before writing it was hidden before and now gives wrong or
//   run-to-run different results, typically from the second inference in a
//   process on. To check: compare outputs with OM_MEMORY_CACHE_MAX=0 (cache
//   off), and run several inferences in one process. Not yet done: running
//   the NNPA tests with poisoned reused buffers (e.g. memset to 0xFF on a hit).
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

// Defined in OMRuntimeNNPA.c.
void *OMHugePageMalloc(size_t size);

#define KB ((size_t)1024)
#define MB (KB * KB)
#define GB (KB * MB)

#define CACHE_MIN_SIZE (128 * KB)
#define CACHE_SMALL_BIN_STEP (64 * KB)
// Largest cached request by default; see "Bins" in the design notes above.
#define CACHE_DEFAULT_LARGE (256 * MB)
// Epoch policy: buffers from this size are never reused across inferences.
#define CACHE_EPOCH_MIN_SIZE (16 * MB)
// OM_MEMORY_CACHE_POLICY values.
#define CACHE_POLICY_NONE 0  // Keep every buffer.
#define CACHE_POLICY_EPOCH 1 // Default: large buffers not reused across epochs.
// Per model .so. If memory use is too high, see "Debugging and testing" above.
#define CACHE_DEFAULT_MAX (1 * GB)
#define CACHE_PAGE_4K (4 * KB)
#define CACHE_HUGE_PAGE (1 * MB)
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
//   incremented at each OMCacheFree. The oldest available buffer of the whole
//   cache is the bin tail with the smallest stamp (the eviction order).
// - epoch: one inference, delimited by OMCacheEpoch calls that the compiler
//   wraps around the model's body. A kept buffer records the epoch of its
//   free.
// - stale buffer (epoch policy): an available buffer of a bin of 16 MB or more
//   freed in an earlier epoch. Its memory has likely left the processor
//   caches. A request that finds one returns it and the older ones of its bin
//   to the system and gets a fresh buffer. The available buffers of 16 MB or
//   more are also returned at the end of each epoch.
// - registry: hash maps a buffer to its descriptor.
// - detach: take an available buffer out of every cache structure (available
//   list and registry), under the mutex.
// - return to the system: free() detached buffers and their descriptors,
//   after the mutex is dropped.
// - evict / trim: reasons to detach. Evict: over the ceiling on available
//   bytes, oldest first. Trim: all available buffers (OMCacheTrim, failed
//   allocation, unload).
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
  // Statistics. statDetached: by eviction, trim, unload or stale release.
  // statStaleReleases: stale buffers returned (at a request or at the end of
  // an epoch).
  uint64_t statNumAvailable, statPeakInUse;
  uint64_t statHits, statMisses, statDetached, statStaleReleases;
} OMCacheBin;

// The whole cache: one static instance per model .so, guarded by cacheMutex
// (except the configuration and bin sizes, which are fixed after init).
typedef struct {
  // Configuration, from the environment.
  int enabled, printStats;
  size_t maxAvailableBytes; // Ceiling on the bytes of available buffers.
  size_t maxCachedSize; // Largest cached request (buffer size of the last bin).
  // OM_MEMORY_CACHE_POLICY (CACHE_POLICY_*). Without transparent huge pages
  // the default is CACHE_POLICY_NONE: releasing and mapping fresh is only
  // cheap with huge pages.
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
  // buffers. statStaleDescs: descriptors of buffers someone released with
  // plain free(), found when their address came back from posix_memalign.
  // statAllocFailureTrims: trims after a failed allocation.
  size_t statRegisteredBytes, statPeakAvailableBytes, statPeakRegisteredBytes;
  uint64_t statEvictions, statTrims, statAllocFailureTrims;
  uint64_t statStaleDescs, statDoubleFrees, statFailedAllocs;
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
} OMCache;

// The cache state and its lock; initCache runs once through cacheOnce.
static OMCache cache;
static pthread_mutex_t cacheMutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t cacheOnce = PTHREAD_ONCE_INIT;

//===----------------------------------------------------------------------===//
// Initialization.
//===----------------------------------------------------------------------===//

// Parse "<number>[K|M|G]"; return dflt if the variable is unset, malformed,
// negative, or too large for size_t (including after the K/M/G scaling).
static size_t getEnvSize(const char *name, size_t dflt) {
  const char *s = getenv(name);
  if (!s)
    return dflt;
  while (*s == ' ' || *s == '\t')
    s++;
  // strtoull accepts a sign and negates the value: reject both.
  if (*s < '0' || *s > '9')
    return dflt;
  char *end = NULL;
  errno = 0;
  unsigned long long v = strtoull(s, &end, 10);
  if (errno == ERANGE || end == s)
    return dflt;
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
    return dflt;
  if (v > (unsigned long long)SIZE_MAX / scale)
    return dflt;
  return (size_t)(v * scale);
}

#ifdef __MVS__
// No fork handlers on z/OS. The z/OS pthread_atfork documentation says that
// locking in the prepare handler and unlocking in the child handler is "not
// practical on the z/OS platform", and that a child of a multithreaded process
// may only call async-signal-safe functions before exec (POSIX requires the
// same everywhere), so such a child must not use the cache.
// Checked: z/OS removes the handlers of a DLL when it is freed, so registering
// them would have been safe across dlclose.
// TODO(z/OS): confirm that no z/OS host forks a multithreaded process and then
// runs a model in the child without exec.
static int registerForkHandlers(void) { return 1; }
#elif defined(__APPLE__)
// No fork handlers on macOS (used only for development builds of the NNPA
// runtime). Tested: macOS does not remove pthread_atfork handlers when the
// library that registered them is closed, so the first fork() after the model
// .so is unloaded jumped into unmapped code and crashed. Without handlers, a
// fork while another thread holds the cache mutex leaves the child unable to
// use the cache, as on z/OS.
static int registerForkHandlers(void) { return 1; }
#else
// Fork handlers: hold the mutex across fork() so that the child gets the cache
// in a consistent state, then release it in both parent and child.
static void lockForFork(void) { pthread_mutex_lock(&cacheMutex); }
static void unlockForFork(void) { pthread_mutex_unlock(&cacheMutex); }

// Return 0 if the handlers could not be registered (ENOMEM). glibc removes
// handlers registered from a dlopen'ed library when the library is closed
// (pthread_atfork passes the library's __dso_handle). TODO(Linux): run the
// dlopen/dlclose-then-fork test on s390x to confirm; it was only run on macOS,
// where it crashed before the __APPLE__ case above was added.
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
#if defined(__linux__) && defined(MADV_HUGEPAGE)
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
  cache.maxAvailableBytes =
      getEnvSize("OM_MEMORY_CACHE_MAX", CACHE_DEFAULT_MAX);
  cache.maxCachedSize =
      getEnvSize("OM_MEMORY_CACHE_LARGE", CACHE_DEFAULT_LARGE);
  const char *stats = getenv("OM_MEMORY_CACHE_STATS");
  cache.printStats = stats && *stats && *stats != '0';
  // Policy: an explicit OM_MEMORY_CACHE_POLICY wins; by default, epoch with
  // transparent huge pages and none without.
  cache.thpAvailable = transparentHugePagesAvailable();
  cache.policy = cache.thpAvailable ? CACHE_POLICY_EPOCH : CACHE_POLICY_NONE;
  const char *policy = getenv("OM_MEMORY_CACHE_POLICY");
  if (policy && strcmp(policy, "epoch") == 0)
    cache.policy = CACHE_POLICY_EPOCH;
  else if (policy && strcmp(policy, "none") == 0)
    cache.policy = CACHE_POLICY_NONE;
  if (cache.maxCachedSize > CACHE_MAX_LARGE)
    cache.maxCachedSize = CACHE_MAX_LARGE;
  cache.enabled =
      cache.maxAvailableBytes > 0 && cache.maxCachedSize >= CACHE_MIN_SIZE;
  if (!cache.enabled)
    return;
  // Keep the mutex usable in a child process forked while it was held. Done
  // first: if it fails, the cache is disabled before any buffer exists, so a
  // forked child cannot deadlock.
  if (!registerForkHandlers()) {
    cache.enabled = 0;
    if (cache.printStats)
      fprintf(stderr, "==OM-MEMORY-CACHE== disabled: pthread_atfork failed\n");
    return;
  }
  // Bin sizes: 64 KB steps up to 1 MB; then multiples of 1 MB, with a step of
  // a quarter of the largest power of two below the size.
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
  // Statistics only, and only when they are printed: the chain walk is the one
  // statistic with a loop (it runs on a miss, not on the hot path).
  if (!cache.printStats)
    return;
  if (cache.numRegistered > cache.statPeakRegistered)
    cache.statPeakRegistered = cache.numRegistered;
  size_t chain = 0;
  for (OMCacheBufferDesc *d = desc; d; d = d->bucketNext)
    chain++;
  if (chain > cache.statMaxBucketChain)
    cache.statMaxBucketChain = chain;
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
// removeFromAvailableList vs detachFromCache:
//
// |                       | removeFromAvailableList | detachFromCache         |
// |-----------------------|-------------------------|-------------------------|
// | bin's available list  | removed                 | removed (calls          |
// |                       |                         | removeFromAvailableList)|
// | availableBytes        | decreased               | decreased               |
// | registry              | stays mapped            | mapping removed         |
// | afterwards, buffer is | still owned by the      | unknown to the cache,   |
// |                       | cache, now in use       | queued on detached, to  |
// |                       |                         | be returned to system   |
// | used by               | a hit in                | evict; trim (from       |
// |                       | OMCacheTryAlloc; detach | OMCacheTrim, an alloc   |
// |                       |                         | failure, or unload)     |
//
// removeFromAvailableList is a building block: a buffer changes state from
// available to in use. Detach means the buffer leaves the cache for good.
//

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
  // for an eviction, anywhere for a stale descriptor.
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
// Called without the mutex, and it must stay so: freeing a large buffer
// munmaps it, and in a process that has run on several CPUs each released
// page costs a global TLB purge (6 MB measured on z17: about 1.3 ms with 4 KB
// pages, 17 us with huge pages), which would stall every thread waiting on the
// mutex; a trim releases many such buffers at once.
static void returnDetachedToSystem(OMCacheBufferDesc *detached) {
  while (detached) {
    OMCacheBufferDesc *nextDesc = detached->availableNext;
    free(detached->buffer);
    free(detached);
    detached = nextDesc;
  }
}

// Detach every available buffer into *detached; return how many were detached.
static size_t detachAllAvailable(OMCacheBufferDesc **detached) {
  size_t n = 0;
  for (int i = 0; i < cache.numBins; i++)
    for (; cache.bins[i].availableHead; n++)
      detachFromCache(cache.bins[i].availableHead, detached);
  return n;
}

// Return the oldest available buffer of the cache, or NULL if there is none.
// Each bin's available list is most recent first, so its tail is the bin's
// oldest; the cache's oldest is the bin tail with the smallest age stamp. This
// scans the bins (a few dozen), and is only used for eviction.
static OMCacheBufferDesc *findOldestAvailable(void) {
  OMCacheBufferDesc *oldest = NULL;
  for (int i = 0; i < cache.numBins; i++) {
    OMCacheBufferDesc *tail = cache.bins[i].availableTail;
    if (tail && (!oldest || tail->freedAt < oldest->freedAt))
      oldest = tail;
  }
  return oldest;
}

// Detach all available buffers of one bin into *detached (stale buffers);
// return how many were detached.
static uint64_t detachBinAvailable(
    OMCacheBin *bin, OMCacheBufferDesc **detached) {
  uint64_t n = 0;
  for (; bin->availableHead; n++)
    detachFromCache(bin->availableHead, detached);
  return n;
}

// Trim after an allocation failed: return every available buffer to the
// system, so that the retry can use that memory. Return 1 if any buffer was
// returned. Takes the mutex.
static int trimOnAllocFailure(void) {
  if (!cache.enabled)
    return 0;
  pthread_mutex_lock(&cacheMutex);
  OMCacheBufferDesc *detached = NULL;
  size_t n = detachAllAvailable(&detached);
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

// Get a new buffer from the system: 4 KB aligned (zdnnx needs it), or 1 MB
// aligned and advised for huge pages from 1 MB up, as OMHugePageMalloc does.
// Checked: posix_memalign is available on z/OS (OMExternalConstant.c also uses
// it there), and its memory may be released with free().
static void *allocBuffer(size_t size) {
  void *buffer = NULL;
  size_t align = size >= CACHE_HUGE_PAGE ? CACHE_HUGE_PAGE : CACHE_PAGE_4K;
  if (posix_memalign(&buffer, align, size) != 0)
    return NULL;
#if defined(__linux__) && defined(MADV_HUGEPAGE)
  // Advise once, when the buffer is created; it is ok if the advice fails.
  if (size >= CACHE_HUGE_PAGE)
    madvise(buffer, size, MADV_HUGEPAGE);
#endif
  return buffer;
}

// Fill desc for a new in-use buffer of bin binIndex, handed out for a request
// of requestSize bytes, and map buffer to it. Called with the mutex held;
// releases it and returns buffer.
static void *mapNewBufferAndUnlock(
    OMCacheBufferDesc *desc, void *buffer, int binIndex, size_t requestSize) {
  size_t size = cache.bins[binIndex].bufferSize;
  // A buffer already mapped at this address was given to free() by someone
  // else and its memory reused by malloc: forget the stale descriptor.
  OMCacheBufferDesc *staleDesc = findBufferDesc(buffer);
  if (staleDesc) {
    removeBufferMapping(staleDesc);
    if (staleDesc->inUse) {
      cache.bins[staleDesc->binIndex].numInUse--;
      statCountGiveBack(staleDesc);
    } else {
      removeFromAvailableList(staleDesc);
    }
    cache.statRegisteredBytes -= staleDesc->bufferSize;
    cache.statStaleDescs++;
    free(staleDesc);
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
// Memory is obtained before taking the mutex: allocBuffer may mmap and madvise
// a large buffer (a few to tens of us), and other threads (zdnnx OpenMP
// workers, concurrent sessions) must not wait on the mutex meanwhile. Misses
// are rare (warm-up, shape changes), so this is a minor gain; the same rule
// matters far more for returnDetachedToSystem. If the descriptor or the buffer
// cannot be allocated, every available buffer is returned to the system and
// the creation is retried once. A buffer is never handed out unmapped.
static void *createBufferOnMiss(int binIndex, size_t requestSize) {
  size_t size = cache.bins[binIndex].bufferSize;
  for (int attempt = 0;; attempt++) {
    OMCacheBufferDesc *desc = malloc(sizeof(OMCacheBufferDesc));
    void *buffer = desc ? allocBuffer(size) : NULL;
    pthread_mutex_lock(&cacheMutex);
    if (buffer) // mapNewBufferAndUnlock releases the mutex.
      return mapNewBufferAndUnlock(desc, buffer, binIndex, requestSize);
    // Failure: buffer is NULL here; desc may be non-NULL (only allocBuffer
    // failed), so free it. On the first attempt, trim (detach all available
    // buffers) and retry if there were any.
    OMCacheBufferDesc *detached = NULL;
    size_t released = 0;
    if (attempt == 0)
      released = detachAllAvailable(&detached);
    if (released) {
      // Trimmed: retry once with that memory back in the system.
      cache.statAllocFailureTrims++;
    } else {
      // Giving up (nothing to trim, or the retry failed too). OMCacheTryAlloc
      // counted this request as in use before calling us; no buffer is handed
      // out, so undo that count. Function will return null after cleanup.
      cache.bins[binIndex].numInUse--;
      cache.statFailedAllocs++;
    }
    pthread_mutex_unlock(&cacheMutex);
    free(desc);
    returnDetachedToSystem(detached);
    if (!released)
      return NULL;
  }
}

//===----------------------------------------------------------------------===//
// Interface.
//===----------------------------------------------------------------------===//

// Hand out the most recently freed available buffer of the request's bin (a
// hit), or create a new buffer (a miss). Returns NULL if the size has no bin.
void *OMCacheTryAlloc(size_t size) {
  ensureInit();
  if (!hasBin(size))
    return NULL;
  int binIndex = binIndexFor(size);
  OMCacheBin *bin = &cache.bins[binIndex];
  pthread_mutex_lock(&cacheMutex);
  // Count the request as in use now, before a miss drops the mutex to allocate:
  // numInUse means "requests not yet freed", in-flight misses included (the
  // planned epoch logic will rely on it). createBufferOnMiss undoes the count
  // if it fails.
  bin->numInUse++;
  if (bin->numInUse > bin->statPeakInUse)
    bin->statPeakInUse = bin->numInUse;
  // Epoch policy: a stale head (see "stale buffer" in the vocabulary) is
  // returned to the system with the older buffers behind it, and the request
  // gets a fresh buffer.
  OMCacheBufferDesc *detached = NULL;
  OMCacheBufferDesc *head = bin->availableHead; // Most recently freed.
  if (cache.policy == CACHE_POLICY_EPOCH && head && cache.epoch > 0 &&
      bin->bufferSize >= CACHE_EPOCH_MIN_SIZE &&
      head->freedEpoch != cache.epoch)
    bin->statStaleReleases += detachBinAvailable(bin, &detached);
  OMCacheBufferDesc *desc = bin->availableHead; // NULL if stale.
  if (desc) {
    removeFromAvailableList(desc);
    desc->inUse = 1;
    statCountHandOut(desc, size);
    bin->statHits++;
    // The buffer still holds data from its previous use; it is not zeroed
    // (see "Debugging and testing" at the top of the file).
    // Read the buffer pointer while the mutex is held.
    void *buffer = desc->buffer;
    pthread_mutex_unlock(&cacheMutex);
    return buffer;
  }
  bin->statMisses++;
  // Drop the mutex: stale buffers are returned and the new buffer is
  // allocated without it (see createBufferOnMiss).
  pthread_mutex_unlock(&cacheMutex);
  returnDetachedToSystem(detached);
  return createBufferOnMiss(binIndex, size);
}

// Cached buffer if possible; otherwise the same allocation as before the
// cache. If that fails too, the available buffers are returned to the system
// and the allocation is retried once (this also covers sizes without a bin).
void *OMCacheAlloc(size_t size) {
  void *buffer = OMCacheTryAlloc(size);
  if (buffer)
    return buffer;
  buffer = OMHugePageMalloc(size);
  if (!buffer && size > 0 && trimOnAllocFailure())
    buffer = OMHugePageMalloc(size);
  return buffer;
}

// Make a cache buffer available again, then evict the oldest available buffers
// while the ceiling is exceeded. Buffers the cache does not own go to free().
void OMCacheFree(void *buffer) {
  if (!buffer)
    return;
  ensureInit();
  if (!cache.enabled) {
    free(buffer);
    return;
  }
  pthread_mutex_lock(&cacheMutex);
  OMCacheBufferDesc *desc = findBufferDesc(buffer);
  if (!desc) {
    // Not a cache buffer: OMCacheAlloc fell back to OMHugePageMalloc (size
    // without a bin, i.e. < 128 KB or > maxCachedSize, or a failed cache
    // allocation), or zdnnx used its own allocator. Both give plain
    // malloc/posix_memalign memory, so free() it. (A buffer in use is always
    // mapped; only available buffers are ever detached.)
    pthread_mutex_unlock(&cacheMutex);
    free(buffer);
    return;
  }
  if (!desc->inUse) {
    // Freed twice: keep the buffer in the cache and ignore the call.
    cache.statDoubleFrees++;
    pthread_mutex_unlock(&cacheMutex);
    return;
  }
  // Kept, not freed: ASan and Valgrind cannot see a later use of buffer (see
  // "Debugging and testing" at the top of the file).
  OMCacheBin *bin = &cache.bins[desc->binIndex];
  desc->inUse = 0;
  bin->numInUse--;
  statCountGiveBack(desc);
  OMCacheBufferDesc *detached = NULL;
  addToAvailableList(desc); // Stamps desc->freedAt from cache.freeClock.
  desc->freedEpoch = cache.epoch;
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

// Free given the allocation size: a size without a bin cannot be a cache
// buffer (hasBin depends only on the configuration, fixed at init), so it is
// freed at once, without the mutex or a registry lookup. This is the common
// case for the many small temporaries of compiled code. Other sizes take the
// OMCacheFree path (they may still not be cache buffers, e.g. after a failed
// cache allocation).
void OMCacheFreeSized(void *buffer, size_t size) {
  if (!buffer)
    return;
  ensureInit();
  if (!hasBin(size)) {
    if (cache.printStats)
      __atomic_fetch_add(&cache.statFastFrees, 1, __ATOMIC_RELAXED);
    free(buffer);
    return;
  }
  OMCacheFree(buffer);
}

// Start (epoch == 0) or end (epoch = the id returned by the start) of an
// inference; called by the compiled entry point around the model's body. The
// epoch policy never reuses buffers of 16 MB or more across inferences: at
// the end, if no later inference has started meanwhile, those available
// buffers are returned to the system at once. Concurrent inferences on one
// model are correct but reuse less (each start makes earlier buffers stale).
uint64_t OMCacheEpoch(uint64_t epoch) {
  ensureInit();
  if (!cache.enabled)
    return epoch == 0 ? 1 : 0;
  OMCacheBufferDesc *detached = NULL;
  pthread_mutex_lock(&cacheMutex);
  if (epoch == 0) {
    uint64_t started = ++cache.epoch;
    cache.statEpochsStarted++;
    pthread_mutex_unlock(&cacheMutex);
    return started;
  }
  cache.statEpochsEnded++;
  if (cache.policy == CACHE_POLICY_EPOCH && epoch == cache.epoch) {
    for (int i = 0; i < cache.numBins; i++) {
      OMCacheBin *bin = &cache.bins[i];
      if (bin->bufferSize < CACHE_EPOCH_MIN_SIZE || !bin->availableHead)
        continue;
      uint64_t n = detachBinAvailable(bin, &detached);
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
  detachAllAvailable(&detached);
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
      "%llu, trims %llu, alloc-failure trims %llu, stale descs %llu, double "
      "frees %llu, failed allocs %llu\n",
      formatBytes(cache.statPeakAvailableBytes, b1),
      formatBytes(cache.statPeakRegisteredBytes, b2),
      (unsigned long long)cache.statEvictions,
      (unsigned long long)cache.statTrims,
      (unsigned long long)cache.statAllocFailureTrims,
      (unsigned long long)cache.statStaleDescs,
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
        "%llu)\n",
        formatBytes(bin->bufferSize, b1), (unsigned long long)bin->statHits,
        (unsigned long long)bin->statMisses,
        (unsigned long long)bin->statPeakInUse,
        (unsigned long long)bin->statNumAvailable,
        (unsigned long long)bin->numInUse,
        (unsigned long long)bin->statDetached,
        (unsigned long long)bin->statStaleReleases);
  }
}

// Runs when the model .so is unloaded, or at process exit. Buffers still in
// use (there should be none) are left alone, and so are their descriptors: a
// late OMCacheFree (another destructor, or a thread still running at exit)
// must still find them in the registry, whose buckets are static. That leaks
// them once per dlclose.
// TODO(z/OS): Open XL accepts __attribute__((destructor)); not yet verified
// that it runs when the model DLL is freed with dllfree/dlclose (if not, the
// available buffers are only reclaimed at process exit).
__attribute__((destructor)) static void finalizeCache(void) {
  if (!cache.enabled)
    return;
  pthread_mutex_lock(&cacheMutex);
  if (cache.printStats)
    printStats();
  OMCacheBufferDesc *detached = NULL;
  detachAllAvailable(&detached);
  pthread_mutex_unlock(&cacheMutex);
  returnDetachedToSystem(detached);
}

#ifdef __cplusplus
}
#endif
