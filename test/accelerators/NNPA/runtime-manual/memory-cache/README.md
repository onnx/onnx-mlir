<!--- SPDX-License-Identifier: Apache-2.0 -->

# Manual tests of the NNPA runtime memory cache

White-box tests of `src/Accelerators/NNPA/Runtime/OMMemoryCache.c`. Each test
`#include`s the `.c` file to check internal state, and replaces
`OMHugePageMalloc` with plain `malloc`, so neither zDNN nor an NNPA machine is
needed.

These tests are **not part of any CI target**. No `CMakeLists.txt` adds this
directory, and lit does not collect `.c` files. Run them by hand after changing
the cache, and on each new platform (Linux on Z, z/OS).

## Running

```sh
test/accelerators/NNPA/runtime-manual/memory-cache/run.sh
```

`./run.sh`, `sh run.sh` and `. run.sh` (from bash) all work, from any working
directory; the script finds the sources relative to its own location, and runs
in a subshell so that nothing leaks into the calling shell.

The script builds into `$TMPDIR/om-memory-cache-test` (or `/tmp/...`), and
prints PASS or FAIL per test. Settings:

| Variable | Default | Use |
|---|---|---|
| `CC` | `cc` | compiler, e.g. `gcc`, `clang`, `ibm-clang64` on z/OS |
| `SAN` | `-fsanitize=address,undefined` | sanitizer flags; set `SAN=` to build without (the script also falls back on its own if they do not compile) |
| `REPEAT` | `5` | runs of `test_cache`, whose thread and fork timing varies |

The script unsets `OM_MEMORY_CACHE_*` and each test sets its own
configuration: `test_cache` runs with `OM_MEMORY_CACHE_LARGE=64M` (to cover
bins up to 64 MB), once more with `OM_MEMORY_CACHE_STATS=1` (which enables the
registry statistics), and `test_cold` sets `OM_MEMORY_CACHE_COLD` itself (8, then 0) and checks
the default largest cached size (256M).

## Tests

| Test | Checks |
|---|---|
| `test_cache` | bin rounding; 100% hits from the 2nd repeated "inference"; registry with 40 forced collisions (remove head, tail, middle); trim; sized free (`OMCacheFreeSized`: sizes without a bin skip the lock); double free; ceiling eviction and its order across bins; 8 threads, with forks in the middle (Linux: through the runtime's own fork handlers; macOS and z/OS have none, see below) |
| `test_stale` | a buffer released with plain `free()` whose address comes back from `posix_memalign`: the stale descriptor is replaced and the counts repaired |
| `test_cold` | `OM_MEMORY_CACHE_COLD`: a hot 8 MB bin stays cached; after a long idle gap a request is cold and its buffer is released when freed; one close reuse makes the bin hot again; the decision follows each buffer when several are in use; bins below 4 MB are never released; with `off`, nothing is released; default largest cached size 256M |
| `test_alloc_failure` | `posix_memalign` failing: trim and retry succeeds; nothing to trim gives NULL; retry failing too gives NULL; in-use counts are undone |
| `test_dlclose_fork` | load a `.so` that contains the cache, use it, `dlclose` it, then `fork()`. glibc must have removed the fork handlers; if not, the child jumps into unmapped code (this crashed on macOS before the runtime stopped registering handlers there) |

## Platform notes

- **Linux on Z:** the main target. `test_dlclose_fork` checks the
  `TODO(Linux)` in `registerForkHandlers`.
- **macOS:** the runtime registers no fork handlers, so `test_cache`
  registers equivalent ones itself for its fork-with-threads section.
- **z/OS:** the runtime registers no fork handlers, and a child of a
  multithreaded process may not use the cache, so `test_cache` skips the
  fork section. The remaining open items are the `TODO(z/OS)` notes in
  `OMMemoryCache.c` (the destructor at `dllfree`, `pthread_atfork`).
