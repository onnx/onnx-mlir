#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
#
# Manual tests of the NNPA runtime memory cache (OMMemoryCache.c). Not part of
# any CI target; see README.md in this directory.

# Works when run (./run.sh, sh run.sh) and when sourced (. run.sh): the body
# runs in a subshell, so nothing leaks into the caller's shell, and the script
# finds its directory from BASH_SOURCE when sourced from bash.
omMemoryCacheTests() (
set -u
script=${BASH_SOURCE:-$0}
here=$(cd "$(dirname "$script")" && pwd)
src="$here/../../../../../src/Accelerators/NNPA/Runtime"
if [ ! -f "$src/OMMemoryCache.c" ]; then
  echo "error: OMMemoryCache.c not found under $src" >&2
  echo "run this script from its directory in an onnx-mlir checkout" >&2
  exit 2
fi
src=$(cd "$src" && pwd)
out=${TMPDIR:-/tmp}/om-memory-cache-test
mkdir -p "$out"
CC=${CC:-cc}
SAN=${SAN--fsanitize=address,undefined}
REPEAT=${REPEAT:-5}
flags="-std=gnu99 -g -O1 -Wall -pthread -I$src"

# The tests set their own configuration: start from the defaults.
unset OM_MEMORY_CACHE_MAX OM_MEMORY_CACHE_LARGE OM_MEMORY_CACHE_STATS \
  OM_MEMORY_CACHE_POLICY

fail=0

# build <name> <args...>: with sanitizers if they compile, else without.
build() {
  name=$1
  shift
  if [ -n "$SAN" ] && $CC $flags $SAN "$@" -o "$out/$name" 2>"$out/$name.build.log"; then
    return 0
  fi
  [ -n "$SAN" ] && echo "note: $name built without sanitizers (see $out/$name.build.log)"
  if ! $CC $flags "$@" -o "$out/$name" 2>>"$out/$name.build.log"; then
    echo "BUILD FAIL $name (see $out/$name.build.log)"
    fail=1
    return 1
  fi
}

# run <label> <command...>
run() {
  label=$1
  shift
  printf '%-28s' "$label"
  if "$@" >"$out/$label.out" 2>&1; then
    echo PASS
  else
    echo "FAIL (see $out/$label.out)"
    fail=1
  fi
}

build test_cache "$here/test_cache.c"
build test_stale "$here/test_stale.c"
build test_alloc_failure "$here/test_alloc_failure.c"
build test_epoch "$here/test_epoch.c"
# The .so and its loader are built without sanitizers: dlopen'ing an
# uninstrumented library is the case to test.
$CC $flags -fPIC -shared "$here/dso_cache.c" -o "$out/libomcachetest.so" \
  2>"$out/dso.build.log" || { echo "BUILD FAIL dso_cache"; fail=1; }
$CC $flags "$here/test_dlclose_fork.c" -o "$out/test_dlclose_fork" -ldl \
  2>"$out/test_dlclose_fork.build.log" ||
  { echo "BUILD FAIL test_dlclose_fork"; fail=1; }

i=1
while [ "$i" -le "$REPEAT" ]; do
  run "test_cache.$i" env OM_MEMORY_CACHE_LARGE=64M "$out/test_cache"
  i=$((i + 1))
done
run test_cache.stats env OM_MEMORY_CACHE_LARGE=64M OM_MEMORY_CACHE_STATS=1 \
  "$out/test_cache"
run test_stale "$out/test_stale"
run test_alloc_failure "$out/test_alloc_failure"
run test_epoch "$out/test_epoch" epoch
run test_epoch.none "$out/test_epoch" none
run test_epoch.default "$out/test_epoch" default
run test_dlclose_fork "$out/test_dlclose_fork" "$out/libomcachetest.so"

if [ "$fail" -eq 0 ]; then
  echo "ALL PASS (outputs in $out)"
else
  echo "SOME TESTS FAILED (outputs in $out)"
fi
exit "$fail"
)

omMemoryCacheTests
