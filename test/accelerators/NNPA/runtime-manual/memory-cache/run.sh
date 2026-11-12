#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
#
# Manual tests of the NNPA runtime memory cache (OMMemoryCache.c). Not part of
# any CI target; see README.md in this directory.

set -u
here=$(cd "$(dirname "$0")" && pwd)
src=$(cd "$here/../../../../../src/Accelerators/NNPA/Runtime" && pwd)
out=${TMPDIR:-/tmp}/om-memory-cache-test
mkdir -p "$out"
CC=${CC:-cc}
SAN=${SAN--fsanitize=address,undefined}
REPEAT=${REPEAT:-5}
flags="-std=gnu99 -g -O1 -Wall -pthread -I$src"

# The tests assume the default configuration.
unset OM_MEMORY_CACHE_MAX OM_MEMORY_CACHE_LARGE OM_MEMORY_CACHE_STATS

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
# The .so and its loader are built without sanitizers: dlopen'ing an
# uninstrumented library is the case to test.
$CC $flags -fPIC -shared "$here/dso_cache.c" -o "$out/libomcachetest.so" \
  2>"$out/dso.build.log" || { echo "BUILD FAIL dso_cache"; fail=1; }
$CC $flags "$here/test_dlclose_fork.c" -o "$out/test_dlclose_fork" -ldl \
  2>"$out/test_dlclose_fork.build.log" ||
  { echo "BUILD FAIL test_dlclose_fork"; fail=1; }

i=1
while [ "$i" -le "$REPEAT" ]; do
  run "test_cache.$i" "$out/test_cache"
  i=$((i + 1))
done
run test_cache.stats env OM_MEMORY_CACHE_STATS=1 "$out/test_cache"
run test_stale "$out/test_stale"
run test_alloc_failure "$out/test_alloc_failure"
run test_dlclose_fork "$out/test_dlclose_fork" "$out/libomcachetest.so"

if [ "$fail" -eq 0 ]; then
  echo "ALL PASS (outputs in $out)"
else
  echo "SOME TESTS FAILED (outputs in $out)"
fi
exit "$fail"
