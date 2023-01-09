# This script behaves like onnx-mlir (same arguments and behavior) but
# with the necessary setup and execution environment to report heap usage.
# Use only on module level compiler passes or with multithreading disabled
# to avoid parallel writing to the log file.
#
# Example which writes heap usage to /tmp/resnet50.heap.log:
#
#  cd build
#  bash  ../utils/onnx-mlir-report-heap.sh \
#    test/backend/Debug/models/resnet50/model.onnx -o=/tmp/resnet50 \
#    --mlir-elide-elementsattrs-if-larger=1 \
#    --report-heap-before=constprop-onnx --report-heap-after=constprop-onnx
#
# Assumptions:
# 1) script is run in the onnx-mlir/build subdir
# 2) onnx-mlir executable is built with the CMAKE_BUILD_TYPE (Debug, Release,
#    RelWithDebInfo) in onnx-mlir/build/CMakeCache.txt
# 3) arguments include --report-heap-before and/or --report-heap-after

# ask git for the onnx-mlir top level dir
ONNX_MLIR=$(git rev-parse --show-toplevel)
if [ $(pwd) != $ONNX_MLIR/build ]; then
  echo "Error: this script must be run from the build dir $ONNX_MLIR/build"
  exit 1
fi
CMAKE_BUILD_TYPE=$(fgrep CMAKE_BUILD_TYPE:STRING= CMakeCache.txt | cut -d= -f2)
ONNX_MLIR_BIN=$CMAKE_BUILD_TYPE/bin

if [[ "$OSTYPE" == "darwin"* ]]; then
  # On MacOS onnx-mlir executes the heap command to report heap usage.
  # The heap command requires the MallocStackLogging and DYLD_INSERT_LIBRARIES
  # environment variables to be set, and onnx-mlir must be run in the debugger.
  export MallocStackLogging=1 DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib
  lldb -o run -o quit -- $ONNX_MLIR_BIN/onnx-mlir "$@"
  unset MallocStackLogging DYLD_INSERT_LIBRARIES
else
  # TODO: Support heap reporting on more operating systems.
  echo "report-heap is not supported for this OS currently" 1>&2
  exit 1
fi
