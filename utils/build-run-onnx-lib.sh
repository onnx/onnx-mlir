# Build the run-onnx-lib utility
#
# When called without parameters, we build the tool for dynamically linking of
# a model. The model .so will be passed at runtime.
#
# When called with one parameter (a compiled model .o or .so), we build the
# tool with that model statically linked in.
#
# Assumptions:
# 1) script run in the onnx-mlir/build subdir.

if [ -z $ONNX_MLIR_HOME ]; then
  # The build path of onnx-mlir is not provided
  # ask git for the onnx-mlir top level dir
  ONNX_MLIR=$(git rev-parse --show-toplevel)
  if [ $(realpath $(pwd)) != $ONNX_MLIR/build ] ; then
    echo "Error: this script must be run from the build dir $ONNX_MLIR/build"
    exit 1
  fi
  ONNX_MLIR_BIN=$ONNX_MLIR/build/Debug/bin
  ONNX_MLIR_LIB=$ONNX_MLIR/build/Debug/lib
else
  # ONNX_MLIR_HOME should be onnx-mlir/build/Debug according to RunONNXModel.py
  ONNX_MLIR_BIN=$ONNX_MLIR_HOME/bin
  ONNX_MLIR_LIB=$ONNX_MLIR_HOME/lib
  ONNX_MLIR=$ONNX_MLIR_HOME/../..
fi

if [ "$#" -eq 0 ] ; then
  echo "Compiling run-onnx-lib for dynamically linked models passed at runtime"
  RUN_BIN=$ONNX_MLIR_BIN/run-onnx-lib
  STATIC_FLAG=0
elif [ "$#" -eq 1 ] ; then
  # Check model exists
  if [ -e $1 ] ; then
    echo "Compiling run-onnx-lib statically linked to model $1"
  else
    echo "Error: could not find model $1"
    exit 1
  fi
  RUN_BIN=$ONNX_MLIR_BIN/run-onnx-lib
  STATIC_FLAG=1
elif [ "$#" -eq 2 ] ; then
  if [ -e $1 ] ; then
    echo "Compiling run-onnx-lib statically linked to model $1"
  else
    echo "Error: could not find model $1"
    exit 1
  fi
  RUN_BIN=$2
  STATIC_FLAG=1
else
  echo "Error: pass zero, one, or two arguments (model.o and optional output path)"
  exit 1
fi

echo "Built binary will be created at $RUN_BIN"

# Source files: driver + debug-runtime helpers (OMDebugRuntime).
# SmallFPConversion is linked from its pre-built library to avoid passing a
# .c file to clang++ (which warns about treating C input as C++).
DRIVER_NAME=$ONNX_MLIR/utils/RunONNXLib.cpp
DEBUG_SRCS="$ONNX_MLIR/src/Runtime/OMTensorHelper.cpp \
            $ONNX_MLIR/src/Runtime/OMTensorListHelper.cpp"

# Include paths:
#   $ONNX_MLIR/include  — public headers (OnnxMlirRuntime.h, onnx-mlir/Runtime/*)
#   $ONNX_MLIR          — src-relative headers (src/Runtime/*.hpp, src/Support/*.h)
INCLUDES="-I $ONNX_MLIR/include -I $ONNX_MLIR"

if [ "$STATIC_FLAG" -eq 0 ] ; then
  # Dynamic case: no model object; link libcruntime.a to supply the C runtime
  # symbols (omTensorCreateWithOwnership, omTensorListCreate, etc.) that
  # OMDebugRuntime calls.
  g++ -g $DRIVER_NAME $DEBUG_SRCS -o $RUN_BIN -std=c++17 \
    -D LOAD_MODEL_STATICALLY=0 \
    $INCLUDES \
    -L $ONNX_MLIR_LIB -lcruntime -lOMSmallFPConversion \
    -lpthread -ldl &&
  echo "Success, built $(basename $RUN_BIN) (dynamic)"
else
  # Static case: model .so is linked against; embed an rpath so dyld can find
  # it at runtime regardless of the current working directory.
  MODEL_ABS=$(realpath "$1")
  MODEL_BASENAME=$(basename "$MODEL_ABS")
  g++ -g $DRIVER_NAME $DEBUG_SRCS -o $RUN_BIN -std=c++17 \
    -D LOAD_MODEL_STATICALLY=1 \
    $INCLUDES \
    -L $ONNX_MLIR_LIB -lOMSmallFPConversion \
    -lpthread -ldl "$MODEL_ABS" &&
  if [ $(uname -s) = Darwin ] ; then
    # The model's install name is typically just its basename (e.g. "model.so").
    # Rewrite it to the absolute path so dyld can locate it from any directory.
    install_name_tool -change "$MODEL_BASENAME" "$MODEL_ABS" "$RUN_BIN"
  fi &&
  echo "Success, built $(basename $RUN_BIN) (static, model=$MODEL_ABS)"
fi
