# Build the run-onnx-lib utility
#
# When called without parameters, we build the tool for dynamically linking of 
# a model. It will need to be passed at runtime.
#
# When called with one parameter, we build the tool for the model passed
# as a parameter. 
#
# Assumptions:
# 1) script run in the onnx-mlir/build subdir. 
# 2) llvm-project is built with all its libraries (needed to run the tool)/

cd ../..
TOP_DIR=`pwd`
cd $TOP_DIR/onnx-mlir/build

LLVM_PROJ_SRC=$TOP_DIR/llvm-project
LLVM_PROJ_BUILD=$LLVM_PROJ_SRC/build

ONNX_MLIR_SRC=$TOP_DIR/onnx-mlir
ONNX_MLIR_UTIL=$ONNX_MLIR_SRC/utils
ONNX_MLIR_BIN=$ONNX_MLIR_SRC/build/Debug/bin
DRIVERNAME=RunONNXLib.cpp
echo $ONNX_MLIR_SRC

if [ "$#" -eq 0 ] ; then
  echo "Compile run-onnx-lib for models passed at runtime"
  g++ $ONNX_MLIR_UTIL/$DRIVERNAME -o $ONNX_MLIR_BIN/run-onnx-lib -std=c++17 \
  -D LOAD_MODEL_STATICALLY=0 -I $LLVM_PROJ_SRC/llvm/include \
  -I $LLVM_PROJ_BUILD/include -I $ONNX_MLIR_SRC/include \
  -L $LLVM_PROJ_BUILD/lib -lLLVMSupport -lLLVMDemangle -lcurses -lpthread -ldl &&
  echo "  success, dynamically linked run-onnx-lib built in $ONNX_MLIR_BIN"
elif  [ "$#" -eq 1 ] ; then
  if [ -e $1 ] ; then
    echo "Compile run-onnx-lib for model $1"
    g++ $ONNX_MLIR_UTIL/$DRIVERNAME -o $ONNX_MLIR_BIN/run-onnx-lib -std=c++14 \
    -D LOAD_MODEL_STATICALLY=1 -I $LLVM_PROJ_SRC/llvm/include \
    -I $LLVM_PROJ_BUILD/include -I $ONNX_MLIR_SRC/include \
    -L $LLVM_PROJ_BUILD/lib -lLLVMSupport -lLLVMDemangle -lcurses -lpthread -ldl $1 \
      &&
    echo "  success, statically linked run-onnx-lib built in $ONNX_MLIR_BIN"
    echo ""
    echo "TO RUN: easiest is to cd into the directory of the model"
   else
     echo "Error: could not find model $1"
   fi
else
  echo "Error: pass either zero/one argument for dynamically/statically linked models"
fi
