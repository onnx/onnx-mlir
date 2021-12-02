# SPDX-License-Identifier: Apache-2.0

add_subdirectory(jni)

# Create static libcruntime.a to be embedded in model.so to make model.so self contained.
# However, by default object code for static library is not compiled with -fPIC. Embedding
# such static library in a shared library can cause runtime failure on some architectures,
# such as z. So we override the default and explicitly compile with -fPIC.
add_onnx_mlir_library(cruntime STATIC
  OMIndexLookup.c
  OMInstrument.c
  OMRandomNormal.c
  OMTensor.c
  OMTensorList.c
  OnnxDataType.c

  EXCLUDE_FROM_OM_LIBS

  INCLUDE_DIRS PRIVATE
  ${ONNX_MLIR_SRC_ROOT}/include
  )
set_target_properties(cruntime
  PROPERTIES
  LANGUAGE C
  POSITION_INDEPENDENT_CODE TRUE
  )

add_onnx_mlir_library(OMTensorUtils
  OMIndexLookup.cpp
  OMInstrument.cpp
  OMRandomNormal.cpp
  OMTensor.cpp
  OMTensorList.cpp
  OnnxDataType.cpp

  EXCLUDE_FROM_OM_LIBS

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include
  )
set_target_properties(OMTensorUtils
  PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE
  )

add_onnx_mlir_library(ExecutionSession
  ExecutionSession.cpp

  EXCLUDE_FROM_OM_LIBS

  LINK_LIBS PUBLIC
  OMTensorUtils
  LLVMSupport
  )
set_target_properties(ExecutionSession
  PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE
  )

pybind11_add_module(PyRuntime PyExecutionSession.cpp)
target_link_libraries(PyRuntime
  PRIVATE
  ExecutionSession
  onnx
  )
